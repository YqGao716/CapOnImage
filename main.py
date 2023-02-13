from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.utils.data as data
from torch import optim
import torch.distributed as distributed

import dataset
import dist
from models import build_model
from train import train, test


def get_args_parser():
  parser = argparse.ArgumentParser("Set basic hyper-parameters", add_help=False)
  
  # Dataset specific
  parser.add_argument("--trn_name_file", type=str, default="")
  parser.add_argument("--val_name_file", type=str, default="")
  parser.add_argument("--anno_file", type=str, default="")
  parser.add_argument("--word2int_file", type=str, default="")
  parser.add_argument("--int2word_file", type=str, default="")
  parser.add_argument("--color2int_file", type=str, default="")
  parser.add_argument("--img_root", type=str, default="")
  parser.add_argument("--train_tasks", nargs="+", help="List of training tasks", default=["mlm"])
  parser.add_argument("--eval_tasks", nargs="+", help="List of evaluation tasks", default=["mlm"])

  # Training hyper-parameters
  parser.add_argument("--batch_size", default=128, type=int)
  parser.add_argument("--tst_batch_size", default=500, type=int)
  parser.add_argument("--num_epoch", default=50, type=int)
  parser.add_argument("--maximum_steps", default=200000, type=int)
  parser.add_argument("--save_iter", default=1000, type=int)
  parser.add_argument("--val_iter", default=1000, type=int)
  parser.add_argument("--monitor_iter", default=100, type=int)
  parser.add_argument("--save_per_epoch", default=False, type=bool)
  parser.add_argument("--val_per_epoch", default=False, type=bool)
  parser.add_argument("--warmup_steps", default=8000, type=int)
  parser.add_argument("--lr", default=1e-4, type=float)
  parser.add_argument("--lr_backbone", default=1e-4, type=float)
  parser.add_argument("--optim", default="adam_bert", type=str, choices=("adam", "adam_bert"))

  # Model parameters
  parser.add_argument("--vocab", type=int, default=0, help="Vocab size")
  parser.add_argument("--grid_size", type=int, default=512, help="Grid size of image")
  parser.add_argument("--num_layer", type=int, default=4, help="Layers of the cross-transformer")
  parser.add_argument("--d_model", type=int, default=512, help="Hidden size of the cross-transformer")
  parser.add_argument("--heads", type=int, default=8, help="Attention heads of the cross-transformer")
  parser.add_argument("--dropout", type=int, default=0.1, help="Dropout used in the cross-transformer")
  parser.add_argument("--decoding", type=str, default="greedy", choices=("greedy", "beam_search", "sample"))
  parser.add_argument("--max_words_in_sent", type=int, default=30, help="max_len of caption")
  parser.add_argument("--max_words_in_info", type=int, default=30, help="max_len of info")

  # Run specific
  parser.add_argument('--is_train', default=False, action='store_true')
  parser.add_argument('--is_finetune', default=False, action='store_true')
  parser.add_argument('--resume_file', default=None)
  parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
  parser.add_argument("--device", default="cuda", help="device to use for training / testing")
  parser.add_argument("--seed", default=12345, type=int)
  parser.add_argument("--num_workers", default=4, type=int)

  # Distributed training parameters
  parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
  parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
  parser.add_argument("--local_rank", default=0, type=int, help="the local rank of current gpu")
  return parser

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def build_optimizer(model_without_ddp, args):
  params_backbone, params_main = [], []
  for name, param in model_without_ddp.named_parameters():
    if param.requires_grad:
      if 'backbone' in name:
        params_backbone.append(param)
      else:
        params_main.append(param)
  param_opts = [
    {
      'params': params_main,
      'lr': args.lr,
    },
    {
      'params': params_backbone,
      'lr': args.lr_backbone,
    },
  ]
  if len(params_main) > 0:
    if args.optim == 'adam':
      optimizer = optim.Adam(param_opts, lr=args.lr)
    elif args.optim == 'adam_bert':
      optimizer = optim.Adam(param_opts, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    else:
      raise RuntimeError(f"Unsupported optimizer {args.optim}")
  else:
    optimizer = None
    print('no traiable parameters')
  return params_main+params_backbone, optimizer

def load_checkpoint(model, ckpt_file, args):
  if args.distributed:
    distributed.barrier()
  state_dicts = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
  num_resumed_vars = 0
  own_state_dict = model.state_dict()
  new_state_dict = {}
  for varname, varvalue in state_dicts.items():
    if varname in own_state_dict:
      new_state_dict[varname] = varvalue
      num_resumed_vars += 1
  own_state_dict.update(new_state_dict)
  model.load_state_dict(own_state_dict)
  print('number of resumed variables: %d'%num_resumed_vars)

def main(args):
  # Init distributed mode
  dist.init_distributed_mode(args)

  print(args)
  device = torch.device(args.device)
  output_dir = Path(args.output_dir)

  # fix the seed for reproducibility
  set_seeds(args.seed+dist.get_rank())

  # Build the model
  model, criterion = build_model(args)
  model.to(device)
  model_without_ddp = model

  if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

  # Build the optimizer
  params, optimizer = build_optimizer(model_without_ddp, args)
  num_params, num_weights = 0, 0
  for varname, varvalue in model_without_ddp.state_dict().items():
    print('%s, shape=%s, num:%d' % (
      varname, str(varvalue.size()), np.prod(varvalue.size())))
    num_params += 1
    num_weights += np.prod(varvalue.size())
  print('num params %d, num weights %d'%(num_params, num_weights))
  print('trainable: num params %d, num weights %d'%(
    len(params), sum([np.prod(param.size()) for param in params])))

  # Build the training and evaluation datasets
  if args.is_train:
    trn_reader, val_reader = {}, {}
    for task in args.train_tasks:
      trn_data = dataset.PretrainDataset(args.trn_name_file, args, is_train=True, task=task)

      if args.distributed:
        trn_sampler = data.DistributedSampler(trn_data)
      else:
        trn_sampler = torch.utils.data.RandomSampler(trn_data)

      batch_trn_sampler = torch.utils.data.BatchSampler(trn_sampler, args.batch_size, drop_last=True)
      r = 1
      if task == 'cap': # cap:itm = 3:1
        r = 3
      trn_reader[task] = (data.DataLoader(trn_data, batch_sampler=batch_trn_sampler, num_workers=args.num_workers), r)
    meta_loader = dataset.MetaLoader(trn_reader)

    for task in args.eval_tasks:
      val_data = dataset.PretrainDataset(args.val_name_file, args, is_train=False, task=task)
      sampler = data.DistributedSampler(val_data, shuffle=False) if args.distributed else data.SequentialSampler(val_data)
      val_reader[task] = data.DataLoader(val_data, batch_size=args.tst_batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)

    if args.resume_file:
      load_checkpoint(model_without_ddp, args.resume_file, args)

    train(model=model, criterion=criterion, optimizer=optimizer, device=device, trn_reader=meta_loader, val_reader=val_reader, args=args)
  
  # Make prediction with a pre-defined checkpoint
  else:
    tst_reader = {}
    for task in args.eval_tasks:
      tst_data = dataset.PretrainDataset(args.val_name_file, args, is_train=False, task=task)
      sampler = data.DistributedSampler(tst_data, shuffle=False) if args.distributed else data.SequentialSampler(tst_data)
      tst_reader[task] = data.DataLoader(tst_data, batch_size=args.tst_batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)

    model_str_scores = []
    is_first_eval = True
    model_files = {'predefined': args.resume_file}

    for measure_name, model_file in model_files.items():
      set_pred_dir = os.path.join(args.output_dir, 'pred')
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir, 
        os.path.splitext(os.path.basename(model_file))[0]+'.json')

      load_checkpoint(model_without_ddp, model_file, args)
      scores = test(model, criterion, device, tst_reader, tst_pred_file, args)

      if is_first_eval:
        score_names = scores.keys()
        model_str_scores.append(','.join(score_names))
        is_first_eval = False
        print(model_str_scores[-1])
      str_scores = [measure_name, os.path.basename(model_file)]
      for score_name in score_names:
        str_scores.append('%.4f'%(scores[score_name]))
      str_scores = ','.join(str_scores)
      print(str_scores)
      model_str_scores.append(str_scores)

    score_log_file = os.path.join(args.output_dir, 'pred', 'scores.csv')
    with open(score_log_file, 'w') as f:
      for str_scores in model_str_scores:
        print(str_scores, file=f)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Training and evaluation script", parents=[get_args_parser()])
  args = parser.parse_args()
  if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
  main(args)



