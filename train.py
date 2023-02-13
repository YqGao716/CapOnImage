from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import metrics
import dist

from tqdm import tqdm
import json
import numpy as np
import datetime
import os
import pdb

def pretty_print_metrics(prefix, metrics):
  metric_str = []
  for measure, score in metrics.items():
    metric_str.append('%s %.4f'%(measure, score))
  metric_str = ' '.join(metric_str)
  print('%s: %s' % (prefix, metric_str))

def save_checkpoint(model, ckpt_file):
  state_dicts = {}
  for varname, varvalue in model.state_dict().items():
    state_dicts[varname] = varvalue.cpu()
  torch.save(state_dicts, ckpt_file)

def train_start(model):
  model.train()
  torch.set_grad_enabled(True)

def eval_start(model):
  model.eval()
  torch.set_grad_enabled(False)

def nopeak_mask(size, device):
  np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
  np_mask =  Variable(torch.from_numpy(np_mask) == 0).to(device)
  return np_mask

def create_masks(trg, task=None):
  trg_mask = (trg != 1).unsqueeze(-2)
  if task == 'cap':
    size = trg.size(1) # get seq_len for matrix
    np_mask = nopeak_mask(size, device=trg.device)
    trg_mask = trg_mask & np_mask
  return trg_mask

def evaluate(model, criterion, device, tst_reader, args, decoding='greedy'):
  pred_sents, pred_colors, names = [], [], []
  score = {}
  n_correct, n_word = 0, 0
  for task in tst_reader:
    cur_reader = tst_reader[task]
    for batch_data in tqdm(cur_reader):
      images = batch_data['images'].to(device)
      locations = batch_data['locations'].to(device)
      prev_loc = batch_data['prev_locs'].to(device)
      post_loc = batch_data['post_locs'].to(device)
      batch_size = images.size(0)
      img_mask = torch.as_tensor([1] * args.grid_size**2, dtype=torch.bool, device=device) \
                .repeat(batch_size, 1) \
                .unsqueeze(1)
      loc_mask = torch.as_tensor([1] * 1, dtype=torch.bool, device=device) \
                .repeat(batch_size, 1) \
                .unsqueeze(1)
      src = batch_data['info_ids'].to(device)
      src = src[:,:max(batch_data['info_lens'])]
      src_mask = create_masks(src)

      if task == 'cap':
        output = model(images, locations, prev_loc, post_loc, src, None, img_mask, loc_mask, src_mask, None, task=task, decoding=decoding)
        captions = cur_reader.dataset.int2sent(output.detach())
        pred_sents.extend(captions)
        names.extend(batch_data['names'])
        if len(pred_sents) == batch_size:
          print(names[:10])
          print(pred_sents[:10])

      elif task == 'itm':
        trg = batch_data['caption_ids'].to(device)
        trg = trg[:,:max(batch_data['caption_lens'])]
        trg_mask = create_masks(trg, task)
        output = model(images, locations, prev_loc, post_loc, src, trg, img_mask, loc_mask, src_mask, trg_mask, task=task)
        target = batch_data['align_label'].to(device)
        output = output[:,args.grid_size**2+1+src.size(1)]
        pred = output.max(1, keepdim=True)[1]
        n_correct += float(pred.eq(target.view_as(pred)).cpu().float().sum())
        n_word += output.size(0)

      elif task == 'color':
        trg = batch_data['caption_ids'].to(device)
        trg = trg[:,:1]
        trg_mask = create_masks(trg, task)
        output = model(images, locations, prev_loc, post_loc, src, trg, img_mask, loc_mask, src_mask, trg_mask, task=task)
        target = batch_data['color_label'].to(device)
        output = output[:,args.grid_size**2+1+src.size(1)]
        pred = output.max(1, keepdim=True)[1]
        pred_colors.extend(pred.squeeze(-1).detach().cpu().numpy().tolist())
        n_correct += float(pred.eq(target.view_as(pred)).cpu().float().sum())
        n_word += output.size(0)

    if task == 'cap':
      score.update(metrics.compute(pred_sents, cur_reader.dataset.loc2cap, names))
      #score.update({'cap':0.0})
    else:
      score.update({task+'_avg_acc':n_correct/n_word})
    prediction = {'names': names, "captions": pred_sents, "colors": pred_colors}

    if args.distributed:
      torch.distributed.barrier()
      score = dist.reduce_dict(score, device)   # not strictly consistent with final corpus evaluation
      prediction = dist.all_gather(prediction)
  return score, prediction

def forward_loss(model, criterion, device, batch_data, task, args, step=None):
  images = batch_data['images'].to(device)
  locations = batch_data['locations'].to(device)
  prev_loc = batch_data['prev_locs'].to(device)
  post_loc = batch_data['post_locs'].to(device)
  batch_size = images.size(0)
  img_mask = torch.as_tensor([1] * args.grid_size**2, dtype=torch.bool, device=device) \
            .repeat(batch_size, 1) \
            .unsqueeze(1)
  loc_mask = torch.as_tensor([1] * 1, dtype=torch.bool, device=device) \
            .repeat(batch_size, 1) \
            .unsqueeze(1)
  src = batch_data['info_ids'].to(device)
  src = src[:,:max(batch_data['info_lens'])]
  src_mask = create_masks(src)
  trg = batch_data['caption_ids'].to(device)
  trg = trg[:,:max(batch_data['caption_lens'])]

  if task == 'itm':
    trg_mask = create_masks(trg, task)
    outputs = model(images, locations, prev_loc, post_loc, src, trg, img_mask, loc_mask, src_mask, trg_mask, task=task)
    loss = criterion[1](outputs[:,args.grid_size**2+1+src.size(1)], batch_data['align_label'].cuda())
  elif task == 'color':
    trg = trg[:, :1]
    trg_mask = create_masks(trg, task)
    outputs = model(images, locations, prev_loc, post_loc, src, trg, img_mask, loc_mask, src_mask, trg_mask, task=task)
    loss = criterion[1](outputs[:,args.grid_size**2+1+src.size(1)], batch_data['color_label'].cuda())
  else:
    trg_input = trg[:, :-1]
    trg_mask = create_masks(trg_input, task)
    outputs = model(images, locations, prev_loc, post_loc, src, trg_input, img_mask, loc_mask, src_mask, trg_mask, task=task)
    outputs = nn.LogSoftmax(dim=-1)(outputs)
    ys = trg[:, 1:].contiguous().view(-1)
    norm = trg[:, 1:].ne(1).sum().item()
    loss = criterion[0](outputs.view(-1, outputs.size(-1)), ys, norm)

  return loss

def train_one_batch(model, criterion, optimizer, device, batch_data, task, step, args):
  optimizer.zero_grad()
  loss = forward_loss(model, criterion, device, batch_data, task, args, step=step)
  loss.backward()

  if args.optim == 'adam_bert':
    lr = args.d_model ** (-0.5) * min(step ** (-0.5), step * args.warmup_steps**(-1.5))
    # only adjust the params_main
    optimizer.param_groups[0]['lr'] = lr
    
  elif args.is_finetune and step in [10000, 20000, 30000, 40000, 50000]:
    # adjust both the params_main and backbone
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
    optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / 2

  optimizer.step()

  loss_value = loss.data.item()
  if args.monitor_iter > 0 and step % args.monitor_iter == 0:
    print('%s\ttrn step %d %s: %.4f' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, 'loss', loss_value))
  return {'loss': loss_value}

def train_one_epoch(model, criterion, optimizer, device, trn_reader, val_reader, step, args):
  train_start(model)
  avg_loss, n_batches = {}, {}
  for task, batch_data in trn_reader:
    loss = train_one_batch(model, criterion, optimizer, device, batch_data, task, step, args)
    for loss_key, loss_value in loss.items():
      avg_loss.setdefault(loss_key, 0)
      n_batches.setdefault(loss_key, 0)
      avg_loss[loss_key] += loss_value
      n_batches[loss_key] += 1
    step += 1

    if args.maximum_steps > 0 and step >= args.maximum_steps:
      exit()

    if args.save_iter > 0 and step % args.save_iter == 0:
      model_without_ddp = model.module if args.distributed else model
      if args.distributed:
        dist.save_on_master(model_without_ddp.state_dict(), os.path.join(args.output_dir, 'model', 'step.%d.th'%step))
      else:
        save_checkpoint(model_without_ddp, os.path.join(args.output_dir, 'model', 'step.%d.th'%step))
    
    if args.val_iter > 0 and step % args.val_iter == 0:
      metrics = validate(model, criterion, device, val_reader, args)
      with open(os.path.join(args.output_dir, 'log', 'val.step.%d.json'%step), 'w') as f:
        json.dump(metrics, f, indent=2)
      pretty_print_metrics('%s\tval step %d'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step), metrics)
      train_start(model)

  for loss_key, loss_value in avg_loss.items():
    avg_loss[loss_key] = loss_value / n_batches[loss_key]
  return avg_loss, step

def validate(model, criterion, device, val_reader, args):
  eval_start(model)
  metrics, _ = evaluate(model, criterion, device, val_reader, args, decoding=args.decoding)
  return metrics

def train(model, criterion, optimizer, device, trn_reader, val_reader, args):
  print("*******************Start training*******************")
  step = 1
  for epoch in range(args.num_epoch):
    if args.distributed:
      trn_reader.set_sampler_epoch(epoch)
    avg_loss, step = train_one_epoch(model, criterion, optimizer, device, trn_reader, val_reader, step, args)
    pretty_print_metrics('epoch (%d/%d) trn'%(epoch, args.num_epoch), avg_loss)
    if epoch % 2 != -1:
      if args.save_per_epoch:
        model_without_ddp = model.module if args.distributed else model
        if args.distributed:
          dist.save_on_master(model_without_ddp.state_dict(), os.path.join(args.output_dir, 'model', 'epoch.%d.th'%epoch))
        else:
          save_on_master(model_without_ddp, os.path.join(args.output_dir, 'model', 'epoch.%d.th'%epoch))
      if args.val_per_epoch:
        metrics = validate(model, criterion, device, val_reader, args)
        with open(os.path.join(args.output_dir, 'log', 
          'val.epoch.%d.step.%d.json'%(epoch, step)), 'w') as f:
          json.dump(metrics, f, indent=2)
        pretty_print_metrics('epoch (%d/%d) val' % (epoch, args.num_epoch), metrics)
        torch.cuda.empty_cache()

def test(model, criterion, device, tst_reader, tst_pred_file, args):
  eval_start(model)
  metrics, pred_data = evaluate(model, criterion, device, tst_reader, args, decoding=args.decoding)
  with open(tst_pred_file, 'w') as f:
    json.dump(pred_data, f)
  return metrics

