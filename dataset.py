from __future__ import print_function
from __future__ import division

import numpy as np
import json
import h5py
import random
import math
import os
import cv2
import pdb
from PIL import Image
os.environ['OPENCV_IO_ENABLE_JASPER']= 'true'

import torch.utils.data
from torchvision import transforms

UNK, PAD, SOS, EOS = 0, 1, 2, 3


class PretrainDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, args, is_train=False, task='cap', _logger=None):
    super(PretrainDataset, self).__init__()

    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.names = np.load(name_file)
    self.num_ft = len(self.names)
    self.print_fn('image size %d' % self.num_ft)
    
    self.ref_captions = json.load(open(args.anno_file))

    self.captions, self.colors, self.locations, self.cap2img = [], [], [], []
    self.loc2cap = {}
    for name in self.names:
      self.captions.extend(self.ref_captions[name]['ocr_txts'])
      self.colors.extend(self.ref_captions[name]['ocr_colors'])
      self.locations.extend(self.ref_captions[name]['ocr_locs'])
      self.cap2img.extend([name] * len(self.ref_captions[name]['ocr_locs']))
      for i in range(len(self.ref_captions[name]['ocr_locs'])):
        loc = self.ref_captions[name]['ocr_locs'][i]
        self.loc2cap[name+'_'+loc] = self.ref_captions[name]['ocr_txts'][i]
    self.num_loc = len(self.locations)
    self.print_fn('location size %d' % self.num_loc)
    
    self.stoi = json.load(open(args.word2int_file))
    self.itos = json.load(open(args.int2word_file))
    self.ctoi = json.load(open(args.color2int_file))
    self.img_root = args.img_root
    self.max_words_in_sent = args.max_words_in_sent
    self.max_words_in_info = args.max_words_in_info
    self.is_train = is_train
    self.task = task
    self.args = args

    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

  def pad_sent(self, x):
    max_len = self.max_words_in_sent
    # padding and adding <sos> and <eos>
    padded = [SOS] + x[:max_len-1] + [EOS] + [PAD] * max(0, max_len - len(x) - 2)
    length = min(len(x)+2, max_len)
    # clip with max_len
    padded = padded[:max_len]
    return np.array(padded), length

  def pad_info(self, x):
    max_len = self.max_words_in_info
    x = x[:max_len] + [PAD] * max(0, max_len - len(x))
    lens = min(len(x), max_len)
    return np.array(x), lens

  def sent2int(self, str_sent):
    int_sent = [self.stoi.get(w, UNK) for w in str_sent]
    return int_sent

  def int2sent(self, batch):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    batch = [[self.itos.get(str(ind), '<unk>') for ind in ex] for ex in batch] # denumericalize
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence
    batch = [trim(ex, '<eos>') for ex in batch] # trim past first eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>', '<mask>')
    batch = ["".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
    return batch

  def load_img(self, path):
    img = cv2.imread(path)
    h, w, c = img.shape
    if c == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif c == 1:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

  def process_img(self, img, all_locs=None, mask_ocr=False, ratio=False):
    """
      img.size: H x W x C
      all_locs: List of all the orc locations in the image
      x1, y1: left corner
      x2, y2: right corner
    """
    h, w, c = img.shape
    if mask_ocr and all_locs is not None:
      for ocr_loc in all_locs:
        if ratio:
          ocr_loc = np.array(ocr_loc.split('_')).astype(np.float32)
          x1, y1, x2, y2 = ocr_loc
          x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        else:
          ocr_loc = np.array(ocr_loc.split('_')).astype(np.int64)
          x1, y1, x2, y2 = ocr_loc

        img[y1:y2+1,x1:x2+1,:] = 0

    # resize the img to 256 x 256 x 3 and transforms
    img = cv2.resize(img, (self.args.grid_size*32, self.args.grid_size*32))
    img = Image.fromarray(img)
    img = self.transform(img)
    return img

  def loc_to_grid(self, loc, h, w, ratio=False):
    # (x1_y1_x2_y2)
    x1, y1, x2, y2 = loc.split('_')
    if not ratio:
      x1, y1, x2, y2 = int(x1)/w, int(y1)/h, int(x2)/w, int(y2)/h
    else:
      x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1_id = min(self.args.grid_size-1,round(x1*self.args.grid_size))
    y1_id = min(self.args.grid_size-1,round(y1*self.args.grid_size))
    x2_id = min(self.args.grid_size-1,round(x2*self.args.grid_size))
    y2_id = min(self.args.grid_size-1,round(y2*self.args.grid_size))
    return np.array([[x1_id, y1_id, x2_id, y2_id]], dtype=np.int64)

  def getNeighbour(self, key, cur_loc):
    cur_idx = self.ref_captions[key]['ocr_locs'].index(cur_loc)
    prev_loc, post_loc = None, None
    if cur_idx == 0:
      prev_loc = '0_0_0_0'
    if cur_idx == len(self.ref_captions[key]['ocr_locs'])-1:
      post_loc = '0_0_0_0'

    if prev_loc is None:
      prev_loc = self.ref_captions[key]['ocr_locs'][cur_idx-1]
    if post_loc is None:
      post_loc = self.ref_captions[key]['ocr_locs'][cur_idx+1]
    return prev_loc, post_loc

  def __len__(self):
    return self.num_loc

  def __getitem__(self, idx):
    outs = {}
    name = self.cap2img[idx]

    img = self.load_img(os.path.join(self.img_root, name+'.png'))
    h, w, c = img.shape
    outs['images'] = self.process_img(img, self.ref_captions[name]['all_locs'], mask_ocr=True, ratio=True)

    info_id, info_len = self.pad_info(self.sent2int(self.ref_captions[name]['info']))
    outs['info_ids'] = info_id
    outs['info_lens'] = info_len

    loc = self.locations[idx]
    prev_loc, post_loc = self.getNeighbour(name, loc)

    cap = self.captions[idx]
    color = self.ctoi.get(self.colors[idx], 0)
      
    if self.task == 'itm':
      rep_prob = random.random()
      if rep_prob < 0.6:
        rep_prob /= 0.6
        align_label = 0

        if rep_prob < 0.3:
          # subtask I: 30% replace with other random ocr_txt in different image
          cap = random.choice(self.captions)
        elif rep_prob < 0.6:
          # subtask II: 30% replace with other valid locations in the same image
          old_loc = loc
          loc = random.choice(self.ref_captions[name]['ocr_locs'])
          prev_loc, post_loc = self.getNeighbour(name, loc)
          if old_loc == loc:
            align_label = 1
        else:
          # subtask III: 40% replace with the ocr_txt in neighbour locations
          if rep_prob < 0.8:
            tmp = '_'.join([name, prev_loc])
          else:
            tmp = '_'.join([name, post_loc])
          if tmp in self.loc2cap:
            cap = self.loc2cap[tmp]
          else:
            align_label = 1

      else:
        align_label = 1
      outs['align_label'] = align_label
    
    outs['locations'] = self.loc_to_grid(loc, h, w, ratio=True)
    outs['prev_locs'] = self.loc_to_grid(prev_loc, h, w, ratio=True)
    outs['post_locs'] = self.loc_to_grid(post_loc, h, w, ratio=True)
    caption_id, caption_len = self.pad_sent(self.sent2int(cap))
    outs['caption_ids'] = caption_id
    outs['caption_lens'] = caption_len
    outs['ref_captions'] = cap
    outs['color_label'] = color
    outs['names'] = name+'_'+loc
    return outs


class MetaLoader(object):
  """ wraps multiple data loaders """
  def __init__(self, loaders, accum_steps=1):
    assert isinstance(loaders, dict)
    self.name2loader = {}
    self.name2iter = {}
    self.sampling_pools = []
    for n, l in loaders.items():
      if isinstance(l, tuple):
        l, r = l
      elif isinstance(l, torch.utils.data.DataLoader):
        r = 1
      else:
        raise ValueError()
      self.name2loader[n] = l
      self.name2iter[n] = iter(l)
      self.sampling_pools.extend([n]*r)
    self.accum_steps = accum_steps
    self.loaders = loaders
    self.step = 0

  def set_sampler_epoch(self, epoch):
    for n, l in self.loaders.items():
      if isinstance(l, tuple):
        l, r = l
      l.batch_sampler.sampler.set_epoch(epoch)

  def __iter__(self):
    """ this iterator will run indefinitely """
    task = self.sampling_pools[0]
    while True:
      if self.step % self.accum_steps == 0:
        task = random.choice(self.sampling_pools)
        self.step += 1
        iter_ = self.name2iter[task]
        try:
          batch = next(iter_)
        except StopIteration:
          iter_ = iter(self.name2loader[task])
          batch = next(iter_)
          self.name2iter[task] = iter_

      yield task, batch


