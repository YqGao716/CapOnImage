import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

import numpy as np
import math
import time
import pdb

from .backbone import resnet50
from .transformer import Transformer
from .common import PositionEmbeddingLearned, PositionEmbeddingSine
from .common import gelu

    
class PretrainModel(nn.Module):
  def __init__(self, args):
    super(PretrainModel, self).__init__()

    self.backbone = resnet50(args.grid_size)
    self.vis_embed = nn.Linear(2048, args.d_model)
    self.spatial_pos_embed = PositionEmbeddingLearned(args)
    self.loc_embed = nn.Linear(args.d_model*2, args.d_model//2)
    self.ctx_embed = nn.Linear(args.d_model*4, args.d_model//2)
    self.word_embed = nn.Embedding(args.vocab, args.d_model)
    self.sequence_pos_embed = PositionEmbeddingSine(args)
    self.mode_embed = nn.Embedding(4, args.d_model)   # img, location, knowledge, caption

    # cross-encoder
    self.transformer = Transformer(args)

    # output layers
    self.logit = nn.Linear(args.d_model, args.vocab)
    self.logit.weight = self.word_embed.weight
    self.cls = nn.Linear(args.d_model, 2)
    self.color_cls = nn.Linear(args.d_model, 100)  # color vocab size

    self.dropout = nn.Dropout(args.dropout)
    self.device = torch.device(args.device)
    self.args = args
    self.init_weights()

  def init_weights(self,):
    for name, p in self.named_parameters():
      if p.dim() > 1 and 'backbone' not in name:
        nn.init.xavier_uniform_(p)

  def img_encode(self, img):
    """
      encode the image with backbone and add positional and modality embeddings
      return (batch_size, grid_size*grid_size, d_model)
    """
    grid_ft = self.backbone(img)     # (batch_size, chanel, grid_size, grid_size)
    pe = self.spatial_pos_embed(grid_ft)
    img_mod = self.mode_embed(torch.as_tensor([0], device=grid_ft.device)).unsqueeze(1)
    # map the image grid feature to a d_model sequence vector
    grid_ft = self.vis_embed(grid_ft.view(img.size(0), grid_ft.size(1), -1).permute(0, 2, 1))
    # make embeddings relatively larger
    grid_ft = grid_ft * math.sqrt(self.args.d_model)
    return self.dropout(grid_ft + pe + img_mod)

  def loc_encode(self, location, prev_loc, post_loc):
    """
      input: location (batch_size, 1, 4) x1,y1,x2,y2
      encode the location with positional embeddings and add the modality embeddings
      return (batch_size, 1, d_model)
    """
    loc_embed = torch.cat([self.spatial_pos_embed.col_embed(location[:,:,0]),\
                           self.spatial_pos_embed.row_embed(location[:,:,1]), \
                           self.spatial_pos_embed.col_embed(location[:,:,2]), \
                           self.spatial_pos_embed.row_embed(location[:,:,3])], dim=-1)

    prev_embed = torch.cat([self.spatial_pos_embed.col_embed(prev_loc[:,:,0]),\
                            self.spatial_pos_embed.row_embed(prev_loc[:,:,1]), \
                            self.spatial_pos_embed.col_embed(prev_loc[:,:,2]), \
                            self.spatial_pos_embed.row_embed(prev_loc[:,:,3])], dim=-1)

    post_embed = torch.cat([self.spatial_pos_embed.col_embed(post_loc[:,:,0]),\
                            self.spatial_pos_embed.row_embed(post_loc[:,:,1]), \
                            self.spatial_pos_embed.col_embed(post_loc[:,:,2]), \
                            self.spatial_pos_embed.row_embed(post_loc[:,:,3])], dim=-1)

    ctx_embed = self.ctx_embed(torch.cat([prev_embed, post_embed], dim=-1))
    loc_embed = self.loc_embed(loc_embed)
    pe = torch.cat([ctx_embed, loc_embed], dim=-1)
    loc_mod = self.mode_embed(torch.as_tensor([1], device=pe.device)).unsqueeze(1)
    return self.dropout(pe + loc_mod)

  def src_encode(self, src):
    """
      encode the info with word embeddings and add positional and modality embeddings
      return (batch_size, step, d_model)
    """
    src_embed = self.word_embed(src)
    pe = self.sequence_pos_embed(src_embed)
    info_mod = self.mode_embed(torch.as_tensor([2], device=src.device)).unsqueeze(1)
    # make embeddings relatively larger
    src_embed = src_embed * math.sqrt(self.args.d_model)
    return self.dropout(src_embed + pe + info_mod)

  def trg_encode(self, trg):
    """
      encode the caption with word embeddings and add positional and modality embeddings
      return (batch_size, step, d_model)
    """
    trg_embed = self.word_embed(trg)
    pe = self.sequence_pos_embed(trg_embed)
    cap_mod = self.mode_embed(torch.as_tensor([3], device=trg.device)).unsqueeze(1)
    # make embeddings relatively larger
    trg_embed = trg_embed * math.sqrt(self.args.d_model)
    return self.dropout(trg_embed + pe + cap_mod)

  def forward(self, img, location, prev_loc, post_loc, src, trg, img_mask, loc_mask, src_mask, trg_mask, task='cap', decoding=None):
    if decoding is not None:    # greedy or sample
      return self.sample(img, location, prev_loc, post_loc, src, img_mask, loc_mask, src_mask, decoding)
    else:    # train
      v_inputs = self.img_encode(img)
      l_inputs = self.loc_encode(location, prev_loc, post_loc)
      s_inputs = self.src_encode(src)
      t_inputs = self.trg_encode(trg)
      input = torch.cat([v_inputs, l_inputs, s_inputs, t_inputs], dim=1)

      if trg_mask is not None and trg_mask.size(1) != 1:
        full_fake = torch.zeros(trg.size(0),1,trg.size(-1)).bool().cuda()
        firmask = torch.cat([img_mask, loc_mask, src_mask, full_fake], dim=-1)
        firmask = firmask.repeat(1, v_inputs.size(1)+l_inputs.size(1)+s_inputs.size(1), 1)
        img_mask = img_mask.repeat(1, trg.size(1), 1)
        loc_mask = loc_mask.repeat(1, trg.size(1), 1)
        src_mask = src_mask.repeat(1, trg.size(1), 1)
        secmask = torch.cat([img_mask, loc_mask, src_mask, trg_mask], dim=-1)
        mask = torch.cat([firmask, secmask], dim=1)
      else:
        mask = torch.cat([img_mask, loc_mask, src_mask, trg_mask], dim=-1)

      e_outputs = self.transformer(input, mask)
      if task == 'itm':
        output = self.cls(gelu(e_outputs))
      elif task == 'color':
        output = self.color_cls(gelu(e_outputs))
      else:
        e_outputs = e_outputs[:,v_inputs.size(1)+l_inputs.size(1)+s_inputs.size(1):]
        output = self.logit(e_outputs)
      return output

  def sample(self, img, location, prev_loc, post_loc, src, img_mask, loc_mask, src_mask, decoding='greedy'):
    init_tok = 2
    batch_size = img.size(0)

    v_inputs = self.img_encode(img)
    l_inputs = self.loc_encode(location, prev_loc, post_loc)
    s_inputs = self.src_encode(src)
    input = torch.cat([v_inputs, l_inputs, s_inputs], dim=1)
    mask = torch.cat([img_mask, loc_mask, src_mask], dim=-1)
    e_outputs = self.transformer(input, mask, step=0)

    outputs = torch.ones(batch_size, 1).fill_(init_tok).long().cuda()
    for i in range(1, self.args.max_words_in_sent):
      trg_mask = self.nopeak_mask(i).repeat(batch_size,1,1)
      t_inputs = self.trg_encode(outputs)
      out = self.logit(self.transformer(t_inputs, trg_mask, step=i))
      if decoding == 'greedy':
        _, next_word = torch.max(out[:,-1], dim=1)
        next_word = next_word.unsqueeze(-1)
      else:
        removed_logits = self.top_k_top_p_filtering(out[:,-1], top_k=3)
        probs = F.softmax(removed_logits, dim=-1)
        next_word = torch.multinomial(probs, 1).cuda()
      outputs = torch.cat([outputs, next_word], dim=1)
    return outputs

  def top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
          logits: logits distribution shape (batch size, vocabulary size)
          if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
          if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
          Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
          Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
      top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
      # Remove all tokens with a probability less than the last token of the top-k
      indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
      logits[indices_to_remove] = filter_value

    if top_p < 1.0:
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

      # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
      sorted_indices_to_remove = cumulative_probs > top_p
      if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
      # Shift the indices to the right to keep also the first token above the threshold
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0

      # scatter sorted tensors to original indexing
      indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
      logits[indices_to_remove] = filter_value
    return logits  

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask
