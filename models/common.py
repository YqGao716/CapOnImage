import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np
import copy
import math


class PositionEmbeddingSine(nn.Module):
  def __init__(self, args, max_seq_len=200):
    super().__init__()
    self.d_model = args.d_model
    # create constant 'pe' matrix with values dependant on pos and i
    pe = torch.zeros(max_seq_len, args.d_model)
    for pos in range(max_seq_len):
      for i in range(0, args.d_model, 2):
        pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/args.d_model)))
        pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/args.d_model)))
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    seq_len = x.size(1)
    pe = Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
    return pe


class PositionEmbeddingLearned(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.row_embed = nn.Embedding(args.grid_size, args.d_model // 2)
    self.col_embed = nn.Embedding(args.grid_size, args.d_model // 2)
    self.args = args
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.uniform_(self.row_embed.weight)
    nn.init.uniform_(self.col_embed.weight)

  def forward(self, x):
    """
      input: (batch_size, chanel, grid_size, grid_size)
      output: (batch_size, grid_size*grid_size, chanel)
    """
    h, w = x.shape[-2:]
    i = torch.arange(w, device=x.device)
    j = torch.arange(h, device=x.device)
    x_emb = self.col_embed(i)
    y_emb = self.row_embed(j)
    pe = torch.cat([x_emb.unsqueeze(0).repeat(h, 1, 1), y_emb.unsqueeze(1).repeat(1, w, 1)], dim=-1) \
        .unsqueeze(0) \
        .repeat(x.shape[0], 1, 1, 1) \
        .view(x.shape[0], -1, self.args.d_model)
    return pe


class Norm(nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()
    self.size = d_model   
    # create two learnable parameters to calibrate normalisation
    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    self.eps = eps
    
  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
      / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
  scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9)
  scores = F.softmax(scores, dim=-1) 
  if dropout is not None:
    scores = dropout(scores)     
  output = torch.matmul(scores, v)
  return output

class MultiHeadAttention(nn.Module):
  def __init__(self, heads, d_model, dropout=0.1):
    super().__init__() 
    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads
    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)  
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

  def shape(self, x):
    bs = x.size(0)
    return x.view(bs, -1, self.h, self.d_k).transpose(1,2)
    
  def forward(self, q, k, v, mask=None, layer_cache=None):
    if layer_cache is not None:
      k = self.shape(self.k_linear(k))
      v = self.shape(self.v_linear(v))
      if layer_cache['self_keys'] is not None:
        if layer_cache['self_keys'].size(0) != k.size(0):
          beam_size = k.size(0) // layer_cache['self_keys'].size(0)
          context = layer_cache['self_keys']
          context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(-3), context.size(-2), context.size(-1))
          layer_cache['self_keys'] = context.contiguous().view(-1, context.size(-3), context.size(-2), context.size(-1))
        k = torch.cat((layer_cache['self_keys'], k), dim=2)
      else:
        layer_cache['self_keys'] = k[:,:,:-1]

      if layer_cache['self_values'] is not None:
        if layer_cache['self_values'].size(0) != v.size(0):
          beam_size = v.size(0) // layer_cache['self_values'].size(0)
          context = layer_cache['self_values']
          context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(-3), context.size(-2), context.size(-1))
          layer_cache['self_values'] = context.contiguous().view(-1, context.size(-3), context.size(-2), context.size(-1))
        v = torch.cat((layer_cache['self_values'], v), dim=2)
      else:
        layer_cache['self_values'] = v[:,:,:-1]
        layer_cache['self_masks'] = mask[:,:,:-1]
    else:
      k = self.shape(self.k_linear(k))
      v = self.shape(self.v_linear(v))

    bs = q.size(0) 
    q =  self.shape(self.q_linear(q))
    if mask.size(-1) != k.size(-2):
      if layer_cache['self_masks'].size(0) != mask.size(0):
        beam_size = mask.size(0) // layer_cache['self_masks'].size(0)
        context = layer_cache['self_masks']
        context = context.unsqueeze(1).expand(context.size(0), beam_size, context.size(-2), context.size(-1))
        layer_cache['self_masks'] = context.contiguous().view(-1, context.size(-2), context.size(-1))
      mask = torch.cat([layer_cache['self_masks'].repeat(1,mask.size(1),1), mask], dim=-1)
    # calculate attention using function we will define next
    scores = attention(q, k, v, self.d_k, mask, self.dropout)
    # concatenate heads and put through final linear layer
    concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
    output = self.out(concat)
    return output


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff=2048, dropout=0.1):
    super().__init__() 
    # We set d_ff as a default to 2048
    self.linear_1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)
    
  def forward(self, x):
    #x = self.dropout(F.relu(self.linear_1(x), inplace=True))
    x = self.dropout(gelu(self.linear_1(x)))
    x = self.linear_2(x)
    return x


def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
