import torch.nn as nn
from .common import *


class EncoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout=0.1):
    super().__init__()
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.ff = FeedForward(d_model, dropout=dropout)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)

  def forward(self, x, mask, layer_cache=None):
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn(x2,x2,x2,mask, layer_cache=layer_cache))
    x2 = self.norm_2(x)
    x = x + self.dropout_2(self.ff(x2))
    return x


class Transformer(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.N = args.num_layer
    self.layers = get_clones(EncoderLayer(args.d_model, args.heads, args.dropout), args.num_layer)
    self.norm = Norm(args.d_model)
    self.cache = None

  def _init_cache(self):
    self.cache = {}
    for i in range(self.N):
      self.cache['layer_%d'%i] = {
        'self_keys': None,
        'self_values': None,
        'self_masks': None,
      }

  def forward(self, x, mask, step=None):
    if step == 0:
      self._init_cache()
    for i in range(self.N):
      layer_cache = self.cache['layer_%d'%i] if step is not None else None
      x = self.layers[i](x, mask, layer_cache=layer_cache)
    return self.norm(x)

