from .model import PretrainModel
from .criterion import LabelSmoothingLoss, KLDLoss

import torch
import torch.nn as nn

def build_model(args):
  device = torch.device(args.device)
  
  model = PretrainModel(args)
  
  criterion = []
  xe = LabelSmoothingLoss(
    label_smoothing=0.1,
    tgt_vocab_size=args.vocab,
    ignore_index=1,
    device=device
    )
  xe.to(device)
  criterion.append(xe)

  classify = nn.CrossEntropyLoss()
  classify.to(device)
  criterion.append(classify)

  return model, criterion