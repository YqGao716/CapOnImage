import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pdb


class resnet50(nn.Module):
  def __init__(self, grid_size=None, global_pool=False):
    super(resnet50, self).__init__()
    self.cnn = torchvision.models.resnet50(pretrained=True)
    self.grid_size = grid_size
    self.global_pool = global_pool

  def forward(self, x):
    x = self.cnn.conv1(x)
    x = self.cnn.bn1(x)
    x = self.cnn.relu(x)
    x = self.cnn.maxpool(x)

    x = self.cnn.layer1(x)
    x = self.cnn.layer2(x)
    res4f_relu = self.cnn.layer3(x)
    res5e_relu = self.cnn.layer4(res4f_relu)

    if self.global_pool:
      globalp = F.adaptive_avg_pool2d(res5e_relu, (1, 1))
      globalp = globalp.view(globalp.size(0), -1)
      return globalp
    else:
      if self.grid_size is not None:
        avgp = F.adaptive_avg_pool2d(res5e_relu, (self.grid_size, self.grid_size))
      else:
        avgp = res5e_relu
      return avgp

