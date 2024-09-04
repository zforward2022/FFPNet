import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

class Pattern_Weight(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(Pattern_Weight, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class Position_Weight(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(Pattern_Weight, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
 
class OCP(nn.Module):
    def __init__(self, in_planes = 128, reduction=16, kernel_size=3):
        super(OCP, self).__init__()
        self.ca1 = Pattern_Weight(in_planes, reduction)
        self.sa1 = Position_Weight(kernel_size)
        self.ca2 = Pattern_Weight(in_planes, reduction)
        self.sa2 = Position_Weight(kernel_size)
 
    def forward(self, x1, x2):
        x2 = x2 * self.ca1(x1) * self.sa1(x1)
        x1 = x1 * self.ca2(x2) * self.sa2(x2)
        return tuple([x1,x2])
