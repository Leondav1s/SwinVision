import torch
import torch.nn as nn
import torch.nn.functional as F
from SOF_backbone.SOFbackbone import SOFBackbone
import numpy as np

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        self.features = SOFBackbone().cuda()
        self.features.load_state_dict(torch.load('SOF_backbone/SOFbackbone.pth'))
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        f1, f2, f3, f4, f5 = self.features(x)
        m1, m2, m3, m4, m5 = self.features(y)
        loss3 = torch.mean(torch.pow(f3-m3, 2))
        loss4 = torch.mean(torch.pow(f4-m4, 2))
        loss5 = torch.mean(torch.pow(f5-m5, 2))
        loss = loss3 + loss4 +loss5
        return loss