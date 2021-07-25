import torch
from torch import nn
import torchvision

class terry_crews_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier[3] = nn.Linear(1024, 2)
    
    def forward(self, x):
        return self.backbone(x)



