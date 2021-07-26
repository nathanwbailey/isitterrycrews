import torch
from torch import nn
import torchvision

class terry_crews_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(24, 1),
        )

        self.classify = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Dropout(0.6),
            nn.Linear(8,2) 
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x



