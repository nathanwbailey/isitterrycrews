import torch
import torchvision

from network import *

'''
Evaluation metrics are mAP and CMC
gallery set=train set

for each query get rid of gallery images from the same camera
Then compute CMC and mAP
'''

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


model = TripletNetwork(num_classes=575)
x, y = model.load_state_dict(torch.load('trained_models/model.pt', map_location=torch.device('cpu')), strict=False)
model.classification = Identity()
torch.save(model, 'model.pth')



