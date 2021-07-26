import enum
import torch
import torchvision
import pytorch_model_summary as pms
from torch import optim, nn
import numpy as np
from dataset import *
from network_mobile import *
from train import *
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sys.stdout.flush()

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()])

train_dataset = terry_dataset('images/', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)
mean = torch.zeros(3).to(device)
std = torch.zeros(3).to(device)
for idx, batch in enumerate(trainloader):
    image = batch[0].to(device)
    image_mean = torch.mean(image, dim=(0,2,3))
    image_std = torch.std(image, dim=(0,2,3))
    mean = torch.add(mean, image_mean) 
    std = torch.add(std, image_std)

mean = (mean/len(trainloader)).to('cpu')
std = (std/len(trainloader)).to('cpu')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

train_dataset = terry_dataset('images/', dataset_type='train', data_transforms=transforms)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)

valid_dataset = terry_dataset('images/', dataset_type='valid', data_transforms=transforms)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, pin_memory=True)

test_dataset = terry_dataset('images/', dataset_type='test', data_transforms=transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)


model = terry_crews_network().to(device)
optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=0.001, momentum=0.9, weight_decay=5e-4)
loss = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=3, min_lr=0.00000001, threshold_mode='abs', threshold=1e-2, verbose=True)

num_epochs = 100
# model = train(model, num_epochs, optimizer, loss, trainloader, validloader, device, scheduler=scheduler)
model.load_state_dict(torch.load('terry_net.pt'))
model.to(device)
test_loss = nn.CrossEntropyLoss()
test(model, testloader, test_loss, device)