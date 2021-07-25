import torch
import torchvision
import pytorch_model_summary as pms
from torch import optim, nn
import numpy as np
from dataset import *
from network import *
from train import *
import sys

#Defining GPU

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU is: {}'.format(torch.cuda.get_device_name(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))))
sys.stdout.flush()
# transforms_main = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()])
# train_dataset = veri_dataset(data_dir='../triplet_network/VeRi', dataset_type='train', data_standard_transforms=transforms_main)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True)
# #Per channel means and standard deviations
# mean = torch.zeros(3).to(device)
# std = torch.zeros(3).to(device)
# for idx, batch in enumerate(trainloader):
#     image = batch['image'].to(device)
#     image_mean = torch.mean(image, dim=(0,2,3))
#     image_std = torch.std(image, dim=(0,2,3))
#     mean = torch.add(mean, image_mean) 
#     std = torch.add(std, image_std)

# mean = (mean/len(trainloader)).to('cpu')
# std = (std/len(trainloader)).to('cpu')

# print(mean)
# print(std)
mean = torch.tensor([0.4207, 0.4204, 0.4266])
std = torch.tensor([0.2119, 0.2110, 0.2126])
#Load the datasets
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
#Train dataset
#Old was 6, 4
train_data = torch.load('train_data.pt')
train_label = torch.load('train_label.pt')
valid_data = torch.load('valid_data.pt')
valid_label = torch.load('valid_label.pt')




# for param in model.backbone.parameters():
#     param.requires_grad = False

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

# model.backbone.last_linear.weight.requires_grad = True
# model.backbone.last_linear.bias.requires_grad = True


train_dataset = veri_triplet_train_dataset(data_dir='../triplet_network/VeRi', dataset_type = 'train', data_transforms=transforms)
batchsampler_t = HardBatchSampler(data_file_to_use=train_data, label_file_to_use=train_label, p=18, k=4)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batchsampler_t, num_workers=8, pin_memory=True)
#Valid Dataset
valid_dataset = veri_triplet_train_dataset(data_dir='../triplet_network/VeRi', dataset_type = 'valid', data_transforms=transforms)
batchsampler_v = HardBatchSampler(data_file_to_use=valid_data, label_file_to_use=valid_label, p=18, k=4)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_sampler=batchsampler_v, num_workers=8, pin_memory=True)

model = TripletNetwork(num_classes=575).to(device)

optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=0.1, momentum=0.9, weight_decay=5e-4)
classification_loss = nn.CrossEntropyLoss()
triplet_loss = nn.TripletMarginLoss(margin=1.3,p=2.0)
loss_param = 0.5
num_epochs = 200
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=3, min_lr=0.0001, threshold_mode='abs', threshold=1e-2, verbose=True)
model = train(model, num_epochs, optimizer, triplet_loss, classification_loss, loss_param, trainloader, validloader, device, patience=10, path_to_model='model.pt', scheduler=scheduler)


train_dataset = veri_triplet_train_dataset(data_dir='../triplet_network/VeRi', dataset_type = 'train', data_transforms=transforms)
batchsampler_t = HardBatchSampler(data_file_to_use=train_data, label_file_to_use=train_label, p=6, k=4)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batchsampler_t, num_workers=2, pin_memory=True)
valid_dataset = veri_triplet_train_dataset(data_dir='../triplet_network/VeRi', dataset_type = 'valid', data_transforms=transforms)
batchsampler_v = HardBatchSampler(data_file_to_use=valid_data, label_file_to_use=valid_label, p=6, k=4)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_sampler=batchsampler_v,pin_memory=True)

for param in model.parameters():
    param.requires_grad = True


optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
classification_loss = nn.CrossEntropyLoss()
triplet_loss = nn.TripletMarginLoss(margin=1.3,p=2.0)
loss_param = 0.5
num_epochs = 200
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=3, min_lr=0.00001, threshold_mode='abs', threshold=1e-2, verbose=True)
model = train(model, num_epochs, optimizer, triplet_loss, classification_loss, loss_param, trainloader, validloader, device, patience=10, path_to_model='model.pt', scheduler=scheduler)


model = TripletNetwork(num_classes=575).to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.to(device)
model.train()

optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
classification_loss = nn.CrossEntropyLoss()
triplet_loss = nn.TripletMarginLoss(margin=1.3,p=2.0)
loss_param = 0.5
num_epochs = 200
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=3, min_lr=0.00000001, threshold_mode='abs', threshold=1e-2, verbose=True)
model = train(model, num_epochs, optimizer, triplet_loss, classification_loss, loss_param, trainloader, validloader, device, patience=10, path_to_model='model.pt', scheduler=scheduler)


