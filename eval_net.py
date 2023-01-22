import torch
from torch import nn
import numpy as np
import torchvision
from network import *
from PIL import Image


def main(image_path):

    model = torch.load('/Users/nathanbailey/Documents/terrycrews/model.pth', map_location=torch.device('cpu'))

    pil_img = Image.open(image_path).convert('RGB')
    mean = torch.tensor([0.5028, 0.4605, 0.4351])
    std = torch.tensor([0.3113, 0.3018, 0.3064])

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])

    tensor_img = transforms(pil_img).unsqueeze(0)
    softmaxed_tensor = nn.functional.softmax(model(tensor_img), dim=1).detach().numpy()
    print(softmaxed_tensor)
    return torch.argmax(nn.functional.softmax(model(tensor_img), dim=1), dim=1).item(), softmaxed_tensor[0][0], softmaxed_tensor[0][1]
