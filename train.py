import torch
from early import pytorchtools
import sys

def train(model, num_epochs, optimizer, loss_function, trainloader, validloader, patience=10, path_to_model = 'model.pth', scheduler=None):
    print('Training Started')
    sys.stdout.flush()
    early_stop = pytorchtools.EarlyStopping(patience=)
