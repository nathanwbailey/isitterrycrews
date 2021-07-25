import torch
from torch import nn
import torchvision
from senet import se_resnet50
import numpy as np
import pytorch_model_summary as pms
from early import pytorchtools
import sys

def calculate_accuracy(outputs, ground_truth):
    softmaxed_output = nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    num_correct = torch.sum(torch.eq(predictions, ground_truth)).item()
    return num_correct, ground_truth.size(0)

def train(model, num_epochs, optimizer, loss_function_1, loss_function_2, loss_param, trainloader, validloader, device, patience=10, path_to_model = 'model.pt', scheduler=None):
    print('Training Started')
    sys.stdout.flush()
    # early_stop_loss_best = np.Inf
    # patience = 0
    # best_model = None
    early_stop = pytorchtools.EarlyStopping(patience=patience, verbose=True, path=path_to_model)
    for epoch in range(1, num_epochs+1):
        triplet_train_loss = []
        classification_train_loss = []
        num_correct_train = 0
        num_examples_train = 0
        triplet_valid_loss = []
        classification_valid_loss = []
        num_correct_valid = 0
        num_examples_valid = 0
        total_train_loss = []
        total_validation_loss = []
        model.train()
        for batch in trainloader:
            # print('hi')
            # sys.stdout.flush()
            optimizer.zero_grad()
            anchor = batch['anchor'][0].to(device)
            anchor_label = batch['anchor'][1].to(device)
            positive = batch['positive'][0].to(device)
            positive_label = batch['positive'][1].to(device)
            negative = batch['negative'][0].to(device)
            negative_label = batch['negative'][1].to(device)
            anchor_emb, anchor_class = model(anchor)
            pos_emb, pos_class = model(positive)
            neg_emb, neg_class = model(negative)
            triplet_loss = loss_function_1(anchor_emb, pos_emb, neg_emb)
            ground_truth = torch.cat((anchor_label, positive_label, negative_label), dim=0)
            predictions = torch.cat((anchor_class, pos_class, neg_class), dim=0)
            classification_loss = loss_function_2(predictions, ground_truth)
            total_loss = torch.add(torch.mul(triplet_loss, loss_param), torch.mul(classification_loss, 1-loss_param))
            total_loss.backward()
            optimizer.step()
            triplet_train_loss.append(triplet_loss.item())
            classification_train_loss.append(classification_loss.item())
            total_train_loss.append(total_loss.item())
            num_corr, num_ex = calculate_accuracy(predictions, ground_truth)
            num_correct_train += num_corr
            num_examples_train += num_ex

        model.eval()
        with torch.no_grad():
            for batch in validloader:
                anchor = batch['anchor'][0].to(device)
                anchor_label = batch['anchor'][1].to(device)
                positive = batch['positive'][0].to(device)
                positive_label = batch['positive'][1].to(device)
                negative = batch['negative'][0].to(device)
                negative_label = batch['negative'][1].to(device)
                anchor_emb, anchor_class = model(anchor)
                pos_emb, pos_class = model(positive)
                neg_emb, neg_class = model(negative)
                triplet_loss = loss_function_1(anchor_emb, pos_emb, neg_emb)
                ground_truth = torch.cat((anchor_label, positive_label, negative_label), dim=0)
                predictions = torch.cat((anchor_class, pos_class, neg_class), dim=0)
                classification_loss = loss_function_2(predictions, ground_truth)
                total_loss = torch.add(torch.mul(triplet_loss, loss_param), torch.mul(classification_loss, 1-loss_param))
                triplet_valid_loss.append(triplet_loss.item())
                classification_valid_loss.append(classification_loss.item())
                total_validation_loss.append(total_loss.item())
                num_corr, num_ex = calculate_accuracy(predictions, ground_truth)
                num_correct_valid += num_corr
                num_examples_valid += num_ex

        print('Epoch: {}, Training Triplet Loss: {:4f}, Training Classification Loss: {:.4f}, Total Train Loss: {:.4f}, Training Accuracy: {:.4f}, Validation Triplet Loss: {:4f}, Validation Classification Loss: {:.4f}, Total Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch, np.mean(triplet_train_loss), np.mean(classification_train_loss), np.mean(total_train_loss), num_correct_train/num_examples_train, np.mean(triplet_valid_loss), np.mean(classification_valid_loss), np.mean(total_validation_loss), num_correct_valid/num_examples_valid))

        if scheduler:
            scheduler.step(np.mean(total_validation_loss))
       

        #Early stopping and model saving
        #Adpated from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
        #Saving of the model every epoch:
        if early_stop.counter == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'validation_loss': np.mean(total_train_loss),
                'train_loss': np.mean(total_validation_loss)
                }, 'full_model.tar')
        

        early_stop_loss = np.mean(total_validation_loss)
        early_stop(early_stop_loss, model)

        if early_stop.early_stop:
            print('Early Stopping at Epoch: {}'.format(epoch))
            break
        sys.stdout.flush()
        
    #Loading the best model
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    return model 





