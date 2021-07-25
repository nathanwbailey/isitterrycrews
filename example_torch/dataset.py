import torch
import torchvision
import xml.etree.ElementTree as ET 
import random
import numpy as np
from PIL import Image
from torch.nn import functional
from itertools import chain
from sklearn.utils import shuffle

'''
Custom triplet Mining batch sampler
Inherit the batch sampler and override methods to sample batches
Hence can be compatible with the dataloader
Give the indexes to sample from the dataset
Adapted from: https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
'''

class HardBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_file_to_use, label_file_to_use, p=18, k=4):
        self.data_file = data_file_to_use
        self.labels = label_file_to_use.tolist()
        self.p = p
        self.k = k
        self.batch_size = p*k

    def get_distance(self, image_a, image_b):
        image_a = torch.flatten(image_a)
        image_b = torch.flatten(image_b)
        image_a = image_a[None, :]
        image_b = image_b[None, :]
        return torch.nn.functional.pairwise_distance(image_a, image_b)

    def get_images_from_dataset(self, label, num):
        idxs = [i for i, data in enumerate(self.labels) if data == label]
        idxs = random.choices(idxs, k=num)
        return idxs

    def __iter__(self):
        for i in range(self.data_file.size(0) // self.batch_size):
            indexes = []
            random_labels = random.choices(self.labels, k=self.p)
            for label in random_labels:
                indexes.append(self.get_images_from_dataset(label, self.k))
            idxs = []
            for i in indexes:
                neg_data = indexes.copy()
                neg_data.remove(i)
                neg_data = list(chain.from_iterable(neg_data))
                for j in i:
                    image_j = self.data_file[j]
                    pos_dist = 0.0
                    neg_dist = np.Inf
                    pos_index, neg_index = 0, 0
                    pos_data = i.copy()
                    pos_data.remove(j)
                    for pos_idx in pos_data:
                        pos_image = self.data_file[pos_idx]
                        distance = self.get_distance(image_j, pos_image)
                        if pos_dist < distance:
                            pos_dist = distance
                            pos_index = pos_idx

                    for neg_idx in neg_data:
                        neg_image = self.data_file[neg_idx]
                        distance = self.get_distance(image_j, neg_image)
                        if neg_dist > distance:
                            neg_dist = distance
                            neg_index = neg_idx
                    
                    idxs.append((j, pos_index, neg_index))

            random.shuffle(idxs)
            # print(idxs)
            yield idxs

    def __len__(self):
        return (self.data_file.size(0) // self.batch_size)




class veri_triplet_train_dataset(torch.utils.data.Dataset):
    #Adapted from https://github.com/lxc86739795/vehiclereid_baseline/blob/master/dataset/dataset.py
    def __init__(self, data_dir, dataset_type, data_transforms=None):
        self.data_transforms=data_transforms
        xml_filename = "train_label.xml"
        self.image_dir = data_dir+'/'+'image_train/'
        xml_file = open(data_dir+'/'+xml_filename, 'r')
        xml_root = ET.fromstring(xml_file.read())
        self.names = []
        self.labels = []
        self.colourIDs = []
        self.typeIDs = []
        self.camIDs = []
        for item in xml_root.findall('Items/Item'):
            self.names.append(item.attrib['imageName'])
            self.labels.append(int(item.attrib['id']))
            self.camIDs.append(item.attrib['cameraID'])
            self.colourIDs.append(item.attrib['colorID'])
            self.typeIDs.append(item.attrib['typeID'])
        
        self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs = shuffle(self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs, random_state=20)
        valid_num = int(len(self.names)*0.2)
        train_num = len(self.names)-valid_num
        if dataset_type == 'train':
            self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs = self.names[:train_num], self.labels[:train_num], self.camIDs[:train_num], self.colourIDs[:train_num], self.typeIDs[:train_num] 
        elif dataset_type == 'valid':
            self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs = self.names[train_num:], self.labels[train_num:], self.camIDs[train_num:], self.colourIDs[train_num:], self.typeIDs[train_num:] 

    def get_labels(self):
        return list(set(self.labels))
    
    def __len__(self):
        return len(self.names)

    def get_image_from_index(self, idx):
        image = self.names[idx]
        image = Image.open(self.image_dir+image)
        if self.data_transforms:
            image = self.data_transforms(image)
        return image

    def get_data_from_label(self, label, num):
        idxs = [i for i, data in enumerate(self.labels) if data == label]
        idxs = random.choices(idxs, k=num)
        images = [self.names[idx] for idx in idxs]
        return idxs

    
    def __getitem__(self, idx):
        # print(idx)
        # print(len(self.names))
        anchor = self.names[idx[0]]
        positive = self.names[idx[1]]
        negative = self.names[idx[2]]

        anchor = Image.open(self.image_dir+anchor)
        positive = Image.open(self.image_dir+positive)
        negative = Image.open(self.image_dir+negative)
        
        if self.data_transforms:
            anchor = self.data_transforms(anchor)
            positive = self.data_transforms(positive)
            negative = self.data_transforms(negative)
        
        return {'anchor': (anchor, self.labels[idx[0]]), 'positive': (positive, self.labels[idx[1]]), 'negative': (negative, self.labels[idx[2]])}

class veri_dataset(torch.utils.data.Dataset):
    #Adapted from https://github.com/lxc86739795/vehiclereid_baseline/blob/master/dataset/dataset.py
    def __init__(self, data_dir, dataset_type, data_standard_transforms):
        self.data_standard_transforms = data_standard_transforms
        xml_filename = "train_label.xml"
        self.image_dir = data_dir+'/'+'image_train/'
        xml_file = open(data_dir+'/'+xml_filename, 'r')
        xml_root = ET.fromstring(xml_file.read())
        self.names = []
        self.labels = []
        self.colourIDs = []
        self.typeIDs = []
        self.camIDs = []
        for item in xml_root.findall('Items/Item'):
            self.names.append(item.attrib['imageName'])
            self.labels.append(int(item.attrib['id']))
            self.camIDs.append(item.attrib['cameraID'])
            self.colourIDs.append(int(item.attrib['colorid']))
            self.typeIDs.append(int(item.attrib['typeid']))
        self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs = shuffle(self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs, random_state=20)
        valid_num = int(len(self.names)*0.2)
        train_num = len(self.names)-valid_num
        if dataset_type == 'train':
            self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs = self.names[:train_num], self.labels[:train_num], self.camIDs[:train_num], self.colourIDs[:train_num], self.typeIDs[:train_num] 
        elif dataset_type == 'valid':
            self.names, self.labels, self.camIDs, self.colourIDs, self.typeIDs = self.names[train_num:], self.labels[train_num:], self.camIDs[train_num:], self.colourIDs[train_num:], self.typeIDs[train_num:] 
    
    def __getitem__(self, idx):
        image = self.names[idx]
        image = Image.open(self.image_dir+image)
        label = self.labels[idx]
        image = self.data_standard_transforms(image)
        return {'image': image, 'label': label}
    
    def __len__(self):
        return len(self.names)

class veri_test_dataset(torch.utils.data.Dataset):
    #Adapted from https://github.com/lxc86739795/vehiclereid_baseline/blob/master/dataset/dataset.py
    def __init__(self, data_dir, dataset_type, data_transforms=None):
        self.data_transforms = data_transforms
        xml_filename = 'test_label.xml'
        image_dir = 'image_test/' if dataset_type == 'gallery' else 'image_query/'
        image_list = 'name_test.txt' if dataset_type == 'gallery' else 'name_query.txt'
        self.image_list = data_dir+'/'+image_list
        self.image_dir = data_dir+'/'+image_dir
        xml_file = open(data_dir+'/'+xml_filename, 'r')
        xml_root = ET.fromstring(xml_file.read())
        self.names = []
        self.labels = []
        self.colourIDs = []
        self.typeIDs = []
        self.camIDs = []
        list_file = open(self.image_list, 'r')
        # print(list_file)
        self.image_list = [line.strip().split(' ')[0] for line in list_file]


        for item in xml_root.findall('Items/Item'):
            if item.attrib['imageName'] in self.image_list:
                self.names.append(item.attrib['imageName'])
                self.labels.append(int(item.attrib['vehicleID']))
                self.camIDs.append(str(item.attrib['cameraID']))
                self.colourIDs.append(item.attrib['colorID'])
                self.typeIDs.append(item.attrib['typeID'])
    
    def __getitem__(self, idx):
        image = self.names[idx]
        image = Image.open(self.image_dir+image)
        label = self.labels[idx]
        camid = self.camIDs[idx]
        if self.data_transforms:
            image = self.data_transforms(image)
        return (image, label, camid)
    
    def __len__(self):
        return len(self.names)
    
