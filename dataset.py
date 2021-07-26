import torch
import xml.etree.ElementTree as ET 
from PIL import Image
from sklearn.utils import shuffle

class terry_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_type='train', data_transforms=None):
        self.data_transforms = data_transforms
        xml_file = open(data_dir+'image_labels.xml')
        self.dir_1 = data_dir+'terry/'
        self.dir_2 = data_dir+'non_terry/'
        xml_root = ET.fromstring(xml_file.read())
        self.names = []
        self.labels = []

        for item in xml_root.findall('Items/Item'):
            self.names.append(item.attrib['imageName'])
            self.labels.append(int(item.attrib['label']))

        valid_num = int(len(self.names)*0.2)
        test_num = int(len(self.names)*0.2)
        train_num = len(self.names)-(valid_num+test_num)
        self.names, self.labels, = shuffle(self.names, self.labels, random_state=20)
        if dataset_type == 'train':
            self.names, self.labels = self.names[:train_num], self.labels[:train_num]
        elif dataset_type == 'valid':
            self.names, self.labels = self.names[train_num:train_num+valid_num], self.labels[train_num:train_num+valid_num]
        else:
            self.names, self.labels = self.names[train_num+valid_num:], self.labels[train_num+valid_num:]
        print(dataset_type)
        print(len(self.names))

    def __getitem__(self, idx):
        image_name = self.names[idx]
        image = Image.open(self.dir_2+image_name).convert('RGB') if 'non_terry' in image_name else Image.open(self.dir_1+image_name).convert('RGB')
        label = self.labels[idx]
        if self.data_transforms:
            image = self.data_transforms(image)
        return (image, label)


    def __len__(self):
        return len(self.names)