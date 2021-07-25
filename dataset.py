import torch
import xml.etree.ElementTree as ET 
from PIL import Image

class terry_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_transforms=None):
        self.data_transforms = data_transforms
        xml_file = data_dir+'image_labels.xml'
        self.dir_1 = data_dir+'/terry/'
        self.dir_2 = data_dir+'/non_terry/'
        xml_root = ET.fromstring(xml_file.read())
        self.names = []
        self.labels = []

        for item in xml_root.findall('Items/Item'):
            self.names.append(item.attrib['imageName'])
            self.labels.append(int(item.attrib['label']))

    def __getitem__(self, idx):
        image = self.names[idx]
        image = Image.open(self.dir_2+image) if 'non_terry' in image else Image.open(self.dir_1+image)
        label = self.labels[idx]
        if self.data_transforms:
            image = self.data_transforms(image)
        return (image, label)


    def __len__(self):
        return len(self.names)