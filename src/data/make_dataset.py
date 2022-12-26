import os
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import get_pollen_family as pf


def get_dataset_roots(dir_path):
    """
    Walks through dir_path returning a dictionary contains all directory names as key and all of its content as list as value.
    
    Args:
    dir_path (str): dataset directory

    Returns:
    Dictionary:
        Key: Directories
        Value: Content file names as list
    """
    class_dict = {}
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        class_name = dirpath.split('/')[-1]
        if class_name != 'dataset':
            for filename in os.listdir(dirpath):
                class_dict[os.path.join(dir_path, class_name, filename)] = class_name
                
    data_dict = pd.DataFrame({'img_file':list(class_dict.keys()), 'label':list(class_dict.values())})
    data_dict['PollenFamily'] = data_dict['label'].apply(pf.get_pollen_family)
    return data_dict


class PollenDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.data = get_dataset_roots(self.root_dir)
        self.class_names = self.data['label'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_root = self.data.iloc[idx,0]
        type = self.data.iloc[idx, 1]
        family = self.data.iloc[idx, 2]
        
        #img = torchvision.io.read_image(img_root)
        img = Image.open(img_root)
        
        if self.transform:
            img = self.transform(img)

        return img, type, family


def create_dataloader(dataset, batch_size, is_train=False):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return dataloader

