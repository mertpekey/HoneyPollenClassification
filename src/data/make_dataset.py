import os
import math
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import get_pollen_family as pf


def get_dataset_roots(dir_path):

    class_dict = {}
    
    for dirpath, dirnames, filenames in os.walk(dir_path):
        class_name = dirpath.split('/')[-1]
        if class_name != 'dataset':
            for filename in os.listdir(dirpath):
                class_dict[os.path.join(dir_path, class_name, filename)] = class_name
                
    data_dict = pd.DataFrame({'img_file':list(class_dict.keys()), 'type':list(class_dict.values())})
    data_dict['family'] = data_dict['type'].apply(pf.get_pollen_family)
    return data_dict


def split_datasets(pollen_df, class_amt_df, label_name = 'type'):

    class_names = pollen_df[label_name].unique()

    first_check = True

    for t in class_names:

        img_amount = class_amt_df[class_amt_df[label_name] == t].img_num.iloc[0]

        if label_name == 'type':
            if img_amount >= 5:
                split_ratio = (0.6,0.2,0.2) # May be changed
            elif img_amount == 4:
                split_ratio = (0.5,0.25,0.25)
            elif img_amount == 3:
                split_ratio = (0.33,0.33,0.33)
            elif img_amount == 2:
                split_ratio = (0.5,0.5,0.0)
        elif label_name == 'family':
            if img_amount >= 5:
                split_ratio = (0.6,0.2,0.2)
            elif img_amount == 4:
                split_ratio = (0.5,0.25,0.25)
            elif img_amount == 3:
                split_ratio = (0.33,0.33,0.33)
            elif img_amount == 2:
                split_ratio = (0.5,0.5,0.0)

        idx_list = list(pollen_df[pollen_df[label_name] == t].index)

        train_idx = np.random.choice(idx_list, size=math.ceil(len(idx_list)*split_ratio[0]), replace=False)
        remaining_idx = []
        for i in idx_list:
            if i not in train_idx:
                remaining_idx.append(i)

        val_idx = np.random.choice(remaining_idx, size=math.ceil(len(remaining_idx)*(split_ratio[1]/(1-split_ratio[0]))), replace=False)

        test_idx = []
        for i in remaining_idx:
            if i not in val_idx:
                test_idx.append(i)

        if first_check:
            train_data = pollen_df.loc[train_idx,['img_file', label_name]]
            val_data = pollen_df.loc[val_idx,['img_file', label_name]]
            test_data = pollen_df.loc[test_idx,['img_file', label_name]]
            first_check = False
        else:
            train_data = pd.concat([train_data, pollen_df.loc[train_idx,['img_file', label_name]]], axis = 0)
            val_data = pd.concat([val_data, pollen_df.loc[val_idx,['img_file', label_name]]], axis = 0)
            test_data = pd.concat([test_data, pollen_df.loc[test_idx,['img_file', label_name]]], axis = 0)

    return train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True)


class PollenDataset(Dataset):
    
    def __init__(self, data, transform=None, is_family=False):
        
        self.transform = transform
        
        self.data = data
        if is_family:
            self.class_names = self.data['family'].unique()
        else:
            self.class_names = self.data['type'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_root = self.data.iloc[idx,0]
        label = self.data.iloc[idx, 1]
        
        #img = torchvision.io.read_image(img_root)
        img = Image.open(img_root)
        
        if self.transform:
            img = self.transform(img)

        return img, label


def create_dataloader(dataset, batch_size, is_train=False):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return dataloader

