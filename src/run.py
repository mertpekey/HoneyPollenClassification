import os
import math
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn

import model.model_classes as models
import model.trainer as trainer
import model.config as config
import utils.other_utils as other_utils
import data.make_dataset as data_funcs

if __name__ == '__main__':
    # Set Seeds
    other_utils.set_seeds(seed = 22)

    class_type = 'family' # or type

    pollen_df = data_funcs.get_dataset_roots(config.DATA_ROOT)
    family_amt_df = pollen_df['family'].value_counts().reset_index().rename(columns={'index':'family', 'family':'img_num'})
    type_amt_df = pollen_df['type'].value_counts().reset_index().rename(columns={'index':'type', 'type':'img_num'})

    # Split Data
    train_df, val_df, test_df = data_funcs.split_datasets(pollen_df, family_amt_df, label_name = class_type)

    # Create Datasets
    train_dataset = data_funcs.PollenDataset(data=train_df, transform=config.train_transform, is_family=True if class_type=='family' else False)
    val_dataset = data_funcs.PollenDataset(data=val_df, transform=config.test_transform, is_family=True if class_type=='family' else False)
    test_dataset = data_funcs.PollenDataset(data=test_df, transform=config.test_transform, is_family=True if class_type=='family' else False)

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Get Model
    my_model = models.get_model(model_name='resnet50',
                            class_names=train_dataset.class_names,
                            full_train=False,
                            pretrained=False)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Train
    model_trainer = trainer.Trainer(model = my_model,
                                    criterion = criterion,
                                    optimizer = optimizer, 
                                    device = config.DEVICE)

    model_trainer.train(train_dataloader, val_dataloader, num_epochs = config.NUM_EPOCHS)