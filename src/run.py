import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    if class_type == 'type':
        train_df, val_df, test_df = data_funcs.split_datasets(pollen_df, type_amt_df, label_name = class_type)
    elif class_type == 'family':
        train_df, val_df, test_df = data_funcs.split_datasets(pollen_df, family_amt_df, label_name = class_type)

    # Create Datasets
    train_dataset = data_funcs.PollenDataset(data=train_df, transform=config.train_transform, is_family=True if class_type=='family' else False)
    val_dataset = data_funcs.PollenDataset(data=val_df, transform=config.test_transform, is_family=True if class_type=='family' else False)
    test_dataset = data_funcs.PollenDataset(data=test_df, transform=config.test_transform, is_family=True if class_type=='family' else False)

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model_name = 'resnet50'
    experiment_name = 'zero_train_transform_8b_64s_1e4lr'

    # Get Model
    my_model = models.get_model(model_name=model_name,
                            class_names=train_dataset.class_names,
                            full_train=True,
                            pretrained=False)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    scheduler = None
    # Train
    model_trainer = trainer(model = my_model,
                                criterion = criterion,
                                optimizer = optimizer, 
                                lr_scheduler = scheduler,
                                device = config.DEVICE,
                                model_name = model_name,
                                experiment_name = experiment_name)

    model_trainer.train(train_dataloader, val_dataloader, num_epochs = config.NUM_EPOCHS, patience = config.PATIENCE)