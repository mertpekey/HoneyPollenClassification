import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

import model.config as config

def set_seeds(seed = 22):

    torch.manual_seed(seed) # Seed for general torch operations
    torch.cuda.manual_seed(seed) # Seed for CUDA
    torch.cuda.manual_seed_all(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)


def get_model_file_size(model_path, model_name='[NOT GIVEN]'):

    from pathlib import Path

    # Get the model size in bytes then convert to megabytes
    model_size = Path(model_path).stat().st_size // (1024*1024)
    print(f"{model_name} model size: {model_size} MB")

def get_normalization_params(dataset): ## Tum dataset olarak degistir

    X, _ = dataset
    means = []
    stdevs = []
    
    n_channels = X.shape[1]
    for c in range(n_channels):
        mean = torch.mean(X[:, c])
        std = torch.std(X[:, c])

        means.append(mean)
        stdevs.append(std)

    return means, stdevs


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def create_writer(experiment_name: str, model_name: str, extra: str=None):
	
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)