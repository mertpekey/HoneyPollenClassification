import torch
from PIL import Image
import torchvision.transforms as T
import utils.other_utils as other_utils
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_MODEL = "/Users/mpekey/Desktop/Mert_SabanciUniv/CS518/HoneyPollenClassification/model_checkpoints/model_ch.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
BATCH_SIZE = 8
NUM_WORKERS = 2
IMG_CHANNELS = 3
IMG_SIZE = (232,232)
DATA_ROOT = '/content/drive/MyDrive/CS518/dataset_resized'
NO_AUGMENT=False
PATIENCE = 5

train_transform = T.Compose([
            T.CenterCrop((224,224)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])


# For No Augmentation
if NO_AUGMENT:
  train_transform = T.Compose([
              T.CenterCrop((224,224)),
              T.ToTensor(),
              T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
          ])

test_transform = T.Compose([
            T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])