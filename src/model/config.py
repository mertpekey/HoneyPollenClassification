import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_MODEL = "gen.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_CHANNELS = 3
DATA_ROOT = '/Users/mpekey/Desktop/Mert_SabanciUniv/CS518/HoneyPollenClassification/dataset'


train_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)