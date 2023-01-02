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
NUM_EPOCHS = 1
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_CHANNELS = 3
DATA_ROOT = '/Users/mpekey/Desktop/Mert_SabanciUniv/CS518/HoneyPollenClassification/dataset'


train_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean = (0,0,0), std = (1,1,1))
        ])
test_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean = (0,0,0), std = (1,1,1))
        ])