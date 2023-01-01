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

DUMMY_TRANSFORM = T.Compose([
            T.Resize((32,32)),
            T.Normalize(mean = (0,0,0), std = (1,1,1)),
            T.ToTensor()
        ])

def get_data_transform(means, stdevs, is_train = True):

    if is_train:
        data_transform = T.Compose([
            T.Resize((32,32)),
            T.Normalize(mean = means, std = stdevs),
            T.ToTensor()
        ])
    else:
        data_transform = T.Compose([
            T.Resize((32,32)),
            T.Normalize(mean = means, std = stdevs),
            T.ToTensor()
        ])
    return data_transform

# train_transform = A.Compose(
#     [
#         A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
#         ToTensorV2(),
#     ]
# )

# test_transform = A.Compose(
#     [
#         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
#         ToTensorV2(),
#     ]
# )