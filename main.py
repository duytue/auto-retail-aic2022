import glob

import cv2
import pytorch_lightning as pl
import torch

from constants import IMAGE_DIR, TRAIN_DIR
from data_loader.data_loaders import AIC2022_AutoRetail_DataLoader
from dataset.track4_dataset import AIC2022_AutoRetail_Dataset

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    dataloader = AIC2022_AutoRetail_DataLoader(data_dir=TRAIN_DIR, batch_size=32)
    dataset = AIC2022_AutoRetail_Dataset(root_dir=TRAIN_DIR)
    for data in dataset:
        print(data)
        cv2.imshow("mosaic", data[0])
        if cv2.waitKey(0) == 27:
            break
        break
