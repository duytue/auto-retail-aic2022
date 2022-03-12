import math
import random

import cv2
import numpy as np
from base import BaseDataLoader
from dataset.track4_dataset import AIC2022_AutoRetail_Dataset
from torchvision import transforms


class AIC2022_AutoRetail_DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.data_dir = data_dir
        self.dataset = None
        self.dataset = AIC2022_AutoRetail_Dataset(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
