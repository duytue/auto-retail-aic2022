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
color = (128, 128, 128)
txt_color = (255, 255, 255)
if __name__ == "__main__":
    dataloader = AIC2022_AutoRetail_DataLoader(data_dir=TRAIN_DIR, batch_size=32)
    dataset = AIC2022_AutoRetail_Dataset(root_dir=TRAIN_DIR)
    for data in dataset:
        img, boxes, labels = data
        lw = max(round(sum(img.shape) / 2 * 0.003), 2)  # line width
        for index, box in enumerate(boxes):
            p1, p2 = (box[0], box[1]), (box[2], box[3])
            cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            cv2.putText(img, labels[index], (p1[0], p1[1] - 2), 0, lw / 3, txt_color, lineType=cv2.LINE_AA)
        cv2.imshow("mosaic", img)
        if cv2.waitKey(0) == 27:
            break
        break
