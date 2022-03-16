import glob
import os
import random
from logging import root

import cv2
import numpy as np
import torch
from constants import IMAGE_DIR, LABEL_GT
from torch.utils.data import Dataset
from utils.augmentation import copy_paste, random_perspective, xyn2xy, xywhn2xyxy


class AIC2022_AutoRetail_Dataset(Dataset):
    """AI City Challenge track 4 AutoRetail dataset."""

    def __init__(self, root_dir, transform=None, img_size=640, stride=32, auto=True):
        self.labels = self.load_labels()
        self.data_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.stride = stride
        self.image_paths = glob.glob(str(IMAGE_DIR / "*.jpg"), recursive=False)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.indices = range(len(self.image_paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_labels(self):
        lines = []
        labels_map = {}
        with open(LABEL_GT) as f:
            lines = f.readlines()

        for line in lines:
            label_name, index = line.strip().split(",")
            labels_map[int(index)] = label_name

        return labels_map

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        img_path = self.image_paths[i]
        im = cv2.imread(img_path)  # BGR
        assert im is not None, f"Image Not Found {img_path}"
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                # interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA,
                interpolation=cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def get_label(self, index: int):
        img_path = self.image_paths[index]
        file_name = os.path.basename(img_path)
        image_index, count = file_name.split("_")
        return self.labels[int(image_index)]

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.tolist()

        # Load mosaic
        img, boxes, labels = self.load_mosaic(index)
        if self.transform:
            img = self.transform(img)

        return img, boxes, labels

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        boxes4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # boxes
            box = [x1b, y1b, x2b - x1b, y2b - y1b]
            box = xywhn2xyxy(box, w, h, padw, padh)  # normalized xywh to pixel xyxy format
            boxes4.append(box)

            # Labels
            label = self.get_label(i)
            labels4.append(label)

        # Concat/clip labels
        # labels4 = np.concatenate(labels4, 0)
        # for x in (labels4[:, 1:], *segments4):
        #     np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp["copy_paste"])
        # img4, labels4 = random_perspective(
        #     img4,
        #     labels4,
        #     segments4,
        #     degrees=self.hyp["degrees"],
        #     translate=self.hyp["translate"],
        #     scale=self.hyp["scale"],
        #     shear=self.hyp["shear"],
        #     perspective=self.hyp["perspective"],
        #     border=self.mosaic_border,
        # )  # border to remove

        return img4, boxes4, labels4

    def load_mosaic9(self, index):
        # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        random.shuffle(indices)
        hp, wp = -1, -1  # height, width previous
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady :, x1 - padx :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc : yc + 2 * s, xc : xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9 = random_perspective(
            img9,
            labels9,
            segments9,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img9, labels9
