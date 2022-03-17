import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def get_labels(fname):
    df = pd.read_csv(fname, header=None)
    label_map = {}
    for i, item in df.iterrows():
        label_map[item[0]] = item[1]
    return label_map


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    if isinstance(x, list):
        x = np.array(x)
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y.squeeze()


def create_yolo_bbox_from(img) -> list:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 0, 0, w - 1, h - 1

    xywh_normalized = xyxy2xywhn([[x1, y1, x2, y2]], w, h)
    return xywh_normalized


def split_label_from_image_name(image_name: Path) -> int:
    basename = image_name.stem
    label_id, image_id = basename.split("_")
    return int(label_id)

def write_yolo_annotations_to_file(root_dir: Path, label_dir: Path, annotations: list):
    output_label_fname = root_dir / "yolo_train.txt"
    written_lines = 0
    with open(output_label_fname, "w") as f1:
        for anno in annotations:
            file_name = anno.get("file_name")
            label_id = anno.get("label_id")
            bbox = anno.get("bbox")
            
            image_label_path = label_dir / (file_name.stem + ".txt")
            with open(image_label_path, "w") as f2:
                x, y, w, h = bbox
                txt = f"{label_id} {x} {y} {w} {h}\n"
                f2.write(txt)
            
            f1.write(file_name.name)
            f1.write("\n")
            written_lines += 1
    print(f"Written {written_lines} rows")


def write_yolo_dataset_info(root_dir: Path, label_mapping: dict):
    """
    YOLO dataset needs two additional info files: obj.names and obj.data
    obj.data
        ```
        classes = 15
        train = data/train.txt
        names = data/obj.names
        backup = backup/
        ```
    obj.names
        ```
        <name of all classes>
        ``
    """
    num_classes = len(label_mapping)
    data_file = root_dir / "obj.data"
    with open(data_file, "w") as f:
        f.write(f"classes = {num_classes}\n")
        f.write("train = data/yolo_train.txt\n")
        f.write("names = data/obj.names\n")
        f.write("backup = backup/\n")
    
    names_file = root_dir / "obj.names"
    with open(names_file, "w") as f:
        for label_name in label_mapping.keys():
            f.write(f"{label_name}\n")


def create_yolo_dataset(root_dir: Path, annotations: list, label_mapping: dict):
    """
    Write yolov5 dataset's contents: obj.names, obj.data, train.txt, labels/*.txt
    """
    YOLO_LABEL_DIR = root_dir / "yolo_labels"
    if not YOLO_LABEL_DIR.is_dir():
        os.makedirs(YOLO_LABEL_DIR)
    
    write_yolo_dataset_info(root_dir, label_mapping)
    write_yolo_annotations_to_file(root_dir, YOLO_LABEL_DIR, annotations)
        

def main():
    DATASET_DIR = Path("../datasets/track4")
    DATASET_ROOT_DIR = DATASET_DIR / "Auto-retail-syndata-release"
    IMAGE_DIR = DATASET_ROOT_DIR / "syn_image_train"
    LABEL_FILE = DATASET_ROOT_DIR / "labels.txt"

    image_files = sorted(list(IMAGE_DIR.glob("**/*.jpg")))
    print(f"Total image files: {len(image_files)}")
    label_mapping = get_labels(LABEL_FILE)

    annotations = []
    for fname in image_files:
        label_id = split_label_from_image_name(fname)
        img = cv2.imread(str(fname))
        xywh_normalized = create_yolo_bbox_from(img)
        anno = {
            "file_name": fname,
            "label_id": label_id - 1, # YOLO index starts at 0
            "bbox": xywh_normalized
        }
        annotations.append(anno)
    
    create_yolo_dataset(DATASET_ROOT_DIR, annotations, label_mapping)


if __name__ == "__main__":
    main()
