import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


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
        x = np.array(x, dtype=np.float32)
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
    label_id, _ = basename.split("_")
    return int(label_id)


def prepare_yolo_annotations(image_files: list) -> list:
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
    return annotations


def write_yolo_anno_to_file(label_dir: Path, fname: Path, file_list: list):
    annotations = prepare_yolo_annotations(file_list)

    written_lines = 0
    print(f" Writing labels to {fname}")
    with open(fname, "w") as f1:
        for anno in annotations:
            file_name = anno.get("file_name")
            label_id = anno.get("label_id")
            bbox = anno.get("bbox")
            
            image_label_path = label_dir / (file_name.stem + ".txt")
            with open(image_label_path, "w") as f2:
                x, y, w, h = bbox
                txt = f"{label_id} {x:.3f} {y:.3f} {w:.3f} {h:.3f}\n"
                f2.write(txt)
            
            f1.write(f"./images/{file_name.name}")
            f1.write("\n")
            written_lines += 1
    print(f" Written {written_lines} rows")


def stratified_k_folds(annotations: list, n_splits: int = 1, test_size: float = 0.2, seed: int = None):
    labels = [split_label_from_image_name(anno) for anno in annotations]
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    folds = []
    for train_index, test_index in skf.split(annotations, labels):
        X_train, X_test = annotations[train_index], annotations[test_index]
        # y is included in annotations
        # y_train, y_test = labels[train_index], labels[test_index]
        folds.append([X_train, X_test])
    return folds


def generate_yolo_annotations(root_dir: Path, label_dir: Path, annotations: list, seed: int = None):
    # Split dataset into train-val
    folds = stratified_k_folds(np.array(annotations), n_splits=4, test_size=0.1, seed=seed)

    for i, fold in enumerate(folds):
        print(f"-- Performing Stratified K-Folds")
        train_set, val_set = fold
        val_set, test_set = stratified_k_folds(
            val_set,
            n_splits=1,
            test_size=0.5,
            seed=seed
        )[0]

        train_label_fname = root_dir / f"small-yolo_train_{i}.txt"
        write_yolo_anno_to_file(label_dir, train_label_fname, train_set)

        val_label_fname = root_dir / f"small-yolo_val_{i}.txt"
        write_yolo_anno_to_file(label_dir, val_label_fname, val_set)
        
        test_label_fname = root_dir / f"small-yolo_test_{i}.txt"
        write_yolo_anno_to_file(label_dir, test_label_fname, test_set)
        print()


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


def create_yolo_dataset(root_dir: Path, image_files: list, label_mapping: dict):
    """
    Write yolov5 dataset's contents: obj.names, obj.data, train.txt, labels/*.txt
    """
    seed = 42
    YOLO_LABEL_DIR = root_dir / "labels"
    if not YOLO_LABEL_DIR.is_dir():
        os.makedirs(YOLO_LABEL_DIR)
    
    write_yolo_dataset_info(root_dir, label_mapping)
    generate_yolo_annotations(root_dir, YOLO_LABEL_DIR, image_files, seed=seed)


def plot_dataset_histogram(image_files, label_mapping):
    from collections import Counter
    labels = [split_label_from_image_name(name) for name in image_files]
    counts = Counter(labels)
    print(counts)
    # plot = sns.displot(labels, bins=len(label_mapping), kde=True)
    # plot.figure.savefig("distibution.png")


def main():
    DATASET_DIR = Path("../datasets/track4")
    DATASET_ROOT_DIR = DATASET_DIR / "Auto-retail-syndata-release"
    IMAGE_DIR = DATASET_ROOT_DIR / "syn_image_train"
    LABEL_FILE = DATASET_ROOT_DIR / "labels.txt"

    image_files = sorted(list(IMAGE_DIR.glob("**/*.jpg")))
    # image_files = random.sample(image_files, 10000)
    print(f"Total image files: {len(image_files)}")
    label_mapping = get_labels(LABEL_FILE)

    plot_dataset_histogram(image_files, label_mapping)

    create_yolo_dataset(DATASET_ROOT_DIR, image_files, label_mapping)


if __name__ == "__main__":
    main()
