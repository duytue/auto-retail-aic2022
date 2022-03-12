from pathlib import Path

TRAIN_DIR = Path("../datasets/track4/Auto-retail-syndata-release")
IMAGE_DIR = TRAIN_DIR / "syn_image_train"
MASK_DIR = TRAIN_DIR / "segmentation_labels"
TEST_DIR = Path("../datasets/track4/TestA")
LABEL_GT = TRAIN_DIR / "labels.txt"
