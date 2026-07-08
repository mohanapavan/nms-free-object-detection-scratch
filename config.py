# Configuration parameters
TRAIN_IMG_DIR = "/kaggle/input/competitions/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/train/images"
VAL_IMG_DIR   = "/kaggle/input/competitions/synthetic-2-real-object-detection-challenge/Synthetic to Real Object Detection Challenge/data/val/images"
NUM_CLASSES   = 1
INPUT_SIZE    = 640
BATCH_SIZE    = 16
EPOCHS        = 100
LR            = 0.01
MOMENTUM      = 0.937

PARAMS = {
    "mosaic": 1.0, "mix_up": 0.0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flip_ud": 0.0, "flip_lr": 0.5,
    "box": 7.5, "cls": 0.5, "dfl": 1.5,
    "weight_decay": 0.0005,
}