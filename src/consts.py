import os
from pathlib import Path
from os.path import dirname, join, abspath
from math import sqrt
import matplotlib
import numpy as np


IMG_SIZE = 32
MIN_CIRCLE_RADUIS_PXLS = 3
MAX_SQUARE_BBOX_SIDE_GAIN_KOEF = sqrt(2)  # if rotated by 45 degree
MIN_SQUARE_SIDE_PXLS = 7

# minimal amount of pixels betweem figure edge and image edge
DEFAULT_FIGURE_BOUNDARIES_GAP = 1

REPO_FOLDER = Path(dirname(dirname(abspath(__file__))))
DATA_FOLDER = REPO_FOLDER / "data"
TEXTURES_FOLDER = DATA_FOLDER / "textures"

TAB10_NORMALIZED_COLORS = matplotlib.colormaps["tab10"].colors
TAB10_RGB_COLORS: np.ndarray = (np.array(TAB10_NORMALIZED_COLORS) * 255).astype(
    np.uint8
)
# convert to list for compatibility with random.sample()
TAB10_RGB_COLORS = list(TAB10_RGB_COLORS)

# DATASETS
DATASET_GENERATION_SEED = 22
DATASETS_FOLDER = DATA_FOLDER / "datasets"
TRAIN_DATASET_ROOT = DATASETS_FOLDER / "train"
VAL_DATASET_ROOT = DATASETS_FOLDER / "val"
TEST_TEXTURE_ROOT = DATASETS_FOLDER / "test" / "texture"
TEST_COLOR_ROOT = DATASETS_FOLDER / "test" / "color"

TRAIN_DATASET_SIZE = 1000
VAL_DATASET_SIZE = 200
TEST_COLOR_SIZE = 100
TEST_TEXTURE_SIZE = 100

# MODELS
MODELS_ROOT = REPO_FOLDER / "models"
CENTER_REGRESSION_REFRENCE_CKPT = (
    MODELS_ROOT / "center_regression_train" / "checkpoint_epoch=17-val_MSE=0.000.ckpt"
)
