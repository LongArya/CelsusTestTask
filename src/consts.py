import os
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

REPO_FOLDER = dirname(dirname(abspath(__file__)))
DATA_FOLDER = join(REPO_FOLDER, "data")
TEXTURES_FOLDER = join(DATA_FOLDER, "textures")

TAB10_NORMALIZED_COLORS = matplotlib.colormaps["tab10"].colors
TAB10_RGB_COLORS: np.ndarray = (np.array(TAB10_NORMALIZED_COLORS) * 255).astype(
    np.uint8
)
# convert to list for compatibility with random.sample()
TAB10_RGB_COLORS = list(TAB10_RGB_COLORS)
