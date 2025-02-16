from pydantic import BaseModel
from .figures import FigureKind, Circle, Square
from typing import Union, Tuple
import numpy as np


class FigureSampleMeta(BaseModel):
    figure_kind: FigureKind
    figure_meta: Union[Circle, Square]


FIGURE_SAMPLE = Tuple[np.ndarray, FigureSampleMeta]
