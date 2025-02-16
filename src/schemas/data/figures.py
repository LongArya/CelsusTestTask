from pydantic import BaseModel
from enum import Enum


class FigureKind(Enum):
    CIRCLE = "CIRCLE"
    SQUARE = "SQUARE"


class Circle(BaseModel):
    """Circle geometric representaion"""

    x_center_pxl: int
    y_center_pxl: int
    radius_pxl: int


class Square(BaseModel):
    """Square geometric representaion"""

    x_center_pxl: int
    y_center_pxl: int
    side_length_pxl: int
    angle_degrees: int
