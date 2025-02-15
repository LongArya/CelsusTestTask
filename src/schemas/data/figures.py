from pydantic import BaseModel


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
