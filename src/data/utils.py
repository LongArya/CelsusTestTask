import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple
from ..schemas.data.figures import Square, Circle

CV_COLOR_TYPE = Union[int, Tuple[int, int, int]]
GRAYSCALE_WHITE = 255


def plot_filled_square(
    img: np.ndarray, square: Square, color: CV_COLOR_TYPE, use_aa: bool
) -> None:
    rect = (
        (square.x_center_pxl, square.y_center_pxl),
        (square.side_length_pxl, square.side_length_pxl),
        square.angle_degrees,
    )
    box_points = cv2.boxPoints(rect)
    box_points = np.round(box_points).astype(np.int32)

    kwargs = {
        "img": img,
        "pts": [box_points],
        "color": color,
    }
    if use_aa:
        kwargs["lineType"] = cv2.LINE_AA

    cv2.fillPoly(**kwargs)


def plot_filled_circle(
    img: np.ndarray, circle: Circle, color: CV_COLOR_TYPE, use_aa: bool
) -> None:
    kwargs = {
        "img": img,
        "center": (circle.x_center_pxl, circle.y_center_pxl),
        "radius": circle.radius_pxl,
        "color": color,
        "thickness": cv2.FILLED,
    }
    if use_aa:
        kwargs["lineType"] = cv2.LINE_AA

    cv2.circle(**kwargs)


def generate_sample_image(
    background_texture: np.ndarray, figure_texture: np.ndarray, figure: Union[Circle, Square]
) -> np.ndarray:
    """_summary_

    Args:
        background_texture (np.ndarray): _description_
        figure_texture (np.ndarray): _description_
        figure (Union[Circle, Square]): _description_
    """

    def _check_image_shapes_correspondence() -> bool:
        bg_h, bg_w = background_texture.shape[:2]
        figure_h, figure_w = figure_texture.shape[:2]
        shapes_correspondence = (bg_h == figure_h) and (bg_w == figure_w)
        return shapes_correspondence

    def _check_figure() -> bool:
        correct_figure_type: bool = isinstance(figure, Circle) or isinstance(figure, Square)
        return correct_figure_type

    if not _check_image_shapes_correspondence():
        raise ValueError(
            f"Conflicting shapes: BG={background_texture.shape}, figure={figure_texture.shape}"
        )

    if not _check_figure():
        raise ValueError(f"Invalid figure type {type(figure)}")

    output_h, output_w = background_texture.shape[:2]
    mask = np.zeros((output_h, output_w), dtype=np.uint8)
    if isinstance(figure, Circle):
        plot_filled_circle(mask, figure, GRAYSCALE_WHITE, use_aa=True)
    elif isinstance(figure, Square):
        plot_filled_square(mask, figure, GRAYSCALE_WHITE, use_aa=True)

    mask_normalized = mask.astype(np.float32) / GRAYSCALE_WHITE
    mask_normalized = np.expand_dims(mask_normalized, -1)

    blended = figure_texture * mask_normalized + background_texture.astype(np.float32) * (
        1 - mask_normalized
    )
    blended = blended.astype(np.uint8)
    return blended
