import cv2
from math import floor, ceil
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple
from ..schemas.data.figures import Square, Circle
from ..consts import MAX_SQUARE_BBOX_SIDE_GAIN_KOEF

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


def generate_random_circle(
    target_img_shape_wh: Tuple[int, int],
    min_radius_pxl: int,
    gap_between_image_boundaries: int = 1,
) -> Circle:
    """
    Generates random circle that could be placed on image.
    Ensures that it will be within image boundaries

    Args:
        target_img_shape_wh (Tuple[int, int]): width, height of target img
        min_radius_pxl (int): minimum allowed radius
        gap_between_image_boundaries (int, optional): Space that will be left empty
        between circle boundary and image edge. Defaults to 1.

    Returns:
        Circle: random circle that could be placed on image of target size
    """

    w, h = target_img_shape_wh
    min_img_side: int = min(w, h)
    max_possible_radius_pxl: int = int(min_img_side // 2) - gap_between_image_boundaries * 2
    circle_radius = rd.randrange(min_radius_pxl, max_possible_radius_pxl + 1)

    # define possible values for circle center
    min_x = gap_between_image_boundaries + circle_radius
    max_x = (w - 1) - circle_radius - gap_between_image_boundaries

    min_y = gap_between_image_boundaries + circle_radius
    max_y = (h - 1) - circle_radius - gap_between_image_boundaries

    # sanity check ranges
    if not (min_x < max_x):
        raise ValueError("Invalid range for center x")

    if not (min_y < max_y):
        raise ValueError("Invalid range for center y")

    x_center: int = rd.randrange(min_x, max_x + 1)
    y_center: int = rd.randrange(min_y, max_y + 1)
    generated_circle = Circle(
        x_center_pxl=x_center, y_center_pxl=y_center, radius_pxl=circle_radius
    )
    return generated_circle


def generate_random_square(
    target_img_shape_wh: Tuple[int, int],
    min_side_pxl: int,
    gap_between_image_boundaries: int = 1,
) -> Square:
    """
    Generates random rotated square that could be placed on image.
    Ensures that it will be within image boundaries

    Args:
        target_img_shape_wh (Tuple[int, int]): _description_
        min_side_pxl (int): _description_
        gap_between_image_boundaries (int, optional): _description_. Defaults to 1.

    Returns:
        Square: _description_
    """

    def _get_rotated_square_bounding_rect_side(box_side_pxl: int, angle_degrees: int) -> int:
        """Gets side of bounding bbox that will cover rotated square, defined by side and angle"""

        origin_point = (0, 0)
        rect = (
            origin_point,
            (box_side_pxl, box_side_pxl),
            angle_degrees,
        )
        rotated_bbox_points = cv2.boxPoints(rect)
        rotated_bbox_w = rotated_bbox_points[:, 0].max() - rotated_bbox_points[:, 0].min()
        rotated_bbox_h = rotated_bbox_points[:, 1].max() - rotated_bbox_points[:, 1].min()
        return int(ceil(max(rotated_bbox_h, rotated_bbox_w)))

    def _get_maximum_square_side() -> int:
        """Gets maximum side, that will allow square to fit in the image with any rotation"""

        w, h = target_img_shape_wh
        min_img_side: int = min(w, h)
        max_square_side = (
            min_img_side - gap_between_image_boundaries * 2
        ) / MAX_SQUARE_BBOX_SIDE_GAIN_KOEF
        max_square_side = int(floor(max_square_side)) - 1
        return max_square_side

    # generate square dimensions
    max_square_side = _get_maximum_square_side()
    square_side = rd.randrange(min_side_pxl, max_square_side + 1)
    angle_degrees = rd.randrange(0, 360 + 1)

    rotated_square_bbox_side: int = _get_rotated_square_bounding_rect_side(
        box_side_pxl=square_side, angle_degrees=angle_degrees
    )

    # generate square center
    w, h = target_img_shape_wh
    rotated_square_bbox_half_side: int = int(rotated_square_bbox_side / 2)
    min_x = gap_between_image_boundaries + rotated_square_bbox_half_side
    max_x = (w - 1) - rotated_square_bbox_half_side - gap_between_image_boundaries

    min_y = gap_between_image_boundaries + rotated_square_bbox_half_side
    max_y = (h - 1) - rotated_square_bbox_half_side - gap_between_image_boundaries

    # sanity check center ranges
    if not (min_x < max_x):
        raise ValueError(f"Invalid range for center x {min_x, max_x}")

    if not (min_y < max_y):
        raise ValueError(f"Invalid range for center y {min_y, max_y}")

    x_center: int = rd.randrange(min_x, max_x + 1)
    y_center: int = rd.randrange(min_y, max_y + 1)

    generate_random_square = Square(
        x_center_pxl=x_center,
        y_center_pxl=y_center,
        side_length_pxl=square_side,
        angle_degrees=angle_degrees,
    )
    return generate_random_square
