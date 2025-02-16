import cv2
from .utils import generate_random_circle, generate_random_square, generate_sample_image
from ..schemas.data.figures import Circle, Square, FigureKind
from ..consts import (
    MIN_CIRCLE_RADUIS_PXLS,
    MIN_SQUARE_SIDE_PXLS,
    DEFAULT_FIGURE_BOUNDARIES_GAP,
    IMG_SIZE,
    TEXTURES_FOLDER,
    TAB10_RGB_COLORS,
)
from ..schemas.data.figure_sample import FIGURE_SAMPLE, FigureSampleMeta
from ..utils import traverse_directory_for_files
from abc import abstractmethod
from typing import List, Tuple, Union
from pydantic import FilePath
import random as rd
import numpy as np


class SampleGeneratorInterface:
    """All classes for figure sample generation should support this interface"""

    def __init__(
        self,
        target_img_size_wh: Tuple[int, int],
        min_circle_radius_pxl: int,
        min_square_side_pxl: int,
        gap_between_image_boundaries: int,
    ):
        self._target_img_w, self._target_img_h = target_img_size_wh
        self._min_circle_radius_pxl = min_circle_radius_pxl
        self._min_square_side_pxl = min_square_side_pxl
        self._gap_between_image_boundaries = gap_between_image_boundaries

    def _generate_random_figure(self) -> Tuple[FigureKind, Union[Circle, Square]]:
        circle_probability: bool = rd.random()
        if circle_probability < 0.5:
            figure_kind = FigureKind.CIRCLE
            figure = generate_random_circle(
                target_img_shape_wh=(self._target_img_w, self._target_img_h),
                min_radius_pxl=self._min_circle_radius_pxl,
                gap_between_image_boundaries=self._gap_between_image_boundaries,
            )
        else:
            figure_kind = FigureKind.SQUARE
            figure = generate_random_square(
                target_img_shape_wh=(self._target_img_w, self._target_img_h),
                min_side_pxl=self._min_square_side_pxl,
                gap_between_image_boundaries=self._gap_between_image_boundaries,
            )
        return figure_kind, figure

    @abstractmethod
    def generate(self) -> FIGURE_SAMPLE:
        """Generates random sample"""


class ColoredFigureSampleGenerator(SampleGeneratorInterface):
    """
    Generates figure sample where background and figure are filled with random colors
    """

    def __init__(
        self,
        rgb_colors_pool: List[Tuple[int, int, int]],
        target_img_size_wh: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
        min_circle_radius_pxl: int = MIN_CIRCLE_RADUIS_PXLS,
        min_square_side_pxl: int = MIN_SQUARE_SIDE_PXLS,
        gap_between_image_boundaries: int = DEFAULT_FIGURE_BOUNDARIES_GAP,
    ):
        self._rgb_colors_pool = rgb_colors_pool
        if len(self._rgb_colors_pool) < 2:
            raise ValueError("RGB color pool should have at least 2 colors")
        super().__init__(
            target_img_size_wh=target_img_size_wh,
            min_circle_radius_pxl=min_circle_radius_pxl,
            min_square_side_pxl=min_square_side_pxl,
            gap_between_image_boundaries=gap_between_image_boundaries,
        )

    def _generate_colored_img(self, color: Tuple[int, int, int]) -> np.ndarray:
        output_img = np.full(
            (self._target_img_h, self._target_img_w, 3), color, dtype=np.uint8
        )
        return output_img

    def generate(self) -> FIGURE_SAMPLE:
        """
        Generates random figure sample using following algorithm:
        0. Pick figure kind
        1. Generates random figure of that type
        2. Picks 2 different random colors for background and figure
        3. Fills background with background color
        4. Places figure with figure color on background
        """

        bg_color, figure_color = rd.sample(population=self._rgb_colors_pool, k=2)
        background_texture: np.ndarray = self._generate_colored_img(bg_color)
        figure_texture: np.ndarray = self._generate_colored_img(figure_color)

        figure_kind: FigureKind
        figure: Union[Circle, Square]
        figure_kind, figure = self._generate_random_figure()

        sample_img = generate_sample_image(
            background_texture=background_texture,
            figure_texture=figure_texture,
            figure=figure,
        )
        meta = FigureSampleMeta(
            image=sample_img, figure_meta=figure, figure_kind=figure_kind
        )
        return sample_img, meta


class TextureFigureSampleGenerator(SampleGeneratorInterface):
    """
    Generates figure sample where background and figure are filled with random textures
    """

    def __init__(
        self,
        textures_pool: List[FilePath],
        target_img_size_wh: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
        min_circle_radius_pxl: int = MIN_CIRCLE_RADUIS_PXLS,
        min_square_side_pxl: int = MIN_SQUARE_SIDE_PXLS,
        gap_between_image_boundaries: int = DEFAULT_FIGURE_BOUNDARIES_GAP,
    ):
        self._textures_pool = textures_pool
        if len(self._textures_pool) < 2:
            raise ValueError("RGB color pool should have at least 2 colors")
        super().__init__(
            target_img_size_wh=target_img_size_wh,
            min_circle_radius_pxl=min_circle_radius_pxl,
            min_square_side_pxl=min_square_side_pxl,
            gap_between_image_boundaries=gap_between_image_boundaries,
        )

    def _get_target_size_texture(self, texture_file: FilePath) -> np.ndarray:
        """
        If texture is smaller than target size, resizes it to fill the target image.
        Otherwise gets random crop of image size
        """

        texture: np.ndarray = cv2.imread(str(texture_file))
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        h, w = texture.shape[:2]
        if h < self._target_img_h or w < self._target_img_w:
            output_texture: np.ndarray = cv2.resize(
                texture, (self._target_img_w, self._target_img_h)
            )
            return output_texture

        # get random crop
        crop_y1: int = 0
        if h > self._target_img_h:
            crop_y1 = rd.randrange(0, h - self._target_img_w)
        crop_x1: int = 0
        if w > self._target_img_w:
            crop_x1 = rd.randrange(0, w - self._target_img_h)

        texture_crop: np.ndarray = texture[
            crop_y1 : crop_y1 + self._target_img_h,
            crop_x1 : crop_x1 + self._target_img_w,
            :,
        ]
        return texture_crop

    def generate(self) -> FIGURE_SAMPLE:
        """
        Generates random figure sample using following algorithm:
        1. Generates random figure of that type
        2. Picks 2 different random textures for background and figure
        2.1 Randomly crops textures from
        3. Fills background with background color
        4. Places figure with figure color on background
        """

        bg_texture_file, figure_texture_file = rd.sample(
            population=self._textures_pool, k=2
        )
        background_texture: np.ndarray = self._get_target_size_texture(bg_texture_file)
        figure_texture: np.ndarray = self._get_target_size_texture(figure_texture_file)

        # generate figure
        figure_kind: FigureKind
        figure: Union[Circle, Square]
        figure_kind, figure = self._generate_random_figure()
        sample_img = generate_sample_image(
            background_texture=background_texture,
            figure_texture=figure_texture,
            figure=figure,
        )
        meta = FigureSampleMeta(
            image=sample_img, figure_meta=figure, figure_kind=figure_kind
        )
        return sample_img, meta


def init_tab10_colors_generator() -> ColoredFigureSampleGenerator:
    generator = ColoredFigureSampleGenerator(rgb_colors_pool=TAB10_RGB_COLORS)
    return generator


def init_texture_generator() -> TextureFigureSampleGenerator:
    texture_files: List[FilePath] = traverse_directory_for_files(TEXTURES_FOLDER)
    generator = TextureFigureSampleGenerator(textures_pool=texture_files)
    return generator
