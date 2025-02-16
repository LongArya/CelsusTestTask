from src.data.fugure_sample_generation import TextureFigureSampleGenerator
from src.consts import IMG_SIZE
from tempfile import NamedTemporaryFile
import numpy as np
import cv2


def test_with_small_texture():
    texture_H, texture_W = IMG_SIZE - 5, IMG_SIZE - 5
    texture = np.zeros((texture_H, texture_W, 3), dtype=np.uint8)
    with NamedTemporaryFile(suffix=".png") as tmp_texture:
        cv2.imwrite(tmp_texture.name, texture)
        generator = TextureFigureSampleGenerator(
            target_img_size_wh=(IMG_SIZE, IMG_SIZE),
            textures_pool=[tmp_texture.name, tmp_texture.name],
        )
        img, meta = generator.generate()
        img_H, img_W = img.shape[:2]
        assert img_H == IMG_SIZE
        assert img_W == IMG_SIZE


def test_with_exact_img_size_texture():
    texture_H, texture_W = IMG_SIZE, IMG_SIZE
    texture = np.zeros((texture_H, texture_W, 3), dtype=np.uint8)
    with NamedTemporaryFile(suffix=".png") as tmp_texture:
        cv2.imwrite(tmp_texture.name, texture)
        generator = TextureFigureSampleGenerator(
            target_img_size_wh=(IMG_SIZE, IMG_SIZE),
            textures_pool=[tmp_texture.name, tmp_texture.name],
        )
        img, meta = generator.generate()
        img_H, img_W = img.shape[:2]
        assert img_H == IMG_SIZE
        assert img_W == IMG_SIZE


def test_with_texture_one_pixel_larger():
    texture_H, texture_W = IMG_SIZE + 1, IMG_SIZE + 1
    texture = np.zeros((texture_H, texture_W, 3), dtype=np.uint8)
    with NamedTemporaryFile(suffix=".png") as tmp_texture:
        cv2.imwrite(tmp_texture.name, texture)
        generator = TextureFigureSampleGenerator(
            target_img_size_wh=(IMG_SIZE, IMG_SIZE),
            textures_pool=[tmp_texture.name, tmp_texture.name],
        )
        img, meta = generator.generate()
        img_H, img_W = img.shape[:2]
        assert img_H == IMG_SIZE
        assert img_W == IMG_SIZE
