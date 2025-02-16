import os
import torch
from tqdm import tqdm
from typing import List
from torch.utils.data import Dataset, default_collate
from pathlib import Path
from pydantic import DirectoryPath, FilePath
from ..schemas.data.figure_sample import FigureSampleMeta
from ..utils import write_json, read_json
from ..schemas.data.dataset_sample import (
    ImagePairFileStructure,
    ImagePair,
    SiameseDsSample,
    SiameseDsWithRegressionSample,
    CenterRegressionSample,
)
from .fugure_sample_generation import SampleGeneratorInterface
from torchvision.transforms.functional import to_tensor
import numpy as np
import cv2


def write_image_pair(image_pair: ImagePair, output_path: Path) -> None:
    """Saves image pair to disk"""

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_structure = ImagePairFileStructure(output_path)
    cv2.imwrite(str(file_structure.img1_path), image_pair.img1)
    cv2.imwrite(str(file_structure.img2_path), image_pair.img2)

    write_json(
        json_file=file_structure.img1_meta_path,
        data=image_pair.img1_meta.model_dump(),
    )
    write_json(
        json_file=file_structure.img2_meta_path,
        data=image_pair.img2_meta.model_dump(),
    )


def read_image_pair(root_folder: DirectoryPath) -> ImagePair:
    """Reads image pair from file strucutre specified in `ImagePairFileStructure`"""

    def _read_rgb(path: FilePath) -> np.ndarray:
        bgr_img = cv2.imread(str(path))
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    file_structure = ImagePairFileStructure(root_folder)
    img1 = _read_rgb(file_structure.img1_path)
    img2 = _read_rgb(file_structure.img2_path)
    img1_meta = FigureSampleMeta.model_validate(
        read_json(file_structure.img1_meta_path)
    )
    img2_meta = FigureSampleMeta.model_validate(
        read_json(file_structure.img2_meta_path)
    )
    img_pair = ImagePair(img1=img1, img2=img2, img1_meta=img1_meta, img2_meta=img2_meta)
    return img_pair


def generate_img_pair(generator: SampleGeneratorInterface) -> ImagePair:
    img1, img1_meta = generator.generate()
    img2, img2_meta = generator.generate()
    img_pair = ImagePair(img1=img1, img1_meta=img1_meta, img2=img2, img2_meta=img2_meta)
    return img_pair


def write_dataset(
    generator: SampleGeneratorInterface, samples_num: int, output_folder: Path
) -> None:
    for i in tqdm(range(samples_num), desc="Writing dataset"):
        pair_folder = output_folder / f"pair_{i:06d}"
        image_pair = generate_img_pair(generator)
        write_image_pair(image_pair, output_path=pair_folder)


class ImagePairReader(Dataset):
    """
    Generic dataset for images pair reading
    Expects data to be organized in the following way
    └── dataset_root   <- Image pair root folder
        └── pair_000000   <- Image pair 0 root folder
            ├── img1.png         <- path to first image
            ├── img2.png         <- path to second image
            ├── img1_meta.json   <- json with first image meta
            ├── img2_meta.json   <- json with second image meta
        └── pair_000001   <- Image pair 1 root folder
            ├── img1.png         <- path to first image
            ├── img2.png         <- path to second image
            ├── img1_meta.json   <- json with first image meta
            ├── img2_meta.json   <- json with second image meta
        ...
    """

    def __init__(self, dataset_root: DirectoryPath):
        super().__init__()
        self._dataset_root = dataset_root
        self._image_pairs_folders: List[DirectoryPath] = [
            self._dataset_root / folder
            for folder in sorted(os.listdir(self._dataset_root))
        ]

    def __len__(self) -> int:
        return len(self._image_pairs_folders)

    def __getitem__(self, index):
        return super().__getitem__(index)


class SiameseSamplesDatasetReader(ImagePairReader):
    def __init__(self, dataset_root: DirectoryPath):
        super().__init__(dataset_root)

    @staticmethod
    def collate_samples(batch: List[SiameseDsSample]) -> SiameseDsSample:
        collated_batch = SiameseDsSample(
            img1=default_collate([sample.img1 for sample in batch]),
            img2=default_collate([sample.img2 for sample in batch]),
            label=default_collate([sample.label for sample in batch]),
        )
        return collated_batch

    def _construct_siamese_ds_sample(self, img_pair: ImagePair) -> SiameseDsSample:
        label: int = img_pair.img1_meta.figure_kind == img_pair.img2_meta.figure_kind
        sample = SiameseDsSample(
            img1=to_tensor(img_pair.img1),
            img2=to_tensor(img_pair.img2),
            label=torch.tensor(label, dtype=torch.float32),
        )
        return sample

    def __getitem__(self, index) -> SiameseDsSample:
        img_pair: ImagePair = read_image_pair(self._image_pairs_folders[index])
        siamese_sample = self._construct_siamese_ds_sample(img_pair)
        return siamese_sample


class SiameseSamplesWithRegressionDatasetReader(ImagePairReader):
    def __init__(self, dataset_root: DirectoryPath):
        super().__init__(dataset_root)

    @staticmethod
    def collate_samples(
        batch: List[SiameseDsWithRegressionSample],
    ) -> SiameseDsWithRegressionSample:
        collated_batch = SiameseDsWithRegressionSample(
            img1=default_collate([sample.img1 for sample in batch]),
            img2=default_collate([sample.img2 for sample in batch]),
            label=default_collate([sample.label for sample in batch]),
            regression_target1=default_collate(
                [sample.regression_target1 for sample in batch]
            ),
            regression_target2=default_collate(
                [sample.regression_target2 for sample in batch]
            ),
        )
        return collated_batch

    def _get_object_center_norm_coordinates(
        self, img: np.ndarray, figure_sample_info: FigureSampleMeta
    ) -> torch.Tensor:
        img_h, img_w = img.shape[:2]
        x_center_norm: float = figure_sample_info.figure_meta.x_center_pxl / img_w
        y_center_norm: float = figure_sample_info.figure_meta.y_center_pxl / img_h
        return torch.tensor([x_center_norm, y_center_norm])

    def _construct_sample_with_regression_target(
        self, img_pair: ImagePair
    ) -> SiameseDsWithRegressionSample:
        label: int = img_pair.img1_meta.figure_kind == img_pair.img2_meta.figure_kind

        sample = SiameseDsWithRegressionSample(
            img1=to_tensor(img_pair.img1),
            img2=to_tensor(img_pair.img2),
            regression_target1=self._get_object_center_norm_coordinates(
                img=img_pair.img1, figure_sample_info=img_pair.img1_meta
            ),
            regression_target2=self._get_object_center_norm_coordinates(
                img=img_pair.img2, figure_sample_info=img_pair.img2_meta
            ),
            label=label,
        )
        return sample

    def __getitem__(self, index):
        img_pair: ImagePair = read_image_pair(self._image_pairs_folders[index])
        siamese_sample = self._construct_sample_with_regression_target(img_pair)
        return siamese_sample


class CenterRegressionDatasetReader(Dataset):
    """Test dataset for center regression feasibility test"""

    def __init__(self, dataset_root: DirectoryPath):
        super().__init__()
        self._dataset_root = dataset_root
        self._image_pairs_folders: List[DirectoryPath] = [
            self._dataset_root / folder
            for folder in sorted(os.listdir(self._dataset_root))
        ]

    @staticmethod
    def collate_samples(
        batch: List[CenterRegressionSample],
    ) -> CenterRegressionSample:
        collated_batch = CenterRegressionSample(
            image=default_collate([sample.image for sample in batch]),
            regression_target=default_collate(
                [sample.regression_target for sample in batch]
            ),
        )
        return collated_batch

    def __len__(self) -> int:
        return len(self._image_pairs_folders) * 2

    def _get_object_center_norm_coordinates(
        self, img: np.ndarray, figure_sample_info: FigureSampleMeta
    ) -> torch.Tensor:
        img_h, img_w = img.shape[:2]
        x_center_norm: float = figure_sample_info.figure_meta.x_center_pxl / img_w
        y_center_norm: float = figure_sample_info.figure_meta.y_center_pxl / img_h
        return torch.tensor([x_center_norm, y_center_norm])

    def __getitem__(self, index) -> CenterRegressionSample:
        read_pair_index = int(index // 2)
        img_pair: ImagePair = read_image_pair(
            self._image_pairs_folders[read_pair_index]
        )
        read_first_image = index % 2 == 0
        sample: CenterRegressionSample
        if read_first_image:
            sample = CenterRegressionSample(
                image=to_tensor(img_pair.img1),
                regression_target=self._get_object_center_norm_coordinates(
                    img=img_pair.img1, figure_sample_info=img_pair.img1_meta
                ),
            )
        else:
            sample = CenterRegressionSample(
                image=to_tensor(img_pair.img2),
                regression_target=self._get_object_center_norm_coordinates(
                    img=img_pair.img2, figure_sample_info=img_pair.img2_meta
                ),
            )
        return sample
