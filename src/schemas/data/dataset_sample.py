import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from .figure_sample import FigureSampleMeta


@dataclass
class ImagePair:
    img1: np.ndarray
    img2: np.ndarray
    img1_meta: FigureSampleMeta
    img2_meta: FigureSampleMeta


@dataclass
class ImagePairFileStructure:
    """
    Represents single image pair file structure on disk

    └── root   <- Image pair root folder
        │
        ├── img1.png         <- path to first image
        ├── img2.png         <- path to second image
        ├── img1_meta.json   <- json with first image meta
        ├── img2_meta.json   <- json with second image meta
    """

    root: Path

    @property
    def img1_path(self) -> Path:
        return self.root / "img1.png"

    @property
    def img2_path(self) -> Path:
        return self.root / "img2.png"

    @property
    def img1_meta_path(self) -> Path:
        return self.root / "img1_meta.json"

    @property
    def img2_meta_path(self) -> Path:
        return self.root / "img2_meta.json"


@dataclass
class SiameseDsSample:
    """Contains siamese sample/batch"""

    img1: torch.Tensor
    img2: torch.Tensor
    label: torch.Tensor  # single value: 1 if images are same else 0


@dataclass
class SiameseDsWithPredsSample(SiameseDsSample):
    emb1: torch.Tensor
    emb2: torch.Tensor

    def get_cosine_similarity(self) -> torch.Tensor:
        return torch.cosine_similarity(self.emb1, self.emb2, dim=-1)


@dataclass
class SiameseDsWithRegressionSample(SiameseDsSample):
    """
    Contains siamese sample/batch

    Additional info - regression target for each image,
    which is figure center in normalized coordinates
    """

    regression_target1: torch.Tensor
    regression_target2: torch.Tensor


@dataclass
class CenterRegressionSample:
    """
    Contains siamese sample/batch

    Additional info - regression target for each image,
    which is figure center in normalized coordinates
    """

    image: torch.Tensor
    regression_target: torch.Tensor


@dataclass
class CenterRegressionSampleWithPrediction(CenterRegressionSample):
    """Extends original interface with field for prediction"""

    prediction: torch.Tensor

    def get_distance_pixels(self) -> torch.Tensor:
        c, h, w = self.image.size()
        denorm_tnz = torch.tensor([w, h], dtype=torch.float32)
        distance = torch.linalg.vector_norm(
            self.prediction * denorm_tnz - self.regression_target * denorm_tnz
        )
        return distance
