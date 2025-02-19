import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from ..data.dataset import SiameseSamplesDatasetReader
from ..train.siamese_network_with_reg_train import (
    SiameseNetworkWithRegressionLightningModule,
)
from ..train.vanilla_siamese_network_train import VanillaSiameseLightningModule
from typing import Union
from ..consts import TEST_COLOR_ROOT, MODELS_ROOT, TRAIN_DATASET_ROOT, VAL_DATASET_ROOT
from ..schemas.config.train_config import TrainConfig, SiameseWithRegressionTrainConfig
from ..schemas.data.dataset_sample import (
    SiameseDsSample,
    SiameseDsWithPredsSample,
)
from typing import List, Optional, Tuple
from ..utils import read_yaml
from matplotlib.pyplot import Axes
from collections import Counter
from torch.nn import CosineEmbeddingLoss


class SiameseNetowrkPredsMemCacher(Dataset):
    """Simple cacher implementation that computes predictions once and stores them in memory"""

    def __init__(
        self,
        base_dataset: SiameseSamplesDatasetReader,
        model: Union[
            SiameseNetworkWithRegressionLightningModule, VanillaSiameseLightningModule
        ],
    ):
        super().__init__()
        self._base_dataset = base_dataset
        self._model = model
        self._model.eval()
        self._predictions_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(len(self._base_dataset))
        ]

    def __len__(self):
        return len(self._base_dataset)

    @torch.no_grad()
    def __getitem__(self, index) -> SiameseDsWithPredsSample:
        sample: SiameseDsSample = self._base_dataset[index]

        emb1: torch.Tensor
        emb2: torch.Tensor
        if self._predictions_cache[index] is None:
            img1 = sample.img1.unsqueeze(0)
            img2 = sample.img2.unsqueeze(0)
            emb1, emb2 = self._model(img1, img2)
        else:
            emb1, emb2 = self._predictions_cache[index]

        output_sample = SiameseDsWithPredsSample(
            img1=sample.img1, img2=sample.img2, emb1=emb1, emb2=emb2, label=sample.label
        )
        return output_sample


def plot_cosine_similarity_distribution_for_each_label(
    ds: SiameseNetowrkPredsMemCacher, ax: Axes
) -> None:
    same_object_cos_sim = []
    diff_object_cos_sim = []
    for sample in ds:
        cos_sim = sample.get_cosine_similarity().item()
        if sample.label == 1:
            same_object_cos_sim.append(cos_sim)
        else:
            diff_object_cos_sim.append(cos_sim)
    ax.hist(same_object_cos_sim, alpha=0.5, label="same objects")
    ax.hist(diff_object_cos_sim, alpha=0.5, label="diff objects")
    ax.legend()
