from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
from torchmetrics import MeanSquaredError
from ..data.dataset import CenterRegressionDatasetReader
from ..train.center_regressor_train import CenterRegressionLightningModule
from ..consts import TEST_TEXTURE_ROOT, TEST_COLOR_ROOT
from ..schemas.config.train_config import TrainConfig
from ..schemas.data.dataset_sample import (
    CenterRegressionSampleWithPrediction,
    CenterRegressionSample,
)
from typing import List, Optional
from ..train.log_utils import tnz_to_numpy
from pydantic import FilePath
from ..utils import read_yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class CenterRegressionPredsMemCacher(Dataset):
    """Simple cacher implementation that computes predictions once and stores them in memory"""

    def __init__(
        self,
        base_dataset: CenterRegressionDatasetReader,
        model: CenterRegressionLightningModule,
    ):
        super().__init__()
        self._base_dataset = base_dataset
        self._model = model
        self._model.eval()
        self._predictions_cache: List[Optional[torch.Tensor]] = [
            None for _ in range(len(self._base_dataset))
        ]

    def __len__(self):
        return len(self._base_dataset)

    @torch.no_grad()
    def __getitem__(self, index) -> CenterRegressionSampleWithPrediction:
        sample: CenterRegressionSample = self._base_dataset[index]
        prediction: torch.Tensor
        if self._predictions_cache[index] is None:
            img = sample.image.unsqueeze(0)
            prediction = self._model(img)[0]
        else:
            prediction = self._predictions_cache[index]

        output_sample = CenterRegressionSampleWithPrediction(
            image=sample.image,
            prediction=prediction,
            regression_target=sample.regression_target,
        )
        return output_sample


def plot_regression_gt_vs_pred(
    sample: CenterRegressionSampleWithPrediction, ax: Axes
) -> None:
    img = tnz_to_numpy(sample.image)
    h, w = img.shape[:2]
    gt_x, gt_y = sample.regression_target.numpy()
    pred_x, pred_y = sample.prediction.numpy()
    ax.imshow(img)
    ax.scatter(int(gt_x * w), int(gt_y * h), label="GT_center", c="g")
    ax.scatter(int(pred_x * w), int(pred_y * h), label="PRED_center", c="r")
    ax.set_title(f"Distance = {sample.get_distance_pixels().item():.2f} pixels")
    ax.legend()


def plot_top_10_predictions(
    dataset: CenterRegressionPredsMemCacher,
    mode="best",
) -> None:
    """PLots top 10 worst or best predictions, based on center distance to gt"""

    if mode not in ("best", "worst"):
        raise ValueError(f"Invalid mode {mode}")

    TOP_N = 10
    COLS = 2
    ROWS = 5

    distances = [sample.get_distance_pixels().item() for sample in dataset]
    top_n_ditsances_indexes: np.ndarray
    if mode == "best":
        top_n_ditsances_indexes = np.argsort(distances)[:TOP_N]
    else:
        top_n_ditsances_indexes = np.argsort(distances)[-TOP_N:][::-1]

    fig, axes = plt.subplots(COLS, ROWS, figsize=(20, 10))
    for plot_index, ds_index in enumerate(top_n_ditsances_indexes):
        ax_i, ax_j = np.unravel_index(plot_index, (COLS, ROWS))
        plot_regression_gt_vs_pred(dataset[ds_index], axes[ax_i, ax_j])

    return fig


def compute_MSE_on_dataset(dataset: CenterRegressionPredsMemCacher) -> float:
    y_true = torch.cat([sample.regression_target for sample in dataset], 0)
    y_pred = torch.cat([sample.prediction for sample in dataset], 0)
    mse = MeanSquaredError()(y_pred, y_true)
    return mse
