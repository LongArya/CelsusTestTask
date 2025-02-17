import torch
import random as rd
from typing import Callable, Any
from torch.utils.data import Dataset
from clearml import Task, Logger
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ..schemas.data.dataset_sample import CenterRegressionSample


SAMPLE_PLOT_FUNCTION = Callable[[Any], Figure]


def init_clearml_task(project_name: str, task_name: str) -> Task:
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        reuse_last_task_id=False,
    )
    return task


def tnz_to_numpy(img_tensor_CHW: torch.Tensor) -> np.ndarray:
    image = img_tensor_CHW.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * 255
    image = image.astype(np.uint8)
    return image


def plot_regression_sample(sample: CenterRegressionSample) -> Figure:
    """Plots regression sample"""

    img = tnz_to_numpy(sample.image)
    h, w = img.shape[:2]
    x, y = sample.regression_target
    x = int(x * w)
    y = int(y * h)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.scatter(x, y, c="r", label="figure center")
    ax.legend()
    return fig


def log_dataset_samples_to_clearml(
    dataset: Dataset,
    title: str,
    series: str,
    plot_function: SAMPLE_PLOT_FUNCTION,
    samples_num: int,
) -> None:
    # select random indices
    logged_samples_num = min(len(dataset), samples_num)
    logged_samples_indices = rd.sample(list(range(len(dataset))), logged_samples_num)
    for clearml_iteration, logged_sample_index in enumerate(logged_samples_indices):
        sample = dataset[logged_sample_index]
        plt_figure: Figure = plot_function(sample)
        Logger.current_logger().report_matplotlib_figure(
            title=title, series=series, figure=plt_figure, iteration=clearml_iteration
        )
