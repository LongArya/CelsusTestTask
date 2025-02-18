import random
import clearml
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from collections import namedtuple
from ..schemas.data.dataset_sample import CenterRegressionSample
from ..schemas.config.train_config import TrainConfig, OptimizerKind
from ..data.dataset import CenterRegressionDatasetReader
from ..nn.models import CenterRegressionModel
from .log_utils import (
    init_clearml_task,
    log_dataset_samples_to_clearml,
    plot_regression_sample,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MeanSquaredError
from ..utils import read_yaml
from ..consts import (
    TRAIN_DATASET_ROOT,
    VAL_DATASET_ROOT,
)


class CenterRegressionLightningModule(pl.LightningModule):
    """Lightning module for training Center Regression model"""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self._config = config
        self._model = CenterRegressionModel(embedding_size=config.embedding_size)
        self._val_ds_gt: torch.Tensor = torch.empty((0, 2))
        self._val_ds_predictions: torch.Tensor = torch.empty((0, 2))
        self._mse_metric = MeanSquaredError()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        output = self._model(img)
        return output

    def training_step(
        self, batch: CenterRegressionSample, batch_idx: int
    ) -> torch.Tensor:
        prediction = self(batch.image)
        mse_loss = F.mse_loss(prediction, batch.regression_target)
        loss_value = mse_loss.cpu().item()
        self.log_dict(
            {
                "mse_loss": loss_value,
            }
        )
        return mse_loss

    def validation_step(self, val_batch: CenterRegressionSample, batch_idx: int):
        """Accumulate targets and prediction to compute metrics at the end"""
        prediction = self(val_batch.image).to(self._val_ds_predictions.device)
        targets = val_batch.regression_target.to(self._val_ds_gt.device)

        self._val_ds_predictions = torch.cat((prediction, self._val_ds_predictions))
        self._val_ds_gt = torch.cat((targets, self._val_ds_gt))

    def on_train_epoch_end(self):
        loss_value = self.trainer.logged_metrics["mse_loss"]
        clearml.Logger.current_logger().report_scalar(
            title="Training",
            series="mse_loss",
            value=loss_value,
            iteration=self.current_epoch,
        )
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        val_mse_tnz: torch.Tensor = self._mse_metric(
            self._val_ds_predictions, self._val_ds_gt
        )
        val_mse_metric: float = val_mse_tnz.item()
        self.log("val_MSE", val_mse_metric)
        clearml.Logger.current_logger().report_scalar(
            title="Val metrics",
            series="MSE",
            value=val_mse_metric,
            iteration=self.current_epoch,
        )

        self._val_ds_gt: torch.Tensor = torch.empty((0, 2))
        self._val_ds_predictions: torch.Tensor = torch.empty((0, 2))

    def configure_optimizers(self):
        opt: torch.optim.Optimizer
        if self._config.optimizer_kind == OptimizerKind.SGD:
            opt = torch.optim.SGD(self.parameters(), **self._config.optimizer_kwargs)
        elif self._config.optimizer_kind == OptimizerKind.ADAM:
            opt = torch.optim.Adam(self.parameters(), **self._config.optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer {self._config.optimizer_kind}")
        return opt


def run_train(
    train_dataset: CenterRegressionModel,
    val_dataset: CenterRegressionModel,
    config: TrainConfig,
):
    # log samples
    task = init_clearml_task(config.clearml_project, config.clearml_task_name)
    task.connect_configuration(config.model_dump())
    log_dataset_samples_to_clearml(
        dataset=train_dataset,
        title="Debug samples",
        series="train",
        plot_function=plot_regression_sample,
        samples_num=config.log_samples_num,
    )
    log_dataset_samples_to_clearml(
        dataset=val_dataset,
        title="Debug samples",
        series="val",
        plot_function=plot_regression_sample,
        samples_num=config.log_samples_num,
    )

    # run train
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=CenterRegressionDatasetReader.collate_samples,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=CenterRegressionDatasetReader.collate_samples,
        shuffle=False,
    )
    model = CenterRegressionLightningModule(config)
    model_ckpt_callback = ModelCheckpoint(
        monitor="val_MSE",
        mode="min",
        auto_insert_metric_name=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        filename="checkpoint_{epoch:02d}-{val_MSE:.3f}",
        save_top_k=5,
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs_num,
        accelerator="auto",
        gpus=[0],
        check_val_every_n_epoch=1,
        callbacks=[model_ckpt_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


def debug_training_pipeline():
    config_data = read_yaml(
        "E:\\dev\\CelsusWorkspace\\CelsusTestTask\\configs\\train_center_regression.yml"
    )
    config = TrainConfig.model_validate(config_data)
    train_ds = CenterRegressionDatasetReader(TRAIN_DATASET_ROOT)
    val_ds = CenterRegressionDatasetReader(VAL_DATASET_ROOT)
    run_train(train_dataset=train_ds, val_dataset=val_ds, config=config)
