import clearml
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ..analytics.utils import get_best_f1_score_point
from ..schemas.metrics.pr_curve import PRCurvePoint
from ..schemas.data.dataset_sample import SiameseDsSample
from ..schemas.config.train_config import TrainConfig, OptimizerKind
from ..data.dataset import SiameseSamplesDatasetReader
from ..nn.models import VanillaSiameseNetwork
from .log_utils import (
    init_clearml_task,
    log_dataset_samples_to_clearml,
    plot_siamese_sample,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from ..utils import read_yaml
from ..consts import TRAIN_DATASET_ROOT, VAL_DATASET_ROOT, TEST_COLOR_ROOT
from sklearn.metrics import precision_recall_curve


def preprocess_label_for_cosine_emb_loss(label: torch.Tensor) -> None:
    """Converts labels from 0/1 to -1/1"""

    label = torch.where(
        label == 0, torch.tensor(-1, device=label.device, dtype=label.dtype), label
    )
    return label


class VanillaSiameseLightningModule(pl.LightningModule):
    """Lightning module for training Vanilla siamese model"""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self._config = config
        self._model = VanillaSiameseNetwork(embedding_size=config.embedding_size)
        self._val_ds_cosine_similarities: torch.Tensor = torch.empty((0,))
        self._val_ds_labels: torch.Tensor = torch.empty((0,))
        self._cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        output = self._model(img1, img2)
        return output

    def training_step(self, batch: SiameseDsSample, batch_idx: int) -> torch.Tensor:
        img1, img2, label = batch.img1, batch.img2, batch.label
        label = preprocess_label_for_cosine_emb_loss(label)
        emb1, emb2 = self(img1, img2)
        cosine_loss = self._cosine_loss(emb1, emb2, label)
        loss_value = cosine_loss.cpu().item()
        self.log_dict(
            {
                "cosine_loss": loss_value,
            }
        )
        return cosine_loss

    def validation_step(self, val_batch: SiameseDsSample, batch_idx: int):
        """Accumulate targets and prediction to compute metrics at the end"""

        emb1, emb2 = self(val_batch.img1, val_batch.img2)
        cos_sim = torch.cosine_similarity(emb1, emb2, dim=-1).to(
            self._val_ds_cosine_similarities.device
        )
        label = val_batch.label.to(self._val_ds_cosine_similarities.device)

        self._val_ds_labels = torch.cat((self._val_ds_labels, label))
        self._val_ds_cosine_similarities = torch.cat(
            (self._val_ds_cosine_similarities, cos_sim)
        )

    def on_train_epoch_end(self):
        loss_value = self.trainer.logged_metrics["cosine_loss"]
        clearml.Logger.current_logger().report_scalar(
            title="Training",
            series="cosine_loss",
            value=loss_value,
            iteration=self.current_epoch,
        )
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        # compute F1 score
        prcn, rcl, thresholds = precision_recall_curve(
            y_true=self._val_ds_labels.cpu().numpy(),
            probas_pred=self._val_ds_cosine_similarities.cpu().numpy(),
        )
        best_point = get_best_f1_score_point(prcn, rcl, thresholds)

        # log all metrics
        self.log_dict({"best_F1": best_point.f1})
        clearml.Logger.current_logger().report_scalar(
            title="Val metrics",
            series="prcn@best_F1",
            value=best_point.prcn,
            iteration=self.current_epoch,
        )
        clearml.Logger.current_logger().report_scalar(
            title="Val metrics",
            series="rcl@best_F1",
            value=best_point.rcl,
            iteration=self.current_epoch,
        )
        clearml.Logger.current_logger().report_scalar(
            title="Val metrics",
            series="thrd@best_F1",
            value=best_point.thrd,
            iteration=self.current_epoch,
        )
        clearml.Logger.current_logger().report_scalar(
            title="Val metrics",
            series="best_F1",
            value=best_point.f1,
            iteration=self.current_epoch,
        )

        self._val_ds_cosine_similarities: torch.Tensor = torch.empty((0,))
        self._val_ds_labels: torch.Tensor = torch.empty((0,))

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
    train_dataset: SiameseSamplesDatasetReader,
    val_dataset: SiameseSamplesDatasetReader,
    config: TrainConfig,
):
    # log samples
    task = init_clearml_task(config.clearml_project, config.clearml_task_name)
    task.connect_configuration(config.model_dump())
    log_dataset_samples_to_clearml(
        dataset=train_dataset,
        title="Debug samples",
        series="train",
        plot_function=plot_siamese_sample,
        samples_num=config.log_samples_num,
    )
    log_dataset_samples_to_clearml(
        dataset=val_dataset,
        title="Debug samples",
        series="val",
        plot_function=plot_siamese_sample,
        samples_num=config.log_samples_num,
    )

    # run train
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=SiameseSamplesDatasetReader.collate_samples,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=SiameseSamplesDatasetReader.collate_samples,
        shuffle=False,
    )
    model = VanillaSiameseLightningModule(config)
    model_ckpt_callback = ModelCheckpoint(
        monitor="best_F1",
        mode="max",
        auto_insert_metric_name=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        filename="checkpoint_{epoch:02d}-{best_F1:.3f}",
        save_top_k=10,
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
        "E:\\dev\\CelsusWorkspace\\CelsusTestTask\\configs\\vanilla_siamese_train.yml"
    )
    config = TrainConfig.model_validate(config_data)
    train_ds = SiameseSamplesDatasetReader(TRAIN_DATASET_ROOT)
    val_ds = SiameseSamplesDatasetReader(VAL_DATASET_ROOT)
    run_train(train_dataset=train_ds, val_dataset=val_ds, config=config)


if __name__ == "__main__":
    debug_training_pipeline()
