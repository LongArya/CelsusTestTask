import click
from ..train.vanilla_siamese_network_train import run_train
from ..data.dataset import SiameseSamplesDatasetReader
from ..schemas.config.train_config import TrainConfig
from ..utils import read_yaml
from ..consts import TRAIN_DATASET_ROOT, VAL_DATASET_ROOT
from pathlib import Path


@click.command()
@click.option("--config_file", type=Path)
def main(config_file: Path):
    config_data = read_yaml(config_file)
    config = TrainConfig.model_validate(config_data)
    train_ds = SiameseSamplesDatasetReader(TRAIN_DATASET_ROOT)
    val_ds = SiameseSamplesDatasetReader(VAL_DATASET_ROOT)
    run_train(train_dataset=train_ds, val_dataset=val_ds, config=config)


if __name__ == "__main__":
    pass
