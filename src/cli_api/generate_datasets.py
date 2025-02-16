from ..consts import (
    DATASET_GENERATION_SEED,
    TRAIN_DATASET_ROOT,
    VAL_DATASET_ROOT,
    TEST_TEXTURE_ROOT,
    TEST_COLOR_ROOT,
    TRAIN_DATASET_SIZE,
    VAL_DATASET_SIZE,
    TEST_COLOR_SIZE,
    TEST_TEXTURE_SIZE,
)
from ..data.fugure_sample_generation import (
    init_tab10_colors_generator,
    init_texture_generator,
)
from ..data.dataset import write_dataset
from pytorch_lightning import seed_everything


COLORS_GENERATOR = init_tab10_colors_generator()
TEXTURE_GENERATOR = init_texture_generator()


def generate_datasets() -> None:
    # train
    write_dataset(
        generator=COLORS_GENERATOR,
        output_folder=TRAIN_DATASET_ROOT,
        samples_num=TRAIN_DATASET_SIZE,
    )
    # val
    write_dataset(
        generator=COLORS_GENERATOR,
        output_folder=VAL_DATASET_ROOT,
        samples_num=VAL_DATASET_SIZE,
    )
    # test (colors)
    write_dataset(
        generator=COLORS_GENERATOR,
        output_folder=TEST_COLOR_ROOT,
        samples_num=TEST_COLOR_SIZE,
    )
    # test (textures)
    write_dataset(
        generator=TEXTURE_GENERATOR,
        output_folder=TEST_TEXTURE_ROOT,
        samples_num=TEST_TEXTURE_SIZE,
    )


if __name__ == "__main__":
    seed_everything(DATASET_GENERATION_SEED)
    generate_datasets()
