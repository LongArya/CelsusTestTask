from enum import Enum
from typing import Dict
from pydantic import BaseModel


class OptimizerKind(Enum):
    SGD = "SGD"
    ADAM = "ADAM"


class TrainConfig(BaseModel):
    # arch
    embedding_size: int

    # opt
    optimizer_kind: OptimizerKind
    optimizer_kwargs: Dict[str, float]

    # training
    batch_size: int
    epochs_num: int

    # log
    clearml_project: str  # "CelsusTestTask"
    clearml_task_name: str
    log_samples_num: int

    class Config:
        use_enum_values = True

    def model_post_init(self, __context):
        if isinstance(self.optimizer_kind, str):
            self.optimizer_kind = OptimizerKind[self.optimizer_kind]
