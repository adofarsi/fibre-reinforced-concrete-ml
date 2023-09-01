import sys
import json
import logging
from dataclasses import dataclass


@dataclass
class ModelConfig:

    # physics_params
    E = 30e3
    nu = 0.3
    K_fold = 3
    # Directories
    data_dir: str = ""
    model_dir: str = ""
    model_version: str = ""

    # Model architecture
    model: str = "encoder-decoder"
    input_shape: int = 1
    dropout: float = 0.0
    device: str = "cpu"

    # Evaluation
    eval_set: str = ""
    max_eval_steps: int = 5000

    # Dataset
    dataset: str = "heat_conductivity"
    save_folder = ''
    best_model_name = ''
    # Optimisation
    alpha: float = 1e-22
    epochs: int = 250
    batch_size: int = 1
    learning_rate: float = 1e-3
    evaluation_metric: str = "L2"
    WEIGHT_DECAY = 5e-4
    
    def __post_init__(self):

        assert self.model in {"encoder-decoder", "cnn", "mlp", "cnn1d", "rnn", "mlp_cohesive", "rnn_cohesive", "cnn1d_cohesive"}

        # if self.batch_size != 1:
        #     # This can easily be implemented by using Firedrake ensemble parallelism.
        #     # Ensemble parallelism is critical if the Firedrake operator composed with PyTorch is expensive to evaluate, e.g. when solving a PDE.
        #     raise NotImplementedError("Batch size > 1 necessitates using Firedrake ensemble parallelism. See https://www.firedrakeproject.org/parallelism.html#ensemble-parallelism")

    def add_input_shape(self, input_shape: int):
        self.input_shape = input_shape

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, "r") as f:
            cfg = json.load(f)
            return cls(**cfg)

    def to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)


def get_logger(name: str = "main"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
