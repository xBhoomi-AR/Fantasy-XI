from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Experiment
    experiment_name: str = "baseline_lstm"
    seed: int = 42

    # Data
    sequence_length: int = 5

    # Features will be defined after inspecting the final dataset
    sequence_features: List[str] = field(default_factory=list)
    context_features: List[str] = field(default_factory=list)

    # Model
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    fc_hidden_size: int = 32

    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 100
    patience: int = 10

    # Loss
    loss_function: str = "mse"


def get_default_config():
    return ExperimentConfig()