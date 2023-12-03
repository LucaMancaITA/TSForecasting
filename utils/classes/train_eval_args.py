
# Import modules
from dataclasses import dataclass


@dataclass
class TrainEvalArgs:
    """Train and evaluation arguments."""
    batch_size: int
    epochs: int
    learning_rate: float
    use_amp: bool
    checkpoints: str
    architecture: str
    model_name: str
