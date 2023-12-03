
# Import modules
from dataclasses import dataclass


@dataclass
class DataArgs:
    """Data arguments."""
    root_path: str
    data_path: str
    input_len: int
    label_len: int
    pred_len: int
    features: str
    target: str
    scale: bool
    timeenc: int
    freq: str
