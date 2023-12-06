
# Import modules
from dataclasses import dataclass


@dataclass
class InformerArgs:
    """Informer model arguments."""
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int

@dataclass
class AutoformerArgs:
    """Autoformer model arguments."""
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int

@dataclass
class LstmArgs:
    """LSTM model arguments."""
    num_layers: int
    hidden_units: int
