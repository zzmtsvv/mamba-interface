import math
from dataclasses import dataclass


@dataclass
class mamba_config:
    hidden_dim: int
    num_layers: int
    input_dim: int
    latent_state_dim: int
    delta_rank: int = math.ceil(hidden_dim / 16)
    inner_dim: int = 2 * hidden_dim
    conv_dim: int = 4
    ...
