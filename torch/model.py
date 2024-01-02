import torch
from torch import nn
from modules import ResidualMambaBlock, RMSNorm


class Mamba(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 inner_dim: int,
                 conv_dim: int,
                 latent_state_dim: int,
                 delta_rank: int,
                 linear_bias: bool = False,
                 conv_bias: bool = True) -> None:
        super().__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        layers = [
            ResidualMambaBlock(hidden_dim, inner_dim, conv_dim, latent_state_dim,
                          delta_rank, linear_bias, conv_bias) for _ in range(num_layers)
        ]
        self.layers = nn.Sequential(*layers)
        self.rmsnorm = RMSNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, input_dim, bias=False)
        self.tie_weights()
    
    def tie_weights(self):
        # https://arxiv.org/pdf/1608.05859v3.pdf
        self.head.weight = self.embedding.weight
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        '''
            inputs: Tensor of shape [batch_size, sequence_length]
            logits: Tensor of shape [batch_size, sequence_length, input_dim]

            input_dim can be interpreted as vocab_size in the language modeling task.
            in this case the inputs will be input_ids.
        '''
        x = self.embedding(inputs)
        x = self.layers(x)
        x = self.rmsnorm(x)
        
        return self.head(x)
