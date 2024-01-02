from typing import Any
import jax
from flax import linen as nn
from modules import ResidualMambaBlock, RMSNorm


class Mamba(nn.Module):
    input_dim: int
    num_layers: int
    hidden_dim: int
    inner_dim: int
    conv_dim: int
    latent_state_dim: int
    delta_rank: int
    linear_bias: bool = False
    conv_bias: bool = True

    def setup(self):
        self.embedding = nn.Embed(self.input_dim, self.hidden_dim)
        layers = [
            ResidualMambaBlock(self.hidden_dim, self.inner_dim, self.conv_dim,
                               self.latent_state_dim, self.delta_rank, self.linear_bias,
                               self.conv_bias) for _ in range(self.num_layers)
        ]
        self.layers = nn.Sequential(layers)
        self.rmsnorm = RMSNorm(self.hidden_dim)

        # there also should be head layer but it is omitted due to 
        # jax issues and common practice to tie weights of the `first` and `last`
        # linear layers in language models. see more in `tie_weights` method right below
    
    def tie_weights(self, x: jax.Array) -> jax.Array:
        # https://arxiv.org/pdf/1608.05859v3.pdf
        return self.embedding.attend(x)
    
    def __call__(self, inputs: jax.Array) -> jax.Array:
        '''
            inputs: Tensor of shape [batch_size, sequence_length]
            logits: Tensor of shape [batch_size, sequence_length, input_dim]

            input_dim can be interpreted as vocab_size in the language modeling task.
            in this case the inputs will be input_ids.
        '''
        x = self.embedding(inputs)
        x = self.layers(x)
        x = self.rmsnorm(x)
        logits = self.tie_weights(x)

        return logits
