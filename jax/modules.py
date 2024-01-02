import jax
from flax import linen as nn
from jax import numpy as jnp
from associative_scan import associative_operator


class RMSNorm(nn.Module):
    # https://arxiv.org/abs/1910.07467
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        out = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)

        return out * weight


class MambaBlock(nn.Module):
    hidden_dim: int
    inner_dim: int
    conv_dim: int
    latent_state_dim: int
    delta_rank: int
    linear_bias: bool = False
    conv_bias: bool = True

    def setup(self) -> None:
        self.input_proj = nn.Dense(features=self.inner_dim * 2,
                                   kernel_init=nn.initializers.normal(),
                                   use_bias=self.linear_bias)
        self.conv = nn.Conv(features=self.inner_dim,
                            kernel_size=(self.conv_dim),
                            feature_group_count=self.inner_dim,
                            padding=self.conv_dim - 1,
                            use_bias=self.conv_bias)
        self.x_proj = nn.Dense(self.delta_rank + self.latent_state_dim * 2,
                               use_bias=False)
        self.delta_proj = nn.Dense(self.inner_dim, use_bias=True)
        self.out_proj = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.normal(),
                                 use_bias=self.linear_bias)
        
        A = jnp.tile(jnp.arange(1, self.latent_state_dim + 1), (self.inner_dim, 1))
        self.A_log = self.param("A_log", lambda rng, shape: jnp.log(A), (self.inner_dim, self.latent_state_dim))
        self.D = self.param("D", nn.initializers.ones, (self.inner_dim,))
    
    def ssm(self, x: jax.Array) -> jax.Array:
        inner_dim, latent_dim = self.A_log.shape

        A = -jnp.exp(self.A_log.astype(jnp.float32))
        D = self.D.astype(jnp.float32)

        x_proj = self.x_proj(x)

        delta, B, C = jnp.split(x_proj, (self.delta_rank, self.delta_rank + latent_dim), axis=-1)
        delta = jax.nn.softplus(self.delta_proj(delta))

        y = self.selective_ssm(x, delta, A, B, C, D)
        return y

    def selective_ssm(self,
                      x: jax.Array,
                      delta: jax.Array,
                      A: jax.Array,
                      B: jax.Array,
                      C: jax.Array,
                      D: jax.Array) -> jax.Array:
        batch_size, seq_len, inner_dim = x.shape
        latent_dim = A.shape[1]

        # discretization of the continuous parameters
        deltaA = jnp.exp(jnp.einsum("b s d, d l -> b s d l", delta, A))
        deltaBx = jnp.einsum("b s d, b s l, b s d -> b s d l", delta, B, x)
        
        # u = jnp.zeros((batch_size, inner_dim, latent_dim))
        # y = []
        # for i in range(seq_len):
        #     u = deltaA[:, i] * u + deltaBx[:, i]
        #     yy = jnp.einsum("b d l, b l -> b d", u, C[:, i, :])
        #     y.append(yy)
        
        # y = jnp.stack(y, axis=1)
        # y = y + u * D

        _, h = jax.lax.associative_scan(associative_operator, (deltaA, deltaBx), axis=1)
        h = jnp.stack(h, axis=0)
        y = jnp.einsum("b s d l, b s l -> b s d", h, C) + x * D

        return y
    
    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len, hidden_dim = x.shape
        
        x, meow = jnp.split(self.input_proj(x), indices_or_sections=(self.inner_dim,), axis=-1)
        x = self.conv(x)[:, :seq_len, :]

        x = jax.nn.silu(x)
        xx = self.ssm(x)
        xx = xx * jax.nn.silu(meow)
        return self.out_proj(xx)


class ResidualMambaBlock(nn.Module):
    hidden_dim: int
    inner_dim: int
    conv_dim: int
    latent_state_dim: int
    delta_rank: int
    linear_bias: bool = False
    conv_bias: bool = True

    def setup(self):
        self.mamba_block = MambaBlock(self.hidden_dim,
                                      self.inner_dim,
                                      self.conv_dim,
                                      self.latent_state_dim,
                                      self.delta_rank,
                                      self.linear_bias,
                                      self.conv_bias)
        self.rmsnorm = RMSNorm(self.hidden_dim)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.mamba_block(self.rmsnorm(x)) + x
