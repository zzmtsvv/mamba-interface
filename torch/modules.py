import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat, rearrange, einsum


class RMSNorm(nn.Module):
    # https://arxiv.org/abs/1910.07467
    def __init__(self,
                 dim: int,
                 eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight
        return out


class MambaBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 inner_dim: int,
                 conv_dim: int,
                 latent_state_dim: int,
                 delta_rank: int,
                 linear_bias: bool = False,
                 conv_bias: bool = True) -> None:
        super().__init__()

        self.inner_dim = inner_dim
        self.delta_rank = delta_rank

        self.input_proj = nn.Linear(hidden_dim, inner_dim * 2, bias=linear_bias)
        self.conv = nn.Conv1d(in_channels=inner_dim,
                              out_channels=inner_dim,
                              kernel_size=conv_dim,
                              bias=conv_bias,
                              padding=conv_dim - 1,
                              groups=inner_dim)
        self.x_proj = nn.Linear(inner_dim, delta_rank + latent_state_dim * 2, bias=False)
        self.delta_proj = nn.Linear(delta_rank, inner_dim, bias=True)
        self.out_proj = nn.Linear(inner_dim, hidden_dim, bias=linear_bias)

        A = repeat(torch.arange(1, latent_state_dim + 1), 'n -> d n', d=inner_dim)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(inner_dim))

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        inner_dim, latent_dim = self.A_log.shape

        # A = -(self.A_log.float().exp()) idk what is the order of the operation so it is commented
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_proj = self.x_proj(x)
        
        delta, B, C = torch.split(x_proj, (self.delta_rank, latent_dim, latent_dim), dim=-1)
        delta = F.softplus(self.delta_proj(delta))

        # TODO: selective scan interface
        y = self.selective_ssm(x, delta, A, B, C, D)
        return y

    def selective_ssm(self,
                      x: torch.Tensor,
                      delta: torch.Tensor,
                      A: torch.Tensor,
                      B: torch.Tensor,
                      C: torch.Tensor,
                      D: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, inner_dim = x.shape
        latent_dim = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "bs len d_in, d_in ld -> bs len d_in ld"))
        deltaBx = einsum(delta, B, x, "bs len d_in, bs len ld, bs len d_in -> bs len d_in ld")

        u = torch.zeros((batch_size, inner_dim, latent_dim), device=deltaA.device)
        y = []
        for i in range(seq_len):
            u = deltaA[:, i] * u + deltaBx[:, i]
            yy = einsum(u, C[:, i, :], "bs d_in ld, bs ld -> bs d_in")
            y.append(yy)
        
        y = torch.stack(y, dim=1)
        y = y + u * D
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        x, meow = torch.split(self.input_proj(x), (self.inner_dim, self.inner_dim), dim=-1)
        x = rearrange(x, 'bs len d_in -> bs d_in len')
        x = self.conv(x)[:, :, :seq_len]
        x = rearrange(x, 'bs d_in len -> bs len d_in')

        x = F.silu(x)
        xx = self.ssm(x)
        xx = xx * F.silu(meow)
        return self.out_proj(xx)


class ResidualMambaBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 inner_dim: int,
                 conv_dim,
                 latent_state_dim: int,
                 delta_rank: int,
                 linear_bias: bool = False,
                 conv_bias: bool = True) -> None:
        super().__init__()

        self.mamba_block = MambaBlock(hidden_dim,
                                      inner_dim,
                                      conv_dim,
                                      latent_state_dim,
                                      delta_rank,
                                      linear_bias,
                                      conv_bias)
        self.rmsnorm = RMSNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba_block(self.rmsnorm(x)) + x
