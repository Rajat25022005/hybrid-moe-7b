"""
Mamba2 Block — Structured State Space Duality with gated architecture.
Pure PyTorch implementation for TPU compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mamba2.ssd import ssd_chunk_scan, ssd_recurrent
from src.model.normalization import RMSNorm


class Mamba2Block(nn.Module):
    """
    Mamba2 block using the Structured State Space Duality (SSD) framework.
    
    Architecture: input -> linear proj -> conv1d -> SSD core -> output proj
    with gated activation (SiLU gate).
    
    Uses scalar-times-identity A matrix for efficient matmul-based computation.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 128,
        conv_dim: int = 4,
        expand: int = 2,
        head_dim: int = 64,
        num_heads: int = 32,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.conv_dim = conv_dim
        self.expand = expand
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = expand * hidden_dim
        self.chunk_size = chunk_size

        # Input projection: project to inner_dim for X and gate
        self.in_proj = nn.Linear(hidden_dim, 2 * self.inner_dim, bias=False)

        # Short convolution (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=conv_dim,
            padding=conv_dim - 1,
            groups=self.inner_dim,
            bias=True,
        )

        # SSM parameters projection: from inner_dim to B, C, and dt (log decay)
        # B: (state_dim), C: (state_dim), A_log: (num_heads)
        self.ssm_proj = nn.Linear(
            self.inner_dim,
            state_dim + state_dim + num_heads,
            bias=False,
        )

        # A parameter (log-space, learnable, initialized to small negative values)
        self.A_log = nn.Parameter(
            torch.log(0.5 + 0.5 * torch.rand(num_heads))
        )

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        # Output projection
        self.out_proj = nn.Linear(self.inner_dim, hidden_dim, bias=False)

        # Normalization
        self.norm = RMSNorm(self.inner_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            state: optional recurrent state for inference
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        B, N, D = hidden_states.shape

        # 1. Input projection with gate
        xz = self.in_proj(hidden_states)  # (B, N, 2 * inner_dim)
        x, z = xz.chunk(2, dim=-1)       # each (B, N, inner_dim)

        # 2. Causal convolution
        x = x.transpose(1, 2)  # (B, inner_dim, N)
        x = self.conv1d(x)[:, :, :N]  # causal: trim to N
        x = x.transpose(1, 2)  # (B, N, inner_dim)
        x = F.silu(x)

        # 3. SSM parameter projection
        ssm_params = self.ssm_proj(x)  # (B, N, state_dim + state_dim + num_heads)
        B_mat = ssm_params[..., :self.state_dim]
        C_mat = ssm_params[..., self.state_dim:2*self.state_dim]
        dt = ssm_params[..., 2*self.state_dim:]  # (B, N, num_heads)

        # Combine learnable A with input-dependent dt
        A_log = -torch.exp(self.A_log).unsqueeze(0).unsqueeze(0)  # (1, 1, heads)
        A_log = A_log * F.softplus(dt)  # (B, N, heads)

        # 4. SSD scan (parallel for training)
        if N > 1:
            y = ssd_chunk_scan(
                B_mat=B_mat,
                C_mat=C_mat,
                X=x,
                A_log=A_log,
                chunk_size=self.chunk_size,
            )
        else:
            # Single token (inference mode)
            y, state = ssd_recurrent(B_mat, C_mat, x, A_log, state)

        # 5. Skip connection with D
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        # 6. Normalize and gate
        y = self.norm(y)
        y = y * F.silu(z)  # gated activation

        # 7. Output projection
        output = self.out_proj(y)

        return output
