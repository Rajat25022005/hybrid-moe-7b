"""
Partial Rotary Positional Embedding (RoPE).
Applied to the last `rope_dim` dimensions only, following DeepSeek V4.
For CSA/HCA: also applied to KV entries and core attention outputs (with negation).
"""

import torch
import torch.nn as nn
import math


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """
    Apply RoPE to the last `rope_dim` dimensions of x.
    
    Args:
        x: (..., head_dim) tensor
        freqs_cis: (seq_len, rope_dim//2) complex tensor
        rope_dim: number of dimensions to apply RoPE to (from the end)
    
    Returns:
        Tensor with RoPE applied to last rope_dim dimensions
    """
    # Split into non-rotary and rotary parts
    x_pass = x[..., :-rope_dim] if rope_dim < x.shape[-1] else torch.tensor([], device=x.device)
    x_rot = x[..., -rope_dim:]

    # Reshape for complex multiplication
    x_rot_complex = torch.view_as_complex(
        x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)
    )

    # Broadcast freqs_cis to match x shape
    # freqs_cis shape: (seq_len, rope_dim//2)
    # x_rot_complex shape: (..., seq_len, rope_dim//2)
    ndim = x_rot_complex.ndim
    shape = [1] * (ndim - 2) + list(freqs_cis.shape)
    freqs_cis = freqs_cis.view(*shape)

    x_rotated = torch.view_as_real(x_rot_complex * freqs_cis).flatten(-2)
    x_rotated = x_rotated.type_as(x)

    if x_pass.numel() > 0:
        return torch.cat([x_pass, x_rotated], dim=-1)
    return x_rotated


def apply_rotary_emb_with_positions(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """
    Apply RoPE with arbitrary position indices.
    Used for compressed KV entries and output de-rotation.
    
    Args:
        x: (..., head_dim) tensor
        freqs_cis: (max_seq_len, rope_dim//2) precomputed frequencies
        positions: (...) integer positions to index into freqs_cis
        rope_dim: number of dimensions to apply RoPE to
    """
    x_pass = x[..., :-rope_dim] if rope_dim < x.shape[-1] else torch.tensor([], device=x.device)
    x_rot = x[..., -rope_dim:]

    x_rot_complex = torch.view_as_complex(
        x_rot.float().reshape(*x_rot.shape[:-1], -1, 2)
    )

    # Gather freqs for the given positions
    pos_freqs = freqs_cis[positions.long()]  # (..., rope_dim//2)

    x_rotated = torch.view_as_real(x_rot_complex * pos_freqs).flatten(-2)
    x_rotated = x_rotated.type_as(x)

    if x_pass.numel() > 0:
        return torch.cat([x_pass, x_rotated], dim=-1)
    return x_rotated


class PartialRoPE(nn.Module):
    """
    Partial RoPE module that applies rotary embeddings to the last `rope_dim`
    dimensions only, as specified in DeepSeek V4.
    """

    def __init__(self, rope_dim: int = 64, max_seq_len: int = 131072, theta: float = 10000.0):
        super().__init__()
        self.rope_dim = rope_dim
        self.max_seq_len = max_seq_len
        # Precompute and register as buffer (not a parameter)
        freqs_cis = precompute_freqs_cis(rope_dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE to sequential positions starting from start_pos."""
        seq_len = x.shape[-2]
        freqs = self.freqs_cis[start_pos: start_pos + seq_len]
        return apply_rotary_emb(x, freqs, self.rope_dim)

    def forward_with_positions(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE with arbitrary position indices."""
        return apply_rotary_emb_with_positions(x, self.freqs_cis, positions, self.rope_dim)
