"""
KV Compressor — shared compression logic for both CSA and HCA.

CSA mode: Overlapped compression with two branches (C_a, C_b).
HCA mode: Simple single-branch compression with higher compression rate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCompressor(nn.Module):
    """
    Compresses KV entries along the sequence dimension.
    
    CSA: Uses overlapped compression with two branches, each KV entry
    derived from 2m tokens (current window + previous window overlap).
    
    HCA: Simple block compression, no overlap, higher compression rate m'.
    """

    def __init__(
        self,
        hidden_dim: int,
        kv_dim: int,
        compress_rate: int,
        overlapped: bool = True,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension d
            kv_dim: KV entry dimension c (head_dim for CSA/HCA)
            compress_rate: m for CSA, m' for HCA
            overlapped: True for CSA (two branches), False for HCA
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kv_dim = kv_dim
        self.compress_rate = compress_rate
        self.overlapped = overlapped

        if overlapped:
            # CSA: two branches for overlapped compression
            self.W_a_kv = nn.Linear(hidden_dim, kv_dim, bias=False)
            self.W_b_kv = nn.Linear(hidden_dim, kv_dim, bias=False)
            self.W_a_z = nn.Linear(hidden_dim, kv_dim, bias=False)
            self.W_b_z = nn.Linear(hidden_dim, kv_dim, bias=False)
            # Learnable positional biases for compression weights
            self.bias_a = nn.Parameter(torch.zeros(compress_rate, kv_dim))
            self.bias_b = nn.Parameter(torch.zeros(compress_rate, kv_dim))
        else:
            # HCA: single branch
            self.W_kv = nn.Linear(hidden_dim, kv_dim, bias=False)
            self.W_z = nn.Linear(hidden_dim, kv_dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(compress_rate, kv_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compress KV entries.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            
        Returns:
            compressed_kv: (batch, seq_len // compress_rate, kv_dim)
        """
        if self.overlapped:
            return self._compress_overlapped(hidden_states)
        else:
            return self._compress_simple(hidden_states)

    def _compress_simple(self, H: torch.Tensor) -> torch.Tensor:
        """HCA-style simple block compression."""
        B, N, D = H.shape
        m = self.compress_rate

        # Pad sequence to be divisible by m
        pad_len = (m - N % m) % m
        if pad_len > 0:
            H = F.pad(H, (0, 0, 0, pad_len))
            N = H.shape[1]

        num_blocks = N // m

        # Compute KV entries and compression weights
        C = self.W_kv(H)           # (B, N, c)
        Z = self.W_z(H)            # (B, N, c)

        # Reshape into blocks
        C = C.view(B, num_blocks, m, -1)    # (B, num_blocks, m, c)
        Z = Z.view(B, num_blocks, m, -1)    # (B, num_blocks, m, c)

        # Add positional bias and compute softmax weights
        Z = Z + self.bias.unsqueeze(0).unsqueeze(0)  # broadcast bias
        S = F.softmax(Z, dim=2)     # (B, num_blocks, m, c) — softmax over m dimension per channel

        # Weighted sum (Hadamard product + sum)
        C_comp = (S * C).sum(dim=2)  # (B, num_blocks, c)

        return C_comp

    def _compress_overlapped(self, H: torch.Tensor) -> torch.Tensor:
        """CSA-style overlapped compression with two branches."""
        B, N, D = H.shape
        m = self.compress_rate

        # Pad sequence
        pad_len = (m - N % m) % m
        if pad_len > 0:
            H = F.pad(H, (0, 0, 0, pad_len))
            N = H.shape[1]

        num_blocks = N // m

        # Compute two branches of KV entries and weights
        C_a = self.W_a_kv(H)  # (B, N, c)
        C_b = self.W_b_kv(H)  # (B, N, c)
        Z_a = self.W_a_z(H)   # (B, N, c)
        Z_b = self.W_b_z(H)   # (B, N, c)

        # Reshape into blocks of size m
        C_a = C_a.view(B, num_blocks, m, -1)  # (B, num_blocks, m, c)
        C_b = C_b.view(B, num_blocks, m, -1)
        Z_a = Z_a.view(B, num_blocks, m, -1)
        Z_b = Z_b.view(B, num_blocks, m, -1)

        # Add positional biases
        Z_a = Z_a + self.bias_a.unsqueeze(0).unsqueeze(0)
        Z_b = Z_b + self.bias_b.unsqueeze(0).unsqueeze(0)

        # For block i: use C_a[m*i : m*(i+1)] and C_b[m*(i-1) : m*i]
        # Shift C_b and Z_b by one block (previous window overlap)
        C_b_shifted = torch.cat([
            torch.zeros(B, 1, m, self.kv_dim, device=H.device, dtype=C_b.dtype),
            C_b[:, :-1]
        ], dim=1)
        Z_b_shifted = torch.cat([
            torch.full((B, 1, m, self.kv_dim), float('-inf'), device=H.device, dtype=Z_b.dtype),
            Z_b[:, :-1]
        ], dim=1)

        # Concatenate along the m dimension: [Z_a; Z_b_shifted] for joint softmax
        Z_cat = torch.cat([Z_a, Z_b_shifted], dim=2)  # (B, num_blocks, 2m, c)
        C_cat = torch.cat([C_a, C_b_shifted], dim=2)   # (B, num_blocks, 2m, c)

        # Softmax across 2m entries per channel
        S = F.softmax(Z_cat, dim=2)  # (B, num_blocks, 2m, c)

        # Weighted sum
        C_comp = (S * C_cat).sum(dim=2)  # (B, num_blocks, c)

        return C_comp
