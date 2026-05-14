"""
Structured State Space Duality (SSD) core for Mamba2.
Pure PyTorch implementation — TPU compatible, no CUDA kernels.

Implements the chunk-based parallel scan for efficient training.
The key insight: scalar-times-identity A matrix enables matmul-based computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def segsum(x: torch.Tensor) -> torch.Tensor:
    """
    Stable segment sum for computing cumulative products in log-space.
    Used for computing the diagonal blocks of the SSM.
    
    Args:
        x: (..., T) — log-space decay values
    Returns:
        (..., T, T) — lower-triangular cumulative sums
    """
    T = x.shape[-1]
    # Cumulative sum along last dimension
    x_cumsum = torch.cumsum(x, dim=-1)  # (..., T)
    # Create the lower-triangular matrix of segment sums
    # For entry (i, j) where j <= i: sum of x[j+1:i+1]
    x_cumsum_expanded = x_cumsum.unsqueeze(-1)  # (..., T, 1)
    x_cumsum_shifted = x_cumsum.unsqueeze(-2)    # (..., 1, T)
    result = x_cumsum_expanded - x_cumsum_shifted  # (..., T, T)
    # Zero out upper triangle and adjust diagonal
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    result = result.masked_fill(mask, float('-inf'))
    return result


def ssd_chunk_scan(
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    X: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunk-based SSD scan for parallel training.
    
    Splits the sequence into chunks and computes:
    1. Intra-chunk: quadratic attention within each chunk
    2. Inter-chunk: linear recurrence across chunks
    
    Args:
        B_mat: (batch, seq_len, state_dim) — input projection B
        C_mat: (batch, seq_len, state_dim) — output projection C
        X: (batch, seq_len, head_dim) — input after expansion
        A_log: (batch, seq_len, num_heads) — log decay rates (scalar per head)
        chunk_size: size of each chunk for the parallel scan
        
    Returns:
        Y: (batch, seq_len, head_dim) — output
    """
    batch, seq_len, d = X.shape
    n = B_mat.shape[-1]  # state_dim

    # Pad to chunk boundary
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        B_mat = F.pad(B_mat, (0, 0, 0, pad_len))
        C_mat = F.pad(C_mat, (0, 0, 0, pad_len))
        X = F.pad(X, (0, 0, 0, pad_len))
        A_log = F.pad(A_log, (0, 0, 0, pad_len))

    T = B_mat.shape[1]
    num_chunks = T // chunk_size
    L = chunk_size

    # Reshape into chunks: (batch, num_chunks, L, ...)
    B_chunks = B_mat.view(batch, num_chunks, L, n)
    C_chunks = C_mat.view(batch, num_chunks, L, n)
    X_chunks = X.view(batch, num_chunks, L, d)
    A_chunks = A_log.view(batch, num_chunks, L, -1)  # (batch, chunks, L, heads)

    # --- Intra-chunk computation (quadratic) ---
    # Compute cumulative decay within each chunk
    # A_chunks has shape (batch, chunks, L, heads)
    # For simplicity, average across heads for the decay
    a_mean = A_chunks.mean(dim=-1)  # (batch, chunks, L)

    # Segment sum for intra-chunk: L x L lower triangular
    decay_matrix = segsum(a_mean)  # (batch, chunks, L, L)
    decay_matrix = torch.exp(decay_matrix)

    # Intra-chunk attention: Y_intra = decay * (C @ B^T) @ X
    # B, C: (batch, chunks, L, n)
    # Attention matrix: (batch, chunks, L, L)
    attn = torch.einsum('bcln,bcmn->bclm', C_chunks, B_chunks)
    attn = attn * decay_matrix  # apply decay
    # Causal mask within chunk
    causal = torch.tril(torch.ones(L, L, device=X.device))
    attn = attn * causal.unsqueeze(0).unsqueeze(0)

    Y_intra = torch.einsum('bclm,bcmd->bcld', attn, X_chunks)

    # --- Inter-chunk computation (linear recurrence) ---
    # Compute chunk-level states
    # For each chunk, compute the final state: h_k = sum of decayed B * X
    # Simplified: accumulate states across chunks
    
    # Decay from end of one chunk to start of next
    chunk_decay = a_mean.sum(dim=-1)  # (batch, chunks) — total decay per chunk
    
    # State at end of each chunk — use list to avoid inplace ops
    state_list = [torch.zeros(batch, n, d, device=X.device, dtype=X.dtype)]
    
    for k in range(num_chunks):
        # Decay factor for this chunk
        decay_k = torch.exp(chunk_decay[:, k]).unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        # Input contribution: B^T @ X within this chunk
        # B_chunks[:, k]: (batch, L, n), X_chunks[:, k]: (batch, L, d)
        bx = torch.einsum('bln,bld->bnd', B_chunks[:, k], X_chunks[:, k])
        state_list.append(state_list[-1] * decay_k + bx)

    # Stack and take prev states (exclude last)
    states = torch.stack(state_list, dim=1)  # (batch, chunks+1, n, d)
    prev_states = states[:, :-1]  # (batch, chunks, n, d)
    Y_inter = torch.einsum('bcln,bcnd->bcld', C_chunks, prev_states)

    # Combine
    Y = Y_intra + Y_inter  # (batch, chunks, L, d)
    Y = Y.view(batch, T, d)

    # Remove padding
    if pad_len > 0:
        Y = Y[:, :seq_len]

    return Y


def ssd_recurrent(
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    X: torch.Tensor,
    A_log: torch.Tensor,
    initial_state: torch.Tensor = None,
) -> tuple:
    """
    Recurrent mode SSD for inference (one token at a time).
    
    Args:
        B_mat: (batch, 1, state_dim)
        C_mat: (batch, 1, state_dim)  
        X: (batch, 1, head_dim)
        A_log: (batch, 1, num_heads)
        initial_state: (batch, state_dim, head_dim) or None
        
    Returns:
        Y: (batch, 1, head_dim)
        new_state: (batch, state_dim, head_dim)
    """
    batch = X.shape[0]
    n = B_mat.shape[-1]
    d = X.shape[-1]

    if initial_state is None:
        state = torch.zeros(batch, n, d, device=X.device, dtype=X.dtype)
    else:
        state = initial_state

    # Decay
    decay = torch.exp(A_log.mean(dim=-1))  # (batch, 1)
    decay = decay.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1, 1) -> squeeze to (batch, 1, 1)
    decay = decay.squeeze(1)

    # Update state: h_t = decay * h_{t-1} + B_t * X_t
    bx = torch.einsum('bin,bid->bnd', B_mat, X)
    new_state = state * decay + bx

    # Output: Y_t = C_t @ h_t
    Y = torch.einsum('bin,bnd->bid', C_mat, new_state)

    return Y, new_state
