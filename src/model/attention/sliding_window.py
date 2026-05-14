"""
Sliding Window Attention branch.
Produces uncompressed KV entries for the n_win most recent tokens,
concatenated with compressed KV entries for local fine-grained dependencies.
"""

import torch
import torch.nn as nn


class SlidingWindowKV(nn.Module):
    """
    Generates sliding window KV entries for local context.
    Each query attends to the most recent n_win uncompressed tokens
    alongside the compressed KV entries.
    """

    def __init__(self, hidden_dim: int, kv_dim: int, window_size: int = 128):
        super().__init__()
        self.window_size = window_size
        self.kv_dim = kv_dim
        self.W_kv_local = nn.Linear(hidden_dim, kv_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Produce local KV entries for sliding window.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            
        Returns:
            local_kv: (batch, seq_len, kv_dim) — full sequence,
                      caller will slice the relevant window per query
        """
        return self.W_kv_local(hidden_states)

    def get_window_kv(
        self,
        local_kv: torch.Tensor,
        query_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extract sliding window KV entries for each query position.
        
        For training (full sequence), returns a windowed view.
        
        Args:
            local_kv: (batch, seq_len, kv_dim)
            query_positions: optional, defaults to all positions
            
        Returns:
            window_kv: (batch, seq_len, window_size, kv_dim)
        """
        B, N, C = local_kv.shape
        w = self.window_size

        # Pad the beginning so position 0 has a valid window
        padded = torch.cat([
            torch.zeros(B, w, C, device=local_kv.device, dtype=local_kv.dtype),
            local_kv,
        ], dim=1)  # (B, N + w, C)

        # Unfold to get windows: for each position i, get [i, i+w)
        # After padding, position i in original = position i+w in padded
        # Window for position i = padded[i+1 : i+w+1] (the w tokens before and including i)
        # Use stride tricks for efficiency
        window_kv = padded.unfold(1, w, 1)[:, :N]  # (B, N, C, w)
        window_kv = window_kv.permute(0, 1, 3, 2).contiguous()   # (B, N, w, C)

        return window_kv
