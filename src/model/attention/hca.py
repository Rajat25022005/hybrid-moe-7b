"""
Heavily Compressed Attention (HCA).
Like CSA but with higher compression rate m' and no sparse selection —
dense attention on all heavily compressed KV entries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.normalization import RMSNorm
from src.model.attention.kv_compressor import KVCompressor
from src.model.attention.sliding_window import SlidingWindowKV
from src.model.attention.rope import PartialRoPE


class HeavilyCompressedAttention(nn.Module):
    """
    HCA: Heavily compresses KV cache (m'=64) and performs dense attention
    on all compressed entries. Combined with sliding window for local context.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_query_heads: int = 32,
        head_dim: int = 128,
        query_compress_dim: int = 512,
        num_output_groups: int = 4,
        group_output_dim: int = 512,
        compress_rate: int = 64,
        sliding_window_size: int = 128,
        rope_dim: int = 64,
        max_seq_len: int = 131072,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim
        self.query_compress_dim = query_compress_dim
        self.num_output_groups = num_output_groups
        self.group_output_dim = group_output_dim
        self.compress_rate = compress_rate

        # Query: low-rank decomposition h -> W_DQ -> c_Q -> W_UQ -> queries
        self.W_DQ = nn.Linear(hidden_dim, query_compress_dim, bias=False)
        self.W_UQ = nn.Linear(query_compress_dim, num_query_heads * head_dim, bias=False)

        # KV compressor (non-overlapped for HCA, higher compression rate)
        self.kv_compressor = KVCompressor(
            hidden_dim=hidden_dim,
            kv_dim=head_dim,
            compress_rate=compress_rate,
            overlapped=False,
        )

        # Sliding window KV
        self.sliding_window = SlidingWindowKV(
            hidden_dim=hidden_dim,
            kv_dim=head_dim,
            window_size=sliding_window_size,
        )

        # Grouped output projection
        heads_per_group = num_query_heads // num_output_groups
        self.group_proj = nn.ModuleList([
            nn.Linear(head_dim * heads_per_group, group_output_dim, bias=False)
            for _ in range(num_output_groups)
        ])
        self.out_proj = nn.Linear(group_output_dim * num_output_groups, hidden_dim, bias=False)

        # Normalization
        self.q_norm = RMSNorm(head_dim)
        self.kv_norm = RMSNorm(head_dim)

        # Attention sink
        self.sink_logits = nn.Parameter(torch.zeros(num_query_heads))

        # Partial RoPE
        self.rope = PartialRoPE(rope_dim=rope_dim, max_seq_len=max_seq_len)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        B, N, D = hidden_states.shape

        # 1. Query low-rank projection
        c_Q = self.W_DQ(hidden_states)
        queries = self.W_UQ(c_Q).view(B, N, self.num_query_heads, self.head_dim)
        queries = self.q_norm(queries)
        queries = self.rope(queries)

        # 2. Heavily compress KV entries
        compressed_kv = self.kv_compressor(hidden_states)  # (B, N//m', c)
        compressed_kv = self.kv_norm(compressed_kv)
        num_blocks = compressed_kv.shape[1]

        # 3. Sliding window KV
        local_kv = self.sliding_window(hidden_states)
        window_kv = self.sliding_window.get_window_kv(local_kv)  # (B, N, win, c)

        # 4. For HCA: dense attention on ALL compressed KV entries (no sparse selection)
        # Expand compressed_kv for each query position
        compressed_kv_expanded = compressed_kv.unsqueeze(1).expand(B, N, num_blocks, self.head_dim)

        # Causal mask for compressed blocks
        token_positions = torch.arange(N, device=hidden_states.device)
        block_positions = torch.arange(num_blocks, device=hidden_states.device)
        causal_mask = block_positions.unsqueeze(0) < (token_positions.unsqueeze(1) // self.compress_rate)
        # causal_mask: (N, num_blocks)

        # 5. Concatenate compressed + window KV
        all_kv = torch.cat([compressed_kv_expanded, window_kv], dim=2)  # (B, N, num_blocks+win, c)

        # 6. Core MQA attention
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.einsum('bnhc,bnkc->bnhk', queries, all_kv) / scale

        # Apply causal mask to compressed part only
        win_size = window_kv.shape[2]
        comp_mask = ~causal_mask  # (N, num_blocks)
        full_mask = torch.cat([
            comp_mask,
            torch.zeros(N, win_size, dtype=torch.bool, device=hidden_states.device),
        ], dim=-1)  # (N, num_blocks + win)
        attn_logits = attn_logits.masked_fill(
            full_mask.unsqueeze(0).unsqueeze(2), float('-inf')
        )

        # Attention sink
        sink = self.sink_logits.view(1, 1, -1, 1).expand(B, N, -1, 1)
        attn_logits_with_sink = torch.cat([attn_logits, sink], dim=-1)
        attn_weights = F.softmax(attn_logits_with_sink, dim=-1)[..., :-1]

        # Weighted sum
        attn_output = torch.einsum('bnhk,bnkc->bnhc', attn_weights, all_kv)

        # 7. Grouped output projection
        heads_per_group = self.num_query_heads // self.num_output_groups
        group_outputs = []
        for g in range(self.num_output_groups):
            start = g * heads_per_group
            end = start + heads_per_group
            group_in = attn_output[:, :, start:end, :].reshape(B, N, -1)
            group_outputs.append(self.group_proj[g](group_in))

        combined = torch.cat(group_outputs, dim=-1)
        output = self.out_proj(combined)

        return output
