"""
Compressed Sparse Attention (CSA).
Full pipeline: KV Compression -> Lightning Indexer -> Sparse Selection ->
Core MQA -> Grouped Output Projection.
With sliding window, attention sink, partial RoPE, Q/KV normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.normalization import RMSNorm
from src.model.attention.kv_compressor import KVCompressor
from src.model.attention.lightning_indexer import LightningIndexer
from src.model.attention.sliding_window import SlidingWindowKV
from src.model.attention.rope import PartialRoPE


class CompressedSparseAttention(nn.Module):
    """
    CSA: Compresses KV cache by factor m, then selects top-k via
    lightning indexer for sparse attention. Combined with sliding window
    for local context.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_query_heads: int = 32,
        head_dim: int = 128,
        query_compress_dim: int = 512,
        num_output_groups: int = 4,
        group_output_dim: int = 512,
        compress_rate: int = 4,
        indexer_heads: int = 32,
        indexer_head_dim: int = 64,
        topk: int = 128,
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
        self.topk = topk

        # Query down-projection (shared between indexer and core attention)
        self.W_DQ = nn.Linear(hidden_dim, query_compress_dim, bias=False)

        # Query up-projection for core attention
        self.W_UQ = nn.Linear(query_compress_dim, num_query_heads * head_dim, bias=False)

        # KV compressor (overlapped for CSA)
        self.kv_compressor = KVCompressor(
            hidden_dim=hidden_dim,
            kv_dim=head_dim,
            compress_rate=compress_rate,
            overlapped=True,
        )

        # Lightning indexer for sparse selection
        self.indexer = LightningIndexer(
            hidden_dim=hidden_dim,
            query_compress_dim=query_compress_dim,
            indexer_heads=indexer_heads,
            indexer_head_dim=indexer_head_dim,
            kv_dim=head_dim,
            compress_rate=compress_rate,
            topk=topk,
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

        # Attention sink: learnable logits per head
        self.sink_logits = nn.Parameter(torch.zeros(num_query_heads))

        # Partial RoPE
        self.rope = PartialRoPE(rope_dim=rope_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_sparse: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            use_sparse: if False, attend to all compressed KV (dense mode for warmup)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        B, N, D = hidden_states.shape

        # 1. Shared compressed query
        c_Q = self.W_DQ(hidden_states)  # (B, N, d_c)

        # 2. Core attention queries
        queries = self.W_UQ(c_Q)  # (B, N, n_h * c)
        queries = queries.view(B, N, self.num_query_heads, self.head_dim)
        queries = self.q_norm(queries)

        # 3. Compress KV entries
        compressed_kv = self.kv_compressor(hidden_states)  # (B, N//m, c)
        compressed_kv = self.kv_norm(compressed_kv)

        # 4. Select sparse KV via indexer (or use all in dense mode)
        if use_sparse:
            K_IComp = self.indexer.compress_indexer_keys(hidden_states)
            index_scores = self.indexer.compute_index_scores(c_Q, hidden_states, K_IComp)
            selected_kv, _ = self.indexer.select_topk(index_scores, compressed_kv)
            # selected_kv: (B, N, topk, c)
        else:
            # Dense mode: use all compressed KV for each query
            num_blocks = compressed_kv.shape[1]
            selected_kv = compressed_kv.unsqueeze(1).expand(B, N, num_blocks, self.head_dim)

        # 5. Sliding window KV
        local_kv = self.sliding_window(hidden_states)  # (B, N, c)
        window_kv = self.sliding_window.get_window_kv(local_kv)  # (B, N, win, c)

        # 6. Concatenate selected compressed + window KV
        all_kv = torch.cat([selected_kv, window_kv], dim=2)  # (B, N, topk+win, c)

        # 7. Apply partial RoPE to queries
        queries = self.rope(queries)

        # 8. Core MQA attention
        # queries: (B, N, n_h, c)
        # all_kv: (B, N, K, c) — shared across all heads (MQA)
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.einsum('bnhc,bnkc->bnhk', queries, all_kv) / scale

        # Attention sink: add learnable denominator term
        sink = self.sink_logits.view(1, 1, -1, 1).expand(B, N, -1, 1)
        attn_logits_with_sink = torch.cat([attn_logits, sink], dim=-1)
        attn_weights = F.softmax(attn_logits_with_sink, dim=-1)
        # Remove the sink weight column (it absorbs probability mass)
        attn_weights = attn_weights[..., :-1]

        # Weighted sum
        attn_output = torch.einsum('bnhk,bnkc->bnhc', attn_weights, all_kv)
        # attn_output: (B, N, n_h, c)

        # 9. Grouped output projection
        heads_per_group = self.num_query_heads // self.num_output_groups
        group_outputs = []
        for g in range(self.num_output_groups):
            start = g * heads_per_group
            end = start + heads_per_group
            group_in = attn_output[:, :, start:end, :].reshape(B, N, -1)
            group_outputs.append(self.group_proj[g](group_in))

        combined = torch.cat(group_outputs, dim=-1)  # (B, N, g * d_g)
        output = self.out_proj(combined)  # (B, N, d)

        return output
