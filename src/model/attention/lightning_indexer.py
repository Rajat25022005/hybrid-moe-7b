"""
Lightning Indexer for CSA sparse selection.
Computes index scores and selects top-k compressed KV entries per query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """
    Lightning Indexer for Compressed Sparse Attention.
    
    Produces indexer queries from the shared compressed latent vector,
    computes index scores against compressed indexer keys, and selects
    top-k compressed KV entries for each query token.
    """

    def __init__(
        self,
        hidden_dim: int,
        query_compress_dim: int,
        indexer_heads: int,
        indexer_head_dim: int,
        kv_dim: int,
        compress_rate: int,
        topk: int,
    ):
        super().__init__()
        self.indexer_heads = indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.topk = topk
        self.kv_dim = kv_dim
        self.compress_rate = compress_rate

        # Indexer query up-projection (from shared compressed query)
        # c_Q -> q_I = c_Q @ W_IUQ
        self.W_IUQ = nn.Linear(
            query_compress_dim,
            indexer_heads * indexer_head_dim,
            bias=False,
        )

        # Indexer head weights: h_t @ W_w -> w_I
        self.W_w = nn.Linear(hidden_dim, indexer_heads, bias=False)

        # Indexer KV compressor (same compression as main KV)
        # Produces compressed indexer keys
        self.W_ik_a = nn.Linear(hidden_dim, indexer_head_dim, bias=False)
        self.W_ik_b = nn.Linear(hidden_dim, indexer_head_dim, bias=False)
        self.W_iz_a = nn.Linear(hidden_dim, indexer_head_dim, bias=False)
        self.W_iz_b = nn.Linear(hidden_dim, indexer_head_dim, bias=False)
        self.bias_ik_a = nn.Parameter(torch.zeros(compress_rate, indexer_head_dim))
        self.bias_ik_b = nn.Parameter(torch.zeros(compress_rate, indexer_head_dim))

    def compress_indexer_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compress hidden states into indexer keys using same overlapped
        compression as main KV entries.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            K_IComp: (batch, seq_len // m, indexer_head_dim)
        """
        B, N, D = hidden_states.shape
        m = self.compress_rate

        pad_len = (m - N % m) % m
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            N = hidden_states.shape[1]

        num_blocks = N // m

        # Two branches
        K_a = self.W_ik_a(hidden_states).view(B, num_blocks, m, -1)
        K_b = self.W_ik_b(hidden_states).view(B, num_blocks, m, -1)
        Z_a = self.W_iz_a(hidden_states).view(B, num_blocks, m, -1)
        Z_b = self.W_iz_b(hidden_states).view(B, num_blocks, m, -1)

        Z_a = Z_a + self.bias_ik_a.unsqueeze(0).unsqueeze(0)
        Z_b = Z_b + self.bias_ik_b.unsqueeze(0).unsqueeze(0)

        c_I = self.indexer_head_dim
        K_b_shifted = torch.cat([
            torch.zeros(B, 1, m, c_I, device=hidden_states.device, dtype=K_b.dtype),
            K_b[:, :-1]
        ], dim=1)
        Z_b_shifted = torch.cat([
            torch.full((B, 1, m, c_I), float('-inf'), device=hidden_states.device, dtype=Z_b.dtype),
            Z_b[:, :-1]
        ], dim=1)

        Z_cat = torch.cat([Z_a, Z_b_shifted], dim=2)
        K_cat = torch.cat([K_a, K_b_shifted], dim=2)
        S = F.softmax(Z_cat, dim=2)
        K_IComp = (S * K_cat).sum(dim=2)  # (B, num_blocks, c_I)

        return K_IComp

    def compute_index_scores(
        self,
        compressed_query: torch.Tensor,
        hidden_states: torch.Tensor,
        K_IComp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute index scores for each query-block pair.
        
        Args:
            compressed_query: (batch, seq_len, query_compress_dim) — shared c_Q
            hidden_states: (batch, seq_len, hidden_dim)
            K_IComp: (batch, num_blocks, indexer_head_dim)
            
        Returns:
            index_scores: (batch, seq_len, num_blocks)
        """
        B, N, _ = compressed_query.shape

        # Indexer queries: c_Q @ W_IUQ -> (B, N, n_h^I * c_I)
        q_I = self.W_IUQ(compressed_query)
        q_I = q_I.view(B, N, self.indexer_heads, self.indexer_head_dim)

        # Indexer head weights: h_t @ W_w -> (B, N, n_h^I)
        w_I = self.W_w(hidden_states)

        # Score: sum over heads of w_h * ReLU(q_I_h · K_IComp_s)
        # q_I: (B, N, n_h^I, c_I)
        # K_IComp: (B, num_blocks, c_I)
        # Dot product: (B, N, n_h^I, num_blocks)
        dots = torch.einsum('bnhc,bmc->bnhm', q_I, K_IComp)
        dots = F.relu(dots)

        # Weighted sum across heads
        # w_I: (B, N, n_h^I) -> (B, N, n_h^I, 1)
        scores = (w_I.unsqueeze(-1) * dots).sum(dim=2)  # (B, N, num_blocks)

        return scores

    def select_topk(
        self,
        index_scores: torch.Tensor,
        compressed_kv: torch.Tensor,
        query_block_ids: torch.Tensor = None,
    ) -> tuple:
        """
        Select top-k compressed KV entries per query based on index scores.
        Applies causal masking: query at position t can only attend to blocks s < floor(t/m).
        
        Args:
            index_scores: (batch, seq_len, num_blocks)
            compressed_kv: (batch, num_blocks, kv_dim)
            
        Returns:
            selected_kv: (batch, seq_len, topk, kv_dim)
            selected_indices: (batch, seq_len, topk)
        """
        B, N, num_blocks = index_scores.shape
        m = self.compress_rate

        # Causal mask: for query at position t, only blocks s < floor(t/m)
        token_positions = torch.arange(N, device=index_scores.device)
        block_positions = torch.arange(num_blocks, device=index_scores.device)
        causal_mask = block_positions.unsqueeze(0) < (token_positions.unsqueeze(1) // m)
        # causal_mask: (N, num_blocks)

        # Apply causal mask
        masked_scores = index_scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))

        # Top-k selection
        k = min(self.topk, num_blocks)
        topk_scores, topk_indices = masked_scores.topk(k, dim=-1)  # (B, N, k)

        # Gather selected KV entries
        # compressed_kv: (B, num_blocks, kv_dim)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, compressed_kv.shape[-1])
        selected_kv = compressed_kv.unsqueeze(1).expand(-1, N, -1, -1)
        selected_kv = torch.gather(selected_kv, 2, topk_indices_expanded)

        return selected_kv, topk_indices
