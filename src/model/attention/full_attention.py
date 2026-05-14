"""
Full Multi-Head Attention — anchor layers.
Standard causal multi-head attention with GQA support, RMSNorm on Q/K,
and partial RoPE on last 64 dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.normalization import RMSNorm
from src.model.attention.rope import PartialRoPE


class FullAttention(nn.Module):
    """
    Standard multi-head attention used as anchor layers (layers 0, 15).
    Provides full global information aggregation without compression.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 32,
        head_dim: int = 64,
        rope_dim: int = 64,
        max_seq_len: int = 131072,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        self.W_Q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.W_O = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Pre-attention normalization on Q and K
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # Partial RoPE
        self.rope = PartialRoPE(rope_dim=rope_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: optional (batch, 1, seq_len, seq_len) or None for causal
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        B, N, D = hidden_states.shape

        # Project Q, K, V
        Q = self.W_Q(hidden_states).view(B, N, self.num_heads, self.head_dim)
        K = self.W_K(hidden_states).view(B, N, self.num_heads, self.head_dim)
        V = self.W_V(hidden_states).view(B, N, self.num_heads, self.head_dim)

        # Normalize Q and K per head
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Apply partial RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        # Transpose for attention: (B, num_heads, N, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((N, N), float('-inf'), device=hidden_states.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attend
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.W_O(attn_output)

        return output
