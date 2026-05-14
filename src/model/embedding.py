"""
Token embedding layer with optional weight tying to the output projection.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding with scaling, following standard transformer practice."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer token IDs
        Returns:
            (batch, seq_len, hidden_dim) embeddings
        """
        return self.embed(token_ids) * math.sqrt(self.hidden_dim)

    @property
    def weight(self) -> torch.Tensor:
        """Return embedding weight matrix for weight tying."""
        return self.embed.weight
