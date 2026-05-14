"""
Expert FFN — Individual expert with SiLU-gated intermediate.
Used as both shared and routed experts in DeepSeekMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """
    Single expert FFN with gated activation.
    h -> W_gate * SiLU(W_up * h) -> W_down -> output
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.W_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.W_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.W_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., hidden_dim)
        Returns:
            (..., hidden_dim)
        """
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))
