"""
Router for DeepSeekMoE — Sqrt(Softplus) affinity scoring with
auxiliary-loss-free load balancing via dynamic bias adjustment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MoERouter(nn.Module):
    """
    DeepSeek V4 style router with:
    - Sqrt(Softplus(·)) affinity scoring (changed from Sigmoid in V3)
    - Top-k expert selection
    - Auxiliary-loss-free dynamic bias for load balancing
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_active: int,
        bias_update_speed: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_active = num_active
        self.bias_update_speed = bias_update_speed

        # Router linear
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Dynamic bias for load balancing (not a learned parameter — updated externally)
        self.register_buffer('expert_bias', torch.zeros(num_experts))

        # Track load for bias updates
        self.register_buffer('expert_load', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """
        Route tokens to top-k experts.
        
        Args:
            hidden_states: (batch * seq_len, hidden_dim) — flattened tokens
            
        Returns:
            expert_weights: (batch * seq_len, num_active) — normalized weights
            expert_indices: (batch * seq_len, num_active) — selected expert IDs
            aux_loss: scalar balance loss
        """
        # Compute affinity scores: Sqrt(Softplus(logits))
        logits = self.gate(hidden_states)  # (tokens, num_experts)
        affinity = torch.sqrt(F.softplus(logits))

        # Add load-balancing bias
        biased_affinity = affinity + self.expert_bias.unsqueeze(0)

        # Top-k selection
        topk_weights, topk_indices = biased_affinity.topk(self.num_active, dim=-1)

        # Normalize weights (using original affinity, not biased)
        original_topk_weights = affinity.gather(-1, topk_indices)
        weights = original_topk_weights / (original_topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Update load statistics for bias adjustment (during training)
        if self.training:
            self._update_load(topk_indices, hidden_states.shape[0])

        # Compute auxiliary balance loss
        aux_loss = self._compute_balance_loss(logits)

        return weights, topk_indices, aux_loss

    def _update_load(self, indices: torch.Tensor, num_tokens: int):
        """Track expert load for dynamic bias updates."""
        load = torch.zeros(self.num_experts, device=indices.device)
        load.scatter_add_(
            0,
            indices.view(-1),
            torch.ones(indices.numel(), device=indices.device),
        )
        self.expert_load = load
        self.total_tokens.fill_(num_tokens)

    def update_bias(self):
        """
        Auxiliary-loss-free bias update.
        Overloaded experts get decreased bias, underloaded get increased.
        Called after each training step.
        """
        if self.total_tokens.item() == 0:
            return

        avg_load = self.total_tokens.item() * self.num_active / self.num_experts
        load_diff = self.expert_load - avg_load

        # Decrease bias for overloaded, increase for underloaded
        self.expert_bias -= self.bias_update_speed * torch.sign(load_diff)

    def _compute_balance_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sequence-level balance loss to prevent extreme imbalance.
        Lighter than traditional auxiliary loss.
        """
        # Fraction of tokens routed to each expert (soft)
        probs = F.softmax(logits, dim=-1)  # (tokens, experts)
        # Average probability per expert
        avg_probs = probs.mean(dim=0)  # (experts,)
        # Variance-based balance loss
        balance_loss = (avg_probs * self.num_experts).var()
        return balance_loss
