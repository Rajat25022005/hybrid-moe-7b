"""
Hash Router — Deterministic routing for initial MoE layers.
Uses predefined hash function on token IDs for stable early training.
"""

import torch
import torch.nn as nn


class HashRouter(nn.Module):
    """
    Hash-based routing for the first few MoE layers (default: first 3).
    Determines expert assignment deterministically based on token ID,
    providing stable gradients during early training.
    No learnable parameters.
    """

    def __init__(self, num_experts: int, num_active: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor = None,
    ) -> tuple:
        """
        Route tokens based on hash of position or token ID.
        
        Args:
            hidden_states: (num_tokens, hidden_dim)
            token_ids: (num_tokens,) optional token IDs for hashing
            
        Returns:
            expert_weights: (num_tokens, num_active) — uniform weights
            expert_indices: (num_tokens, num_active) — deterministic expert IDs
            aux_loss: 0 (no loss for hash routing)
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device

        if token_ids is not None:
            # Hash based on token ID
            hash_vals = token_ids.long()
        else:
            # Hash based on position index
            hash_vals = torch.arange(num_tokens, device=device)

        # Generate num_active different expert indices per token
        # Using different prime multipliers for each active slot
        primes = [7, 13, 31, 61, 127, 251, 509, 1021][:self.num_active]
        indices = []
        for p in primes:
            idx = (hash_vals * p + p) % self.num_experts
            indices.append(idx)

        expert_indices = torch.stack(indices, dim=-1)  # (num_tokens, num_active)

        # Ensure no duplicate experts per token
        for i in range(1, self.num_active):
            for j in range(i):
                collision = expert_indices[:, i] == expert_indices[:, j]
                expert_indices[:, i] = torch.where(
                    collision,
                    (expert_indices[:, i] + 1) % self.num_experts,
                    expert_indices[:, i],
                )

        # Uniform weights
        expert_weights = torch.ones(
            num_tokens, self.num_active,
            device=device, dtype=hidden_states.dtype,
        ) / self.num_active

        return expert_weights, expert_indices, torch.tensor(0.0, device=device)
