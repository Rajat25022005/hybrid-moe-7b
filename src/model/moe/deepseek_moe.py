"""
DeepSeekMoE Layer — fine-grained MoE with shared + routed experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.moe.expert import ExpertFFN
from src.model.moe.router import MoERouter
from src.model.moe.hash_router import HashRouter


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE: shared expert(s) always active + routed experts via top-k.
    Supports both learned routing and hash routing (for early layers).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_shared_experts: int = 1,
        num_routed_experts: int = 32,
        num_active_experts: int = 4,
        expert_intermediate_dim: int = 1024,
        shared_expert_intermediate_dim: int = 2048,
        use_hash_routing: bool = False,
        bias_update_speed: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_shared = num_shared_experts
        self.num_routed = num_routed_experts
        self.num_active = num_active_experts
        self.use_hash_routing = use_hash_routing

        # Shared experts (always active for every token)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(hidden_dim, shared_expert_intermediate_dim)
            for _ in range(num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            ExpertFFN(hidden_dim, expert_intermediate_dim)
            for _ in range(num_routed_experts)
        ])

        # Router
        if use_hash_routing:
            self.router = HashRouter(num_routed_experts, num_active_experts)
        else:
            self.router = MoERouter(
                hidden_dim, num_routed_experts, num_active_experts,
                bias_update_speed=bias_update_speed,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            token_ids: optional (batch, seq_len) for hash routing
        Returns:
            output: (batch, seq_len, hidden_dim)
            aux_loss: scalar balance loss
        """
        B, N, D = hidden_states.shape

        # 1. Shared experts (always active)
        shared_output = torch.zeros_like(hidden_states)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(hidden_states)

        # 2. Route tokens to experts
        flat_hidden = hidden_states.view(B * N, D)
        flat_ids = token_ids.view(B * N) if token_ids is not None else None

        if self.use_hash_routing:
            weights, indices, aux_loss = self.router(flat_hidden, flat_ids)
        else:
            weights, indices, aux_loss = self.router(flat_hidden)

        # 3. Compute routed expert outputs (efficient masked routing for CUDA)
        routed_output = torch.zeros(B * N, D, device=hidden_states.device, dtype=hidden_states.dtype)

        for k in range(self.num_active):
            expert_idx = indices[:, k]   # (B*N,)
            expert_wt = weights[:, k]    # (B*N,)

            for e in range(self.num_routed):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = flat_hidden[mask]
                    expert_out = self.routed_experts[e](expert_input)
                    routed_output[mask] = routed_output[mask] + expert_wt[mask].unsqueeze(-1) * expert_out

        routed_output = routed_output.view(B, N, D)

        # 4. Combine shared + routed
        output = shared_output + routed_output

        return output, aux_loss
