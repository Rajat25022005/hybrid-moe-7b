"""
HybridMoEModel — Top-level 7B MoE model.
32 transformer blocks with hybrid layer schedule + MTP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.embedding import TokenEmbedding
from src.model.normalization import RMSNorm
from src.model.transformer_block import TransformerBlock
from src.model.mhc.hyper_connections import ManifoldHyperConnection
from src.model.mtp.multi_token_pred import MultiTokenPrediction


class HybridMoEModel(nn.Module):
    """
    7B Hybrid MoE LLM with:
    - 12 Mamba2 + 12 FFN-only + 2 Full Attention + 3 CSA + 3 HCA layers
    - DeepSeekMoE FFN on every layer
    - Manifold-Constrained Hyper-Connections (mHC)
    - Multi-Token Prediction (MTP)
    - Weight-tied embedding and output projection
    """

    def __init__(self, config: ModelConfig, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_dim)

        # Initial mHC state builder
        self.mhc_init = ManifoldHyperConnection(
            hidden_dim=config.hidden_dim,
            expansion=config.mhc.expansion,
            sinkhorn_iters=config.mhc.sinkhorn_iters,
        )

        # 32 transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                config=config,
                layer_idx=i,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            for i in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.hidden_dim)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying

        # Multi-Token Prediction
        self.mtp = MultiTokenPrediction(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            depth=config.mtp.depth,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
        use_sparse: bool = True,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            target_ids: (batch, seq_len) target token IDs for loss computation
            use_sparse: whether CSA uses sparse attention (False during dense warmup)
            
        Returns:
            dict with:
                logits: (batch, seq_len, vocab_size)
                loss: scalar (if target_ids provided)
                aux_loss: scalar MoE balance loss
                mtp_loss: scalar MTP loss
        """
        B, N = input_ids.shape

        # 1. Embed tokens
        hidden_states = self.embedding(input_ids)  # (B, N, D)

        # 2. Initialize mHC expanded residual state
        X = self.mhc_init.init_state(hidden_states)  # (B, N, n_hc, D)

        # 3. Pass through all transformer blocks
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        for layer in self.layers:
            X, aux_loss = layer(X, use_sparse=use_sparse)
            total_aux_loss = total_aux_loss + aux_loss

        # 4. Extract final hidden states from mHC state
        hidden_states = self.mhc_init.extract_output(X)  # (B, N, D)

        # 5. Final norm + LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)  # (B, N, vocab_size)

        # 6. Compute losses
        result = {"logits": logits, "aux_loss": total_aux_loss}

        if target_ids is not None:
            # Main LM loss (next-token prediction)
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
            )
            result["loss"] = lm_loss

            # MTP loss
            _, mtp_loss = self.mtp(
                hidden_states, self.embedding.weight, target_ids
            )
            result["mtp_loss"] = mtp_loss
        else:
            result["loss"] = torch.tensor(0.0, device=input_ids.device)
            result["mtp_loss"] = torch.tensor(0.0, device=input_ids.device)

        return result

    def count_parameters(self) -> dict:
        """Count total and activated parameters."""
        total = sum(p.numel() for p in self.parameters())
        
        # Activated = everything except inactive routed experts
        # Each layer has num_routed experts, only num_active are used
        moe_cfg = self.config.moe
        expert_params = 3 * self.config.hidden_dim * moe_cfg.expert_intermediate_dim
        inactive_per_layer = (moe_cfg.num_routed_experts - moe_cfg.num_active_experts) * expert_params
        total_inactive = inactive_per_layer * self.config.num_layers
        activated = total - total_inactive

        return {
            "total": total,
            "activated": activated,
            "total_billions": total / 1e9,
            "activated_billions": activated / 1e9,
        }

    def set_hash_routing_layers(self, num_hash_layers: int):
        """Enable hash routing for the first N layers."""
        from src.model.moe.hash_router import HashRouter
        for i, layer in enumerate(self.layers):
            if i < num_hash_layers:
                layer.moe.router = HashRouter(
                    self.config.moe.num_routed_experts,
                    self.config.moe.num_active_experts,
                )
                layer.moe.use_hash_routing = True
