"""
Transformer Block — Generic wrapper composing attention/Mamba2 + mHC + DeepSeekMoE.
Handles all 5 block types: full_attn, csa, hca, mamba2, ffn_only.
"""

import torch
import torch.nn as nn

# Use XLA-compatible gradient checkpointing when available
try:
    from torch_xla.utils.checkpoint import checkpoint as grad_checkpoint
except ImportError:
    from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.model.normalization import RMSNorm
from src.model.attention.full_attention import FullAttention
from src.model.attention.csa import CompressedSparseAttention
from src.model.attention.hca import HeavilyCompressedAttention
from src.model.mamba2.mamba2_block import Mamba2Block
from src.model.moe.deepseek_moe import DeepSeekMoE
from src.model.mhc.hyper_connections import ManifoldHyperConnection
from src.model.config import ModelConfig


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    1. Pre-block mHC mixing -> extract layer input
    2. RMSNorm -> Attention/Mamba2 (or skip for FFN-only)
    3. mHC residual update
    4. RMSNorm -> DeepSeekMoE FFN
    5. Post-block mHC update
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        self.use_gradient_checkpointing = use_gradient_checkpointing
        d = config.hidden_dim

        # mHC for this block
        self.mhc = ManifoldHyperConnection(
            hidden_dim=d,
            expansion=config.mhc.expansion,
            sinkhorn_iters=config.mhc.sinkhorn_iters,
        )

        # Pre-attention/SSM norm
        self.attn_norm = RMSNorm(d)

        # Sequence mixing layer (depends on block type)
        if self.block_type == "full_attn":
            self.seq_mixer = FullAttention(
                hidden_dim=d,
                num_heads=config.attention.num_query_heads,
                head_dim=config.attention.head_dim,
                rope_dim=config.attention.rope_dim,
                max_seq_len=config.max_seq_len * 2,
            )
        elif self.block_type == "csa":
            self.seq_mixer = CompressedSparseAttention(
                hidden_dim=d,
                num_query_heads=config.attention.num_query_heads,
                head_dim=config.attention.csa_hca_head_dim,
                query_compress_dim=config.attention.query_compress_dim,
                num_output_groups=config.attention.num_output_groups,
                group_output_dim=config.attention.group_output_dim,
                compress_rate=config.csa.compress_rate,
                indexer_heads=config.csa.indexer_heads,
                indexer_head_dim=config.csa.indexer_head_dim,
                topk=config.csa.topk,
                sliding_window_size=config.attention.sliding_window_size,
                rope_dim=config.attention.rope_dim,
                max_seq_len=config.max_seq_len * 2,
            )
        elif self.block_type == "hca":
            self.seq_mixer = HeavilyCompressedAttention(
                hidden_dim=d,
                num_query_heads=config.attention.num_query_heads,
                head_dim=config.attention.csa_hca_head_dim,
                query_compress_dim=config.attention.query_compress_dim,
                num_output_groups=config.attention.num_output_groups,
                group_output_dim=config.attention.group_output_dim,
                compress_rate=config.hca.compress_rate,
                sliding_window_size=config.attention.sliding_window_size,
                rope_dim=config.attention.rope_dim,
                max_seq_len=config.max_seq_len * 2,
            )
        elif self.block_type == "mamba2":
            self.seq_mixer = Mamba2Block(
                hidden_dim=d,
                state_dim=config.mamba2.state_dim,
                conv_dim=config.mamba2.conv_dim,
                expand=config.mamba2.expand,
                head_dim=config.mamba2.head_dim,
                num_heads=config.mamba2.num_heads,
            )
        elif self.block_type == "ffn_only":
            self.seq_mixer = None  # No sequence mixing
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")

        # Pre-FFN norm
        self.ffn_norm = RMSNorm(d)

        # DeepSeekMoE FFN
        use_hash = layer_idx < config.moe.num_shared_experts  # Actually use hash_routing_layers from training config
        self.moe = DeepSeekMoE(
            hidden_dim=d,
            num_shared_experts=config.moe.num_shared_experts,
            num_routed_experts=config.moe.num_routed_experts,
            num_active_experts=config.moe.num_active_experts,
            expert_intermediate_dim=config.moe.expert_intermediate_dim,
            shared_expert_intermediate_dim=config.moe.shared_expert_intermediate_dim,
            use_hash_routing=False,  # Set externally via set_hash_routing
        )

    def set_hash_routing(self, use_hash: bool):
        """Enable/disable hash routing for this layer's MoE."""
        if use_hash and not isinstance(self.moe.router, type(self.moe.router)):
            from src.model.moe.hash_router import HashRouter
            self.moe.router = HashRouter(
                self.moe.num_routed, self.moe.num_active
            )

    def _forward_impl(
        self,
        X: torch.Tensor,
        use_sparse: bool = True,
    ) -> tuple:
        """
        Core forward pass.
        
        Args:
            X: (batch, seq_len, n_hc, hidden_dim) — expanded residual state
            use_sparse: whether to use sparse attention in CSA
        Returns:
            X_new: updated residual state
            aux_loss: MoE balance loss
        """
        # 1. mHC pre-block: extract layer input
        layer_input, mappings_attn = self.mhc.pre_block(X)

        # 2. Sequence mixing (if not FFN-only)
        if self.seq_mixer is not None:
            normed = self.attn_norm(layer_input)
            if self.block_type == "csa":
                mixer_out = self.seq_mixer(normed, use_sparse=use_sparse)
            else:
                mixer_out = self.seq_mixer(normed)
            # Residual via mHC
            X = self.mhc.post_block(X, mixer_out, mappings_attn)
        # For FFN-only: skip sequence mixing, X stays the same

        # 3. mHC pre-block for FFN
        ffn_input, mappings_ffn = self.mhc.pre_block(X)

        # 4. DeepSeekMoE FFN
        normed_ffn = self.ffn_norm(ffn_input)
        ffn_out, aux_loss = self.moe(normed_ffn)

        # 5. mHC post-block
        X = self.mhc.post_block(X, ffn_out, mappings_ffn)

        return X, aux_loss

    def forward(
        self,
        X: torch.Tensor,
        use_sparse: bool = True,
    ) -> tuple:
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            def run_fn(x):
                return self._forward_impl(x, use_sparse)
            X, aux_loss = checkpoint(run_fn, X, use_reentrant=False)
        else:
            X, aux_loss = self._forward_impl(X, use_sparse)
        return X, aux_loss

