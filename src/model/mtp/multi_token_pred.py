"""
Multi-Token Prediction (MTP) module.
Predicts the next D+1 tokens (D=1: predicts next 2 tokens).
Uses shared embedding weights and residual mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.normalization import RMSNorm


class MultiTokenPrediction(nn.Module):
    """
    MTP module following DeepSeek V3/V4 design.
    Depth=1: one additional prediction head for the next-next token.
    Uses residual mixing between current and lookahead representations.
    """

    def __init__(self, hidden_dim: int, vocab_size: int, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim

        # For each prediction depth, a mixing + projection module
        self.mtp_norms = nn.ModuleList([
            RMSNorm(hidden_dim) for _ in range(depth)
        ])
        self.mtp_proj = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False) for _ in range(depth)
        ])
        self.mtp_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size, bias=False) for _ in range(depth)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedding_weight: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) — from main model
            embedding_weight: (vocab_size, hidden_dim) — shared embedding
            target_ids: (batch, seq_len) — target token IDs for computing loss
            
        Returns:
            mtp_logits: list of (batch, seq_len, vocab_size) for each depth
            mtp_loss: scalar loss (0 if no targets)
        """
        B, N, D = hidden_states.shape
        mtp_logits_list = []
        mtp_loss = torch.tensor(0.0, device=hidden_states.device)

        current_repr = hidden_states

        for d in range(self.depth):
            # Normalize current representation
            normed = self.mtp_norms[d](current_repr)

            if target_ids is not None and d + 1 < N:
                # Get the target embeddings for the next position (shifted by d+1)
                # target_ids[:, d+1:] gives us the tokens at positions d+1, d+2, ...
                shifted_ids = target_ids[:, d+1:]
                target_embeds = F.embedding(shifted_ids, embedding_weight)

                # Truncate current repr to match
                truncated = normed[:, :-(d+1)]

                # Residual mixing: concat current repr with target embedding
                mixed = torch.cat([truncated, target_embeds], dim=-1)
                projected = self.mtp_proj[d](mixed)

                # Predict token at position t+d+2
                logits = self.mtp_heads[d](projected)
                mtp_logits_list.append(logits)

                # Compute loss against target at position t+d+2
                if d + 2 <= N:
                    loss_targets = target_ids[:, d+2:d+2+logits.shape[1]]
                    if loss_targets.shape[1] > 0 and logits.shape[1] > 0:
                        min_len = min(logits.shape[1], loss_targets.shape[1])
                        loss = F.cross_entropy(
                            logits[:, :min_len].reshape(-1, logits.shape[-1]),
                            loss_targets[:, :min_len].reshape(-1),
                        )
                        mtp_loss = mtp_loss + loss
            else:
                # Inference: just produce logits
                logits = self.mtp_heads[d](normed)
                mtp_logits_list.append(logits)

        return mtp_logits_list, mtp_loss
