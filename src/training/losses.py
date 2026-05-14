"""
Loss functions: LM loss, MTP loss, MoE balance loss.
"""

import torch
import torch.nn.functional as F


def compute_total_loss(
    model_output: dict,
    balance_loss_weight: float = 0.0001,
    mtp_loss_weight: float = 0.3,
) -> tuple:
    """
    Compute the total training loss combining LM, MTP, and balance losses.
    
    Args:
        model_output: dict from model.forward() with 'loss', 'mtp_loss', 'aux_loss'
        balance_loss_weight: weight for MoE balance loss
        mtp_loss_weight: weight for MTP loss
        
    Returns:
        total_loss: combined scalar loss
        loss_dict: dict with individual loss components for logging
    """
    lm_loss = model_output["loss"]
    mtp_loss = model_output.get("mtp_loss", torch.tensor(0.0))
    aux_loss = model_output.get("aux_loss", torch.tensor(0.0))

    total = lm_loss + mtp_loss_weight * mtp_loss + balance_loss_weight * aux_loss

    return total, {
        "lm_loss": lm_loss.item(),
        "mtp_loss": mtp_loss.item(),
        "aux_loss": aux_loss.item(),
        "total_loss": total.item(),
    }
