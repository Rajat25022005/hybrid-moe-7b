"""
Checkpoint utilities — save, load, and manage model checkpoints.
Keeps only the last N checkpoints to manage disk space.
"""

import os
import glob
import json
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    muon_optimizer,
    adamw_optimizer,
    scheduler,
    step: int,
    epoch: int,
    loss: float,
    config_dict: dict,
    checkpoint_dir: str,
    keep_last_n: int = 5,
    save_optimizer: bool = True,
):
    """
    Save a training checkpoint.
    
    Args:
        model: the model
        muon_optimizer: Muon optimizer
        adamw_optimizer: AdamW optimizer
        scheduler: LR scheduler
        step: current training step
        epoch: current epoch
        loss: current loss value
        config_dict: model + training config as dict
        checkpoint_dir: directory to save checkpoints
        keep_last_n: number of recent checkpoints to keep
        save_optimizer: whether to save optimizer states
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-step-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))

    # Save optimizer states
    if save_optimizer:
        torch.save(muon_optimizer.state_dict(), os.path.join(checkpoint_path, "muon_optimizer.pt"))
        torch.save(adamw_optimizer.state_dict(), os.path.join(checkpoint_path, "adamw_optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

    # Save metadata
    metadata = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "config": config_dict,
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")

    # Cleanup old checkpoints
    _cleanup_checkpoints(checkpoint_dir, keep_last_n)


def load_checkpoint(
    checkpoint_path: str,
    model,
    muon_optimizer=None,
    adamw_optimizer=None,
    scheduler=None,
    load_optimizer: bool = True,
) -> dict:
    """
    Load a checkpoint.
    
    Args:
        checkpoint_path: path to checkpoint directory
        model: model to load weights into
        muon_optimizer: optional Muon optimizer to restore
        adamw_optimizer: optional AdamW optimizer to restore
        scheduler: optional scheduler to restore
        load_optimizer: whether to load optimizer states
        
    Returns:
        metadata dict with step, epoch, loss, config
    """
    # Load model weights
    model_path = os.path.join(checkpoint_path, "model.pt")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model from {model_path}")

    # Load optimizer states
    if load_optimizer:
        if muon_optimizer is not None:
            muon_path = os.path.join(checkpoint_path, "muon_optimizer.pt")
            if os.path.exists(muon_path):
                muon_optimizer.load_state_dict(
                    torch.load(muon_path, map_location="cpu", weights_only=True)
                )
                logger.info("Loaded Muon optimizer state")

        if adamw_optimizer is not None:
            adamw_path = os.path.join(checkpoint_path, "adamw_optimizer.pt")
            if os.path.exists(adamw_path):
                adamw_optimizer.load_state_dict(
                    torch.load(adamw_path, map_location="cpu", weights_only=True)
                )
                logger.info("Loaded AdamW optimizer state")

        if scheduler is not None:
            sched_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.exists(sched_path):
                scheduler.load_state_dict(
                    torch.load(sched_path, map_location="cpu", weights_only=True)
                )
                logger.info("Loaded scheduler state")

    # Load metadata
    meta_path = os.path.join(checkpoint_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"step": 0, "epoch": 0, "loss": float("inf")}

    return metadata


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*"))
    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-step-")[-1]))
    return checkpoints[-1]


def _cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int):
    """Remove old checkpoints, keeping only the last N."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*"))
    if len(checkpoints) <= keep_last_n:
        return

    checkpoints.sort(key=lambda x: int(x.split("-step-")[-1]))
    to_remove = checkpoints[:-keep_last_n]

    for ckpt in to_remove:
        import shutil
        shutil.rmtree(ckpt)
        logger.info(f"Removed old checkpoint: {ckpt}")
