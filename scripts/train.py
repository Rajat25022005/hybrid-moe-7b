#!/usr/bin/env python3
"""
Training entry point for the Hybrid MoE LLM.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --resume ./checkpoints/checkpoint-step-5000
    python scripts/train.py --config configs/config.yaml --smoke-test
    python scripts/train.py --config configs/config.yaml --count-params
    python scripts/train.py --config configs/config.yaml --check-gradients
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.config import load_config, ModelConfig
from src.model.model import HybridMoEModel
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_tiny_config():
    """Create a tiny config (~5M params) for local testing on CPU."""
    from src.model.config import (
        AttentionConfig, CSAConfig, HCAConfig, Mamba2Config,
        MoEConfig, MHCConfig, MTPConfig,
    )
    return ModelConfig(
        hidden_dim=128,
        num_layers=4,
        vocab_size=1000,
        max_seq_len=256,
        layer_types=["full_attn", "mamba2", "csa", "ffn_only"],
        attention=AttentionConfig(
            num_query_heads=4, head_dim=32, csa_hca_head_dim=32,
            query_compress_dim=64, num_output_groups=2, group_output_dim=64,
            rope_dim=32, sliding_window_size=32,
        ),
        csa=CSAConfig(compress_rate=4, indexer_heads=4, indexer_head_dim=16, topk=8),
        hca=HCAConfig(compress_rate=16),
        mamba2=Mamba2Config(state_dim=16, conv_dim=4, expand=2, head_dim=32, num_heads=4),
        moe=MoEConfig(
            num_shared_experts=1, num_routed_experts=4, num_active_experts=2,
            expert_intermediate_dim=128, shared_expert_intermediate_dim=128,
        ),
        mhc=MHCConfig(expansion=2, sinkhorn_iters=5),
        mtp=MTPConfig(depth=1),
    )


def smoke_test(model_config: ModelConfig):
    """Run a single forward + backward pass with a TINY model on CPU."""
    logger.info("=== Smoke Test (tiny config for local CPU) ===")

    # Use tiny config so it runs on any machine
    tiny_config = _make_tiny_config()

    device = torch.device("cpu")
    model = HybridMoEModel(tiny_config, use_gradient_checkpointing=False)
    model = model.to(device)

    # Count params
    info = model.count_parameters()
    logger.info(f"Tiny model — Total params: {info['total']:,} ({info['total_billions']:.3f}B)")
    logger.info(f"Tiny model — Activated params: {info['activated']:,} ({info['activated_billions']:.3f}B)")

    # Also report what the FULL model would be (without allocating)
    full_model_temp = HybridMoEModel.__new__(HybridMoEModel)
    full_model_temp.config = model_config
    logger.info(f"Full model config: hidden_dim={model_config.hidden_dim}, layers={model_config.num_layers}")

    # Create random input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logger.info("Running forward pass...")
    output = model(input_ids, target_ids=target_ids, use_sparse=False)
    logger.info(f"Logits shape: {output['logits'].shape}")
    logger.info(f"LM loss: {output['loss'].item():.4f}")
    logger.info(f"MTP loss: {output['mtp_loss'].item():.4f}")
    logger.info(f"Aux loss: {output['aux_loss'].item():.4f}")

    # Backward pass
    logger.info("Running backward pass...")
    total_loss = output['loss'] + 0.3 * output['mtp_loss'] + 0.0001 * output['aux_loss']
    total_loss.backward()

    # Check gradient flow
    grad_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None and param.grad.norm().item() > 0:
                grad_count += 1

    logger.info(f"Gradient flow: {grad_count}/{total_params} params have nonzero gradients")
    if grad_count == 0:
        logger.error("NO gradients flowing — model is broken!")
        return False

    logger.info("=== Smoke Test PASSED ===")
    return True


def check_gradients(model_config: ModelConfig):
    """Verify gradient flow through all layers using tiny config."""
    logger.info("=== Gradient Flow Check (tiny config) ===")

    tiny_config = _make_tiny_config()
    model = HybridMoEModel(tiny_config, use_gradient_checkpointing=False)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 32))
    target_ids = torch.randint(0, tiny_config.vocab_size, (1, 32))

    output = model(input_ids, target_ids=target_ids, use_sparse=False)
    loss = output['loss']
    loss.backward()

    for i, layer in enumerate(model.layers):
        has_grad = False
        max_grad = 0.0
        for name, param in layer.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 0:
                has_grad = True
                max_grad = max(max_grad, param.grad.norm().item())

        status = "✓" if has_grad else "✗ DEAD"
        logger.info(f"Layer {i:2d} ({tiny_config.layer_types[i]:10s}): {status} (max_grad={max_grad:.6f})")

    logger.info("=== Gradient Check Complete ===")


def count_params(model_config: ModelConfig):
    """Print detailed parameter count."""
    model = HybridMoEModel(model_config, use_gradient_checkpointing=False)
    info = model.count_parameters()

    logger.info(f"{'='*50}")
    logger.info(f"Total parameters:     {info['total']:>15,} ({info['total_billions']:.3f}B)")
    logger.info(f"Activated parameters: {info['activated']:>15,} ({info['activated_billions']:.3f}B)")
    logger.info(f"{'='*50}")

    # Per-component breakdown
    components = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0]
        if component not in components:
            components[component] = 0
        components[component] += param.numel()

    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        logger.info(f"  {comp:30s}: {count:>12,} ({count/1e6:.1f}M)")


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid MoE LLM")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    parser.add_argument("--count-params", action="store_true", help="Count parameters")
    parser.add_argument("--check-gradients", action="store_true", help="Check gradient flow")
    args = parser.parse_args()

    # Load config
    model_config, train_config = load_config(args.config)

    if args.smoke_test:
        smoke_test(model_config)
        return

    if args.count_params:
        count_params(model_config)
        return

    if args.check_gradients:
        check_gradients(model_config)
        return

    # Full training
    trainer = Trainer(model_config, train_config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
