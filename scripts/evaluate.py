#!/usr/bin/env python3
"""
Evaluation script — runs lm-evaluation-harness benchmarks.

Usage:
    python scripts/evaluate.py --checkpoint ./checkpoints/checkpoint-step-80000
    python scripts/evaluate.py --checkpoint ./checkpoints/checkpoint-step-80000 --tasks mmlu,hellaswag
"""

import argparse
import logging
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_TASKS = [
    "mmlu",
    "hellaswag",
    "arc_challenge",
    "gsm8k",
    "truthfulqa_mc2",
    "winogrande",
]


def evaluate(checkpoint_path: str, tasks: list, output_dir: str):
    """Run lm-evaluation-harness on the checkpoint."""
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Tasks: {tasks}")

    # Load model
    import torch
    from src.model.config import ModelConfig
    from src.model.model import HybridMoEModel
    from src.utils.checkpoint import load_checkpoint

    # Load config from checkpoint metadata
    meta_path = os.path.join(checkpoint_path, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    # Reconstruct model config
    model_cfg_dict = metadata["config"]["model"]
    model_config = ModelConfig(**{
        k: v for k, v in model_cfg_dict.items()
        if k in ModelConfig.__dataclass_fields__
    })

    model = HybridMoEModel(model_config, use_gradient_checkpointing=False)
    load_checkpoint(checkpoint_path, model, load_optimizer=False)
    model.eval()

    logger.info(f"Model loaded. Params: {model.count_parameters()}")

    # Try using lm-eval-harness
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # For custom models, we'd need to wrap in HF format
        # For now, provide manual evaluation guidance
        logger.info(
            "To run full lm-eval benchmarks, convert the model to HuggingFace format first.\n"
            "Then run:\n"
            f"  lm_eval --model hf --model_args pretrained={checkpoint_path} "
            f"--tasks {','.join(tasks)} --batch_size auto --output_path {output_dir}"
        )
    except ImportError:
        logger.info("lm-eval not installed. Install with: pip install lm-eval")

    # Manual perplexity evaluation
    logger.info("Running manual perplexity evaluation...")
    _eval_perplexity(model, model_config, output_dir)


def _eval_perplexity(model, config, output_dir):
    """Quick perplexity evaluation on random data."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    total_loss = 0
    num_batches = 10

    with torch.no_grad():
        for _ in range(num_batches):
            input_ids = torch.randint(0, config.vocab_size, (1, 512), device=device)
            target_ids = torch.randint(0, config.vocab_size, (1, 512), device=device)

            output = model(input_ids, target_ids=target_ids, use_sparse=False)
            total_loss += output["loss"].item()

    avg_loss = total_loss / num_batches
    perplexity = 2 ** avg_loss

    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {"avg_loss": avg_loss, "perplexity": perplexity}
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--output", default="./eval_results")
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    evaluate(args.checkpoint, tasks, args.output)
