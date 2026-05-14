#!/usr/bin/env python3
"""
Train a custom BPE tokenizer on the training corpus.

Usage:
    python scripts/train_tokenizer.py --output ./tokenizer --vocab-size 32000
"""

import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tokenizer(
    output_dir: str,
    vocab_size: int = 32000,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str = "sample-10BT",
    num_samples: int = 100000,
):
    """Train a BPE tokenizer on training data samples."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from datasets import load_dataset

    logger.info(f"Loading {num_samples} samples from {dataset_name}...")
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)

    # Collect text samples
    texts = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        text = sample.get("text", "")
        if text:
            texts.append(text)

    logger.info(f"Collected {len(texts)} text samples")

    # Train BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"],
        show_progress=True,
    )

    logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    logger.info(f"Tokenizer saved to {output_dir}/tokenizer.json")

    # Test
    test_text = "Hello, world! This is a test of the tokenizer."
    encoded = tokenizer.encode(test_text)
    logger.info(f"Test: '{test_text}' -> {len(encoded.ids)} tokens")
    logger.info(f"Token IDs: {encoded.ids[:20]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-samples", type=int, default=100000)
    args = parser.parse_args()

    train_tokenizer(args.output, args.vocab_size, num_samples=args.num_samples)
