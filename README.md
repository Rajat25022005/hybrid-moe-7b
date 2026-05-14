# Hybrid MoE LLM — 7B Total / ~1.7B Activated

A 32-layer Mixture-of-Experts language model combining **Full Attention**, **Compressed Sparse Attention (CSA)**, **Heavily Compressed Attention (HCA)**, **Mamba2 SSM**, and **DeepSeekMoE FFN** layers, following the [DeepSeek V4](https://arxiv.org/abs/2505.xxxxx) training methodology.

## Architecture

| Component | Details |
|-----------|---------|
| **Total Params** | ~7B |
| **Activated Params** | ~1.7B per token |
| **Layers** | 32 (12 Mamba2 + 12 FFN-only + 2 Full Attn + 3 CSA + 3 HCA) |
| **Hidden Dim** | 2048 |
| **MoE** | 32 routed experts (top-4) + 1 shared expert per layer |
| **Residual** | Manifold-Constrained Hyper-Connections (mHC) |
| **Training** | Multi-Token Prediction + Muon optimizer |

## Key Features

- **Hybrid attention**: Full Attention anchors + Compressed Sparse Attention + Heavily Compressed Attention
- **Mamba2 SSM layers**: State space models for efficient sequence mixing (pure PyTorch, TPU-compatible)
- **DeepSeekMoE**: Fine-grained experts with Sqrt(Softplus) affinity and auxiliary-loss-free load balancing
- **mHC**: Doubly stochastic residual connections via Sinkhorn-Knopp
- **Progressive context**: 4K → 16K → 64K with dense-to-sparse attention transition
- **Muon optimizer**: Hybrid Newton-Schulz orthogonalization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (CPU, no data needed)
python scripts/train.py --config configs/config.yaml --smoke-test

# Count parameters
python scripts/train.py --config configs/config.yaml --count-params

# Check gradient flow
python scripts/train.py --config configs/config.yaml --check-gradients

# Train tokenizer
python scripts/train_tokenizer.py --output ./tokenizer --vocab-size 32000

# Full training (TPU v5e-8)
python scripts/train.py --config configs/config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/config.yaml --resume ./checkpoints/checkpoint-step-5000

# Evaluate
python scripts/evaluate.py --checkpoint ./checkpoints/checkpoint-step-80000
```

## Checkpointing

- Checkpoints saved every 1000 steps (configurable)
- Keeps last 5 checkpoints automatically (configurable)
- Saves model weights, both optimizers (Muon + AdamW), and LR scheduler
- Resume training seamlessly from any checkpoint

## Evaluation

Comparable models for benchmarking:

| Model | Total | Active | Architecture |
|-------|-------|--------|-------------|
| **OLMoE-1B-7B** | 7B | 1B | MoE (primary comparison) |
| **Gemma-2-2B** | 2.6B | 2.6B | Dense (similar activated) |
| **Llama-2-7B** | 7B | 7B | Dense (upper bound) |

Benchmarks: MMLU, HellaSwag, ARC-Challenge, GSM8K, TruthfulQA, WinoGrande

## Project Structure

```
MoE/
├── configs/config.yaml          # All hyperparameters
├── src/model/                   # Model architecture
│   ├── attention/               # Full Attn, CSA, HCA, RoPE
│   ├── mamba2/                  # Mamba2 SSD block
│   ├── moe/                     # DeepSeekMoE + routing
│   ├── mhc/                     # Hyper-connections
│   └── mtp/                     # Multi-token prediction
├── src/training/                # Training pipeline
│   ├── muon.py                  # Muon optimizer
│   ├── trainer.py               # Main training loop
│   └── data.py                  # Data pipeline
├── src/utils/                   # Checkpoint + logging
├── scripts/                     # Entry points
└── tests/                       # Tests
```

## License

Research use.
