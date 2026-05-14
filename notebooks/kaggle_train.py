"""
Kaggle Notebook for training Hybrid MoE 7B on TPU v5e-8.

Instructions:
1. Create a new Kaggle notebook
2. Set Accelerator to "TPU v5e-8" (Settings → Accelerator)
3. Upload your MoE/ project as a Kaggle Dataset (or use GitHub)
4. Paste this entire file into Cell 1 and run

This notebook handles:
- Environment setup & dependency installation
- Code import from dataset/GitHub
- TPU device detection & SPMD sharding
- Progressive context training with checkpointing
"""

# =============================================================================
# CELL 1: Setup & Install Dependencies
# =============================================================================
import subprocess
import sys
import os

def run(cmd):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Install missing deps (torch & torch-xla are pre-installed on Kaggle TPU)
run("pip install -q pyyaml einops tokenizers datasets wandb safetensors tqdm")

# Verify TPU is available
import warnings
warnings.filterwarnings("ignore", message=".*tensorflow.*")

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

print(f"PyTorch: {torch.__version__}")
print(f"Torch-XLA: {torch_xla.__version__}")
print(f"TPU cores available: {xr.global_runtime_device_count()}")
device = torch_xla.device()
print(f"XLA device: {device}")

# =============================================================================
# CELL 2: Get the Code
# =============================================================================
CODE_DIR = "/kaggle/working/hybrid-moe-7b"

if not os.path.exists(CODE_DIR):
    run("git clone https://github.com/Rajat25022005/hybrid-moe-7b.git " + CODE_DIR)
else:
    print(f"Code already exists at {CODE_DIR}")

os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)
print(f"Working directory: {os.getcwd()}")

# =============================================================================
# CELL 3: Quick Smoke Test (Tiny Model — ~30 seconds)
# =============================================================================
print("\n" + "="*60)
print("SMOKE TEST — Tiny model on TPU")
print("="*60)

from src.model.config import (
    ModelConfig, AttentionConfig, CSAConfig, HCAConfig,
    Mamba2Config, MoEConfig, MHCConfig, MTPConfig,
)
from src.model.model import HybridMoEModel

tiny_config = ModelConfig(
    hidden_dim=128, num_layers=4, vocab_size=1000, max_seq_len=256,
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

model = HybridMoEModel(tiny_config, use_gradient_checkpointing=False).to(device)
x = torch.randint(0, 1000, (2, 64)).to(device)
t = torch.randint(0, 1000, (2, 64)).to(device)

out = model(x, target_ids=t, use_sparse=False)
torch_xla.sync()  # Force XLA compilation

loss = out['loss'] + 0.3 * out['mtp_loss']
loss.backward()
torch_xla.sync()

print(f"✓ Forward:  logits={out['logits'].shape}, loss={out['loss'].item():.4f}")
print(f"✓ Backward: gradients computed successfully")
print(f"✓ TPU smoke test PASSED on {device}")

del model, x, t, out  # Free memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc; gc.collect()

# =============================================================================
# CELL 4: Build Full 7B Model on TPU
# =============================================================================
print("\n" + "="*60)
print("BUILDING FULL 7B MODEL")
print("="*60)

from src.model.config import load_config
from src.training.muon import create_optimizer_groups
from src.training.scheduler import WarmupCosineScheduler
from src.training.losses import compute_total_loss
from src.training.data import create_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
import time
import json

# Load config
model_config, train_config = load_config("configs/config_2b.yaml")

# Override checkpoint dir for Kaggle
train_config.checkpoint.checkpoint_dir = "/kaggle/working/checkpoints"
train_config.checkpoint.save_every_steps = 500    # Save more often on Kaggle
train_config.checkpoint.keep_last_n = 3           # Keep fewer due to disk limits

# Override wandb for Kaggle (set your key or disable)
# To enable: run `wandb login YOUR_API_KEY` or set env var
USE_WANDB = False  # Set True if you've configured wandb

# Build model
model = HybridMoEModel(
    model_config,
    use_gradient_checkpointing=train_config.gradient_checkpointing,
)

# Set hash routing for first 3 layers
model.set_hash_routing_layers(train_config.hash_routing_layers)

# Count parameters BEFORE moving to TPU
param_info = model.count_parameters()
print(f"Total params:     {param_info['total_billions']:.3f}B ({param_info['total']:,})")
print(f"Activated params: {param_info['activated_billions']:.3f}B ({param_info['activated']:,})")

# Move to TPU
print("Moving model to TPU...")
model = model.to(device)
torch_xla.sync()
print(f"✓ Model on {device}")

# Create optimizers
muon_opt, adamw_opt = create_optimizer_groups(model, train_config)

# Total steps
total_steps = sum(s.total_steps for s in train_config.context_stages)
print(f"Total training steps: {total_steps:,}")

# Scheduler
scheduler = WarmupCosineScheduler(
    optimizers=[muon_opt, adamw_opt],
    warmup_steps=train_config.warmup_steps,
    total_steps=total_steps,
    peak_lr=train_config.peak_lr,
    min_lr=train_config.min_lr,
)

# Resume from checkpoint if exists
global_step = 0
ckpt_path = find_latest_checkpoint(train_config.checkpoint.checkpoint_dir)
if ckpt_path:
    print(f"Resuming from {ckpt_path}")
    meta = load_checkpoint(ckpt_path, model, muon_opt, adamw_opt, scheduler)
    global_step = meta.get("step", 0)
    print(f"Resumed at step {global_step}")

# =============================================================================
# CELL 5: Training Loop
# =============================================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

if USE_WANDB:
    import wandb
    wandb.init(project=train_config.wandb_project, name="kaggle-tpu-7b")

from dataclasses import asdict

for stage_idx, stage in enumerate(train_config.context_stages):
    print(f"\n--- Stage {stage_idx}: seq_len={stage.seq_len}, "
          f"mode={stage.attention_mode}, steps={stage.total_steps} ---")

    use_sparse = (stage.attention_mode == "sparse")

    # Create dataloader
    dataloader = create_dataloader(
        dataset_name=train_config.dataset_name,
        dataset_subset=train_config.dataset_subset,
        tokenizer_path=train_config.tokenizer_path,
        seq_len=stage.seq_len,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )

    # MTP weight
    mtp_weight = train_config.mtp_loss_weight
    if stage_idx == len(train_config.context_stages) - 1:
        mtp_weight = train_config.mtp_loss_weight_decay

    stage_step = 0
    data_iter = iter(dataloader)
    step_start = time.time()

    while stage_step < stage.total_steps:
        if global_step < 0:  # Skip for resumed training
            stage_step += 1
            global_step += 1
            continue

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Forward
        model.train()
        output = model(input_ids=input_ids, target_ids=target_ids, use_sparse=use_sparse)

        # Loss
        total_loss, loss_dict = compute_total_loss(
            output,
            balance_loss_weight=train_config.balance_loss_weight,
            mtp_loss_weight=mtp_weight,
        )

        # Backward
        scaled_loss = total_loss / train_config.grad_accum_steps
        scaled_loss.backward()

        if (stage_step + 1) % train_config.grad_accum_steps == 0:
            # Clip gradients
            # Note: gradient sync handled by torch_xla on multi-chip automatically
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Step optimizers
            muon_opt.step()
            adamw_opt.step()
            muon_opt.zero_grad()
            adamw_opt.zero_grad()
            torch_xla.sync()  # Execute pending XLA ops

            # LR update
            lr = scheduler.step()

            # Update MoE biases
            for layer in model.layers:
                if hasattr(layer.moe.router, 'update_bias'):
                    layer.moe.router.update_bias()

        # Log
        if global_step % train_config.log_every_steps == 0:
            elapsed = time.time() - step_start
            tokens_per_sec = input_ids.numel() / max(elapsed, 1e-6)
            print(
                f"step={global_step:>6d} | "
                f"loss={loss_dict['total_loss']:.4f} | "
                f"lm={loss_dict['lm_loss']:.4f} | "
                f"mtp={loss_dict['mtp_loss']:.4f} | "
                f"lr={scheduler.get_lr():.2e} | "
                f"tok/s={tokens_per_sec:.0f}"
            )
            if USE_WANDB:
                wandb.log({**loss_dict, "lr": lr, "tokens_per_sec": tokens_per_sec}, step=global_step)
            step_start = time.time()

        # Checkpoint
        if (global_step + 1) % train_config.checkpoint.save_every_steps == 0:
            print(f"Saving checkpoint at step {global_step}...")
            torch_xla.sync()
            # Move model to CPU for saving
            save_checkpoint(
                model=model,
                muon_optimizer=muon_opt,
                adamw_optimizer=adamw_opt,
                scheduler=scheduler,
                step=global_step,
                epoch=0,
                loss=loss_dict["total_loss"],
                config_dict={
                    "model": asdict(model_config),
                    "training": asdict(train_config),
                },
                checkpoint_dir=train_config.checkpoint.checkpoint_dir,
                keep_last_n=train_config.checkpoint.keep_last_n,
                save_optimizer=train_config.checkpoint.save_optimizer,
            )
            print(f"✓ Checkpoint saved")

        stage_step += 1
        global_step += 1

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — {global_step} steps")
print(f"{'='*60}")

# Final save
save_checkpoint(
    model=model, muon_optimizer=muon_opt, adamw_optimizer=adamw_opt,
    scheduler=scheduler, step=global_step, epoch=0,
    loss=loss_dict.get("total_loss", 0),
    config_dict={"model": asdict(model_config), "training": asdict(train_config)},
    checkpoint_dir=train_config.checkpoint.checkpoint_dir,
    keep_last_n=train_config.checkpoint.keep_last_n,
    save_optimizer=True,
)
print(f"Final checkpoint saved to {train_config.checkpoint.checkpoint_dir}")

if USE_WANDB:
    wandb.finish()
