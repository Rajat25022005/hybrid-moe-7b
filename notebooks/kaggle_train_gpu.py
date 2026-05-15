"""
Kaggle Notebook for training Hybrid MoE on T4 GPU (CUDA).

Instructions:
1. Create a new Kaggle notebook
2. Set Accelerator to "GPU T4 x2" or "GPU T4 x1" (Settings → Accelerator)
3. Paste cells and run in order
"""

# =============================================================================
# CELL 1: Setup & Install Dependencies
# =============================================================================
import subprocess, sys, os

def run(cmd):
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

run("pip install -q pyyaml einops tokenizers datasets wandb safetensors tqdm")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    device = torch.device("cuda")
else:
    print("WARNING: No GPU found, falling back to CPU")
    device = torch.device("cpu")

# =============================================================================
# CELL 2: Get the Code
# =============================================================================
CODE_DIR = "/kaggle/working/hybrid-moe-7b"

if not os.path.exists(CODE_DIR):
    run("git clone https://github.com/Rajat25022005/hybrid-moe-7b.git " + CODE_DIR)
else:
    print(f"Code already exists at {CODE_DIR}, pulling latest...")
    run(f"cd {CODE_DIR} && git pull")

os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)
print(f"Working directory: {os.getcwd()}")

# =============================================================================
# CELL 3: Smoke Test (Tiny Model — ~10 seconds)
# =============================================================================
print("\n" + "="*60)
print("SMOKE TEST — Tiny model on GPU")
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

model = HybridMoEModel(tiny_config, use_gradient_checkpointing=True).to(device)
x = torch.randint(0, 1000, (2, 64)).to(device)
t = torch.randint(0, 1000, (2, 64)).to(device)

with torch.amp.autocast('cuda', dtype=torch.float16):
    out = model(x, target_ids=t, use_sparse=False)
    loss = out['loss'] + 0.3 * out['mtp_loss']

loss.backward()

print(f"✓ Forward:  logits={out['logits'].shape}, loss={out['loss'].item():.4f}")
print(f"✓ Backward: gradients computed successfully")
print(f"✓ GPU smoke test PASSED on {device}")

del model, x, t, out, loss
torch.cuda.empty_cache()
import gc; gc.collect()

# =============================================================================
# CELL 4: Build Full Model on GPU
# =============================================================================
print("\n" + "="*60)
print("BUILDING MODEL")
print("="*60)

from src.model.config import load_config
from src.training.muon import create_optimizer_groups
from src.training.scheduler import WarmupCosineScheduler
from src.training.losses import compute_total_loss
from src.training.data import create_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
import time

# Load T4 config
model_config, train_config = load_config("configs/config_t4.yaml")

# Override checkpoint dir for Kaggle
train_config.checkpoint.checkpoint_dir = "/kaggle/working/checkpoints"

# Wandb (set True if configured)
USE_WANDB = False

# Build model with gradient checkpointing
model = HybridMoEModel(
    model_config,
    use_gradient_checkpointing=train_config.gradient_checkpointing,
)
model.set_hash_routing_layers(train_config.hash_routing_layers)

# Count params
param_info = model.count_parameters()
print(f"Total params:     {param_info['total_billions']:.3f}B ({param_info['total']:,})")
print(f"Activated params: {param_info['activated_billions']:.3f}B ({param_info['activated']:,})")

# Move to GPU
model = model.to(device)
print(f"✓ Model on {device}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

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

# Mixed precision scaler
scaler = torch.amp.GradScaler('cuda')

# Resume from checkpoint if exists
global_step = 0
ckpt_path = find_latest_checkpoint(train_config.checkpoint.checkpoint_dir)
if ckpt_path:
    print(f"Resuming from {ckpt_path}")
    meta = load_checkpoint(ckpt_path, model, muon_opt, adamw_opt, scheduler)
    global_step = meta.get("step", 0)
    print(f"Resumed at step {global_step}")

print(f"GPU memory after setup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# =============================================================================
# CELL 5: Training Loop
# =============================================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

if USE_WANDB:
    import wandb
    wandb.init(project=train_config.wandb_project, name="kaggle-t4-gpu")

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
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Forward with mixed precision
        model.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids=input_ids, target_ids=target_ids, use_sparse=use_sparse)
            total_loss, loss_dict = compute_total_loss(
                output,
                balance_loss_weight=train_config.balance_loss_weight,
                mtp_loss_weight=mtp_weight,
            )
            scaled_loss = total_loss / train_config.grad_accum_steps

        # Backward with scaler
        scaler.scale(scaled_loss).backward()

        if (stage_step + 1) % train_config.grad_accum_steps == 0:
            # Unscale, clip, step
            scaler.unscale_(muon_opt)
            scaler.unscale_(adamw_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(muon_opt)
            scaler.step(adamw_opt)
            scaler.update()

            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)

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
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(
                f"step={global_step:>6d} | "
                f"loss={loss_dict['total_loss']:.4f} | "
                f"lm={loss_dict['lm_loss']:.4f} | "
                f"mtp={loss_dict['mtp_loss']:.4f} | "
                f"lr={scheduler.get_lr():.2e} | "
                f"tok/s={tokens_per_sec:.0f} | "
                f"mem={gpu_mem:.1f}GB"
            )
            if USE_WANDB:
                wandb.log({**loss_dict, "lr": lr, "tokens_per_sec": tokens_per_sec, "gpu_mem": gpu_mem}, step=global_step)
            step_start = time.time()

        # Checkpoint
        if (global_step + 1) % train_config.checkpoint.save_every_steps == 0:
            print(f"Saving checkpoint at step {global_step}...")
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
