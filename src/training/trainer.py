"""
Main training loop for the Hybrid MoE LLM.
Supports progressive context extension, dense->sparse attention transition,
dual optimizer (Muon + AdamW), gradient accumulation, and checkpointing.
"""

import torch
import logging
import os
from dataclasses import asdict

from src.model.config import ModelConfig, TrainingConfig
from src.model.model import HybridMoEModel
from src.training.muon import create_optimizer_groups
from src.training.scheduler import WarmupCosineScheduler
from src.training.losses import compute_total_loss
from src.training.data import create_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.utils.logging import TrainingLogger

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer with DeepSeek V4 training methodology:
    - Muon + AdamW dual optimizer
    - Progressive context length (4K -> 16K -> 64K)
    - Dense attention warmup before sparse
    - Auxiliary-loss-free load balancing
    - MTP loss weight annealing
    - Gradient checkpointing
    - Periodic checkpoint saving with cleanup
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainingConfig):
        self.model_config = model_config
        self.train_config = train_config
        self.device = self._get_device()

    def _get_device(self):
        """Detect the best available device."""
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

    def train(self, resume_from: str = None):
        """Run the full training loop."""
        tc = self.train_config

        # Initialize logging
        train_logger = TrainingLogger(
            project=tc.wandb_project,
            run_name=tc.wandb_run_name,
            use_wandb=True,
            log_every=tc.log_every_steps,
        )

        # Build model
        logger.info("Building model...")
        model = HybridMoEModel(
            self.model_config,
            use_gradient_checkpointing=tc.gradient_checkpointing,
        )

        # Set hash routing for early layers
        model.set_hash_routing_layers(tc.hash_routing_layers)

        # Log parameter count
        param_info = model.count_parameters()
        logger.info(f"Total params: {param_info['total_billions']:.2f}B")
        logger.info(f"Activated params: {param_info['activated_billions']:.2f}B")

        model = model.to(self.device)

        # Create optimizers
        muon_opt, adamw_opt = create_optimizer_groups(model, tc)

        # Calculate total steps across all context stages
        total_steps = sum(s.total_steps for s in tc.context_stages)

        # Create scheduler
        scheduler = WarmupCosineScheduler(
            optimizers=[muon_opt, adamw_opt],
            warmup_steps=tc.warmup_steps,
            total_steps=total_steps,
            peak_lr=tc.peak_lr,
            min_lr=tc.min_lr,
        )

        # Resume from checkpoint if specified
        global_step = 0
        start_epoch = 0
        if resume_from is None:
            resume_from = find_latest_checkpoint(tc.checkpoint.checkpoint_dir)
        if resume_from is not None:
            logger.info(f"Resuming from {resume_from}")
            metadata = load_checkpoint(
                resume_from, model, muon_opt, adamw_opt, scheduler
            )
            global_step = metadata.get("step", 0)
            start_epoch = metadata.get("epoch", 0)

        # Training loop across context stages
        for stage_idx, stage in enumerate(tc.context_stages):
            logger.info(
                f"=== Stage {stage_idx}: seq_len={stage.seq_len}, "
                f"attention={stage.attention_mode}, steps={stage.total_steps} ==="
            )

            use_sparse = (stage.attention_mode == "sparse")

            # Create dataloader for this stage
            dataloader = create_dataloader(
                dataset_name=tc.dataset_name,
                dataset_subset=tc.dataset_subset,
                tokenizer_path=tc.tokenizer_path,
                seq_len=stage.seq_len,
                batch_size=tc.batch_size,
                num_workers=tc.num_workers,
            )

            # Determine MTP loss weight
            mtp_weight = tc.mtp_loss_weight
            if stage_idx == len(tc.context_stages) - 1:
                mtp_weight = tc.mtp_loss_weight_decay  # Reduce at final stage

            stage_step = 0
            data_iter = iter(dataloader)

            while stage_step < stage.total_steps:
                # Skip steps if resuming
                if global_step < stage_step:
                    stage_step += 1
                    global_step += 1
                    continue

                # Get batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                # Forward pass
                model.train()
                output = model(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    use_sparse=use_sparse,
                )

                # Compute loss
                total_loss, loss_dict = compute_total_loss(
                    output,
                    balance_loss_weight=tc.balance_loss_weight,
                    mtp_loss_weight=mtp_weight,
                )

                # Backward pass (with gradient accumulation)
                scaled_loss = total_loss / tc.grad_accum_steps
                scaled_loss.backward()

                if (stage_step + 1) % tc.grad_accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Optimizer step
                    muon_opt.step()
                    adamw_opt.step()
                    muon_opt.zero_grad()
                    adamw_opt.zero_grad()

                    # Update LR
                    lr = scheduler.step()

                    # Update MoE load-balancing biases
                    for layer in model.layers:
                        if hasattr(layer.moe.router, 'update_bias'):
                            layer.moe.router.update_bias()

                # Logging
                loss_dict["tokens_processed"] = input_ids.numel()
                train_logger.log_step(global_step, loss_dict, lr=scheduler.get_lr())

                # Save checkpoint
                if (global_step + 1) % tc.checkpoint.save_every_steps == 0:
                    save_checkpoint(
                        model=model,
                        muon_optimizer=muon_opt,
                        adamw_optimizer=adamw_opt,
                        scheduler=scheduler,
                        step=global_step,
                        epoch=0,
                        loss=loss_dict["total_loss"],
                        config_dict={
                            "model": asdict(self.model_config),
                            "training": asdict(self.train_config),
                        },
                        checkpoint_dir=tc.checkpoint.checkpoint_dir,
                        keep_last_n=tc.checkpoint.keep_last_n,
                        save_optimizer=tc.checkpoint.save_optimizer,
                    )

                stage_step += 1
                global_step += 1

        # Final checkpoint
        save_checkpoint(
            model=model,
            muon_optimizer=muon_opt,
            adamw_optimizer=adamw_opt,
            scheduler=scheduler,
            step=global_step,
            epoch=0,
            loss=loss_dict.get("total_loss", 0),
            config_dict={
                "model": asdict(self.model_config),
                "training": asdict(self.train_config),
            },
            checkpoint_dir=tc.checkpoint.checkpoint_dir,
            keep_last_n=tc.checkpoint.keep_last_n,
            save_optimizer=tc.checkpoint.save_optimizer,
        )

        train_logger.finish()
        logger.info(f"Training complete. Final step: {global_step}")
