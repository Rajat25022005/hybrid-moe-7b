"""
Training metrics logging — supports wandb and console output.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Logs training metrics to console and optionally to wandb."""

    def __init__(
        self,
        project: str = "hybrid-moe-7b",
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        log_every: int = 10,
    ):
        self.log_every = log_every
        self.use_wandb = use_wandb
        self.start_time = time.time()
        self.step_start_time = time.time()

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name)
                self.wandb = wandb
            except (ImportError, Exception) as e:
                logger.warning(f"wandb not available: {e}. Falling back to console only.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def log_step(self, step: int, metrics: dict, lr: float = None):
        """Log metrics for a training step."""
        if step % self.log_every != 0:
            return

        elapsed = time.time() - self.step_start_time
        self.step_start_time = time.time()

        # Add computed metrics
        metrics["lr"] = lr
        metrics["step_time_s"] = elapsed
        metrics["tokens_per_sec"] = metrics.get("tokens_processed", 0) / max(elapsed, 1e-6)

        # Console log
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        logger.info(" | ".join(parts))

        # wandb log
        if self.use_wandb and self.wandb:
            self.wandb.log(metrics, step=step)

    def log_eval(self, step: int, metrics: dict):
        """Log evaluation metrics."""
        logger.info(f"[EVAL step={step}] " + " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ))
        if self.use_wandb and self.wandb:
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            self.wandb.log(eval_metrics, step=step)

    def finish(self):
        """Finish logging."""
        total_time = time.time() - self.start_time
        logger.info(f"Training finished. Total time: {total_time:.1f}s")
        if self.use_wandb and self.wandb:
            self.wandb.finish()
