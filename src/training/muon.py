"""
Muon Optimizer — DeepSeek V4 style.
Hybrid Newton-Schulz iterations for orthogonalization.
Applied to most parameters; AdamW for embeddings, norms, mHC statics.
"""

import torch
from torch.optim import Optimizer
import math


class Muon(Optimizer):
    """
    Muon optimizer with:
    - Hybrid Newton-Schulz (10 iters: 8 fast + 2 stabilization)
    - Nesterov momentum
    - RMS rescaling of update matrix
    - Weight decay
    """

    def __init__(
        self,
        params,
        lr: float = 2.7e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        rms_rescale: float = 0.18,
        ns_steps: int = 10,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rms_rescale=rms_rescale,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            gamma = group['rms_rescale']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize momentum buffer
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Accumulate momentum
                buf.mul_(mu).add_(grad)

                # Nesterov trick
                update = mu * buf + grad

                # Apply Newton-Schulz orthogonalization for 2D+ parameters
                if p.dim() >= 2:
                    update = self._newton_schulz(update, group['ns_steps'])

                    # RMS rescaling
                    n, m = update.shape[0], update.shape[1] if update.dim() > 1 else 1
                    rms_scale = math.sqrt(max(n, m)) * gamma
                    update = update * rms_scale
                else:
                    # For 1D params (biases, etc.), just use the raw update
                    pass

                # Weight decay
                p.mul_(1 - lr * wd)

                # Update
                p.add_(update, alpha=-lr)

        return loss

    def _newton_schulz(self, M: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Hybrid Newton-Schulz iterations for approximate orthogonalization.
        
        Stage 1 (8 steps): (a, b, c) = (3.4445, -4.7750, 2.0315) — fast convergence
        Stage 2 (2 steps): (a, b, c) = (2, -1.5, 0.5) — precise stabilization
        """
        # Reshape to 2D if needed
        original_shape = M.shape
        if M.dim() > 2:
            M = M.reshape(M.shape[0], -1)

        # Normalize by Frobenius norm
        M = M / (M.norm() + 1e-7)

        for i in range(num_steps):
            if i < 8:
                a, b, c = 3.4445, -4.7750, 2.0315
            else:
                a, b, c = 2.0, -1.5, 0.5

            MMT = M @ M.T
            M = a * M + b * (MMT @ M) + c * (MMT @ MMT @ M)

        return M.reshape(original_shape)


def create_optimizer_groups(model, training_config) -> tuple:
    """
    Create separate parameter groups for Muon and AdamW.
    
    Muon: most parameters
    AdamW: embedding, prediction head, RMSNorm weights, mHC static biases/gating
    
    Returns:
        muon_params: list of parameters for Muon
        adamw_params: list of parameters for AdamW
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # AdamW group: embeddings, norms, mHC statics, LM head
        is_adamw = (
            'embedding' in name or
            'lm_head' in name or
            'norm' in name.lower() or
            'S_pre' in name or 'S_res' in name or 'S_post' in name or
            'alpha_' in name or
            'sink_logits' in name or
            'bias' in name
        )

        if is_adamw:
            adamw_params.append(param)
        else:
            muon_params.append(param)

    muon_optimizer = Muon(
        muon_params,
        lr=training_config.peak_lr,
        momentum=training_config.muon_momentum,
        weight_decay=training_config.muon_weight_decay,
        rms_rescale=training_config.muon_rms_rescale,
    )

    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=training_config.peak_lr,
        betas=(training_config.adamw_beta1, training_config.adamw_beta2),
        eps=training_config.adamw_eps,
        weight_decay=training_config.adamw_weight_decay,
    )

    return muon_optimizer, adamw_optimizer
