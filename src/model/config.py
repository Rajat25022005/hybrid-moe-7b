"""
Model and training configuration dataclasses.
All hyperparameters for the 7B Hybrid MoE LLM.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class AttentionConfig:
    """Shared attention configuration for Full/CSA/HCA."""
    num_query_heads: int = 32
    head_dim: int = 64
    csa_hca_head_dim: int = 128
    query_compress_dim: int = 512
    num_output_groups: int = 4
    group_output_dim: int = 512
    rope_dim: int = 64
    sliding_window_size: int = 128


@dataclass
class CSAConfig:
    """Compressed Sparse Attention configuration."""
    compress_rate: int = 4
    indexer_heads: int = 32
    indexer_head_dim: int = 64
    topk: int = 128


@dataclass
class HCAConfig:
    """Heavily Compressed Attention configuration."""
    compress_rate: int = 64


@dataclass
class Mamba2Config:
    """Mamba2 SSD block configuration."""
    state_dim: int = 128
    conv_dim: int = 4
    expand: int = 2
    head_dim: int = 64
    num_heads: int = 32


@dataclass
class MoEConfig:
    """DeepSeekMoE configuration."""
    num_shared_experts: int = 1
    num_routed_experts: int = 32
    num_active_experts: int = 4
    expert_intermediate_dim: int = 1024
    shared_expert_intermediate_dim: int = 2048
    activation: str = "sqrt_softplus"


@dataclass
class MHCConfig:
    """Manifold-Constrained Hyper-Connections configuration."""
    expansion: int = 4
    sinkhorn_iters: int = 20


@dataclass
class MTPConfig:
    """Multi-Token Prediction configuration."""
    depth: int = 1


@dataclass
class ContextStage:
    """Progressive context training stage."""
    seq_len: int = 4096
    attention_mode: str = "dense"  # "dense" or "sparse"
    total_steps: int = 50000


@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""
    checkpoint_dir: str = "./checkpoints"
    save_every_steps: int = 1000
    keep_last_n: int = 5
    save_optimizer: bool = True


@dataclass
class ModelConfig:
    """Top-level model configuration."""
    hidden_dim: int = 2048
    num_layers: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 4096

    layer_types: List[str] = field(default_factory=lambda: [
        "full_attn", "mamba2", "csa", "mamba2", "ffn_only", "mamba2",
        "hca", "ffn_only", "mamba2", "csa", "ffn_only", "mamba2",
        "hca", "ffn_only", "mamba2", "full_attn", "ffn_only", "mamba2",
        "csa", "ffn_only", "mamba2", "hca", "ffn_only", "mamba2",
        "ffn_only", "mamba2", "ffn_only", "mamba2", "ffn_only", "mamba2",
        "ffn_only", "ffn_only",
    ])

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    csa: CSAConfig = field(default_factory=CSAConfig)
    hca: HCAConfig = field(default_factory=HCAConfig)
    mamba2: Mamba2Config = field(default_factory=Mamba2Config)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mhc: MHCConfig = field(default_factory=MHCConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)

    def __post_init__(self):
        assert len(self.layer_types) == self.num_layers, (
            f"layer_types length ({len(self.layer_types)}) != num_layers ({self.num_layers})"
        )
        valid_types = {"full_attn", "csa", "hca", "mamba2", "ffn_only"}
        for lt in self.layer_types:
            assert lt in valid_types, f"Invalid layer type: {lt}"


@dataclass
class TrainingConfig:
    """Training configuration."""
    precision: str = "bf16"
    gradient_checkpointing: bool = True

    # Muon optimizer
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_rms_rescale: float = 0.18

    # AdamW optimizer (for embeddings, norms, mHC statics)
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_eps: float = 1e-20
    adamw_weight_decay: float = 0.1

    # LR schedule
    warmup_steps: int = 2000
    peak_lr: float = 2.7e-4
    min_lr: float = 2.7e-5

    # Batch
    batch_size: int = 4
    grad_accum_steps: int = 8

    # Progressive context
    context_stages: List[ContextStage] = field(default_factory=lambda: [
        ContextStage(seq_len=4096, attention_mode="dense", total_steps=50000),
        ContextStage(seq_len=16384, attention_mode="dense", total_steps=20000),
        ContextStage(seq_len=65536, attention_mode="sparse", total_steps=10000),
    ])

    # MoE balance
    aux_loss_free: bool = True
    bias_update_speed: float = 0.001
    balance_loss_weight: float = 0.0001
    hash_routing_layers: int = 3

    # MTP
    mtp_loss_weight: float = 0.3
    mtp_loss_weight_decay: float = 0.1

    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Logging
    log_every_steps: int = 10
    eval_every_steps: int = 2000
    wandb_project: str = "hybrid-moe-7b"
    wandb_run_name: Optional[str] = None

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    tokenizer_path: str = "./tokenizer"
    num_workers: int = 4


def load_config(path: str) -> tuple:
    """Load model and training configs from YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    train_raw = raw.get("training", {})

    # Build nested configs
    attn_cfg = AttentionConfig(
        num_query_heads=model_raw.get("num_query_heads", 32),
        head_dim=model_raw.get("head_dim", 64),
        csa_hca_head_dim=model_raw.get("csa_hca_head_dim", 128),
        query_compress_dim=model_raw.get("query_compress_dim", 512),
        num_output_groups=model_raw.get("num_output_groups", 4),
        group_output_dim=model_raw.get("group_output_dim", 512),
        rope_dim=model_raw.get("rope_dim", 64),
        sliding_window_size=model_raw.get("sliding_window_size", 128),
    )
    csa_cfg = CSAConfig(
        compress_rate=model_raw.get("csa_compress_rate", 4),
        indexer_heads=model_raw.get("csa_indexer_heads", 32),
        indexer_head_dim=model_raw.get("csa_indexer_head_dim", 64),
        topk=model_raw.get("csa_topk", 128),
    )
    hca_cfg = HCAConfig(compress_rate=model_raw.get("hca_compress_rate", 64))
    mamba2_cfg = Mamba2Config(
        state_dim=model_raw.get("mamba2_state_dim", 128),
        conv_dim=model_raw.get("mamba2_conv_dim", 4),
        expand=model_raw.get("mamba2_expand", 2),
        head_dim=model_raw.get("mamba2_head_dim", 64),
        num_heads=model_raw.get("mamba2_num_heads", 32),
    )
    moe_cfg = MoEConfig(
        num_shared_experts=model_raw.get("num_shared_experts", 1),
        num_routed_experts=model_raw.get("num_routed_experts", 32),
        num_active_experts=model_raw.get("num_active_experts", 4),
        expert_intermediate_dim=model_raw.get("expert_intermediate_dim", 1024),
        shared_expert_intermediate_dim=model_raw.get("shared_expert_intermediate_dim", 2048),
        activation=model_raw.get("moe_activation", "sqrt_softplus"),
    )
    mhc_cfg = MHCConfig(
        expansion=model_raw.get("mhc_expansion", 4),
        sinkhorn_iters=model_raw.get("mhc_sinkhorn_iters", 20),
    )
    mtp_cfg = MTPConfig(depth=model_raw.get("mtp_depth", 1))

    model_config = ModelConfig(
        hidden_dim=model_raw.get("hidden_dim", 2048),
        num_layers=model_raw.get("num_layers", 32),
        vocab_size=model_raw.get("vocab_size", 32000),
        max_seq_len=model_raw.get("max_seq_len", 4096),
        layer_types=model_raw.get("layer_types", ModelConfig.__dataclass_fields__['layer_types'].default_factory()),
        attention=attn_cfg,
        csa=csa_cfg,
        hca=hca_cfg,
        mamba2=mamba2_cfg,
        moe=moe_cfg,
        mhc=mhc_cfg,
        mtp=mtp_cfg,
    )

    # Context stages
    ctx_stages = []
    for s in train_raw.get("context_stages", []):
        ctx_stages.append(ContextStage(**s))

    ckpt_raw = {
        "checkpoint_dir": train_raw.get("checkpoint_dir", "./checkpoints"),
        "save_every_steps": train_raw.get("save_every_steps", 1000),
        "keep_last_n": train_raw.get("keep_last_n", 5),
        "save_optimizer": train_raw.get("save_optimizer", True),
    }

    train_config = TrainingConfig(
        precision=train_raw.get("precision", "bf16"),
        gradient_checkpointing=train_raw.get("gradient_checkpointing", True),
        muon_momentum=train_raw.get("muon_momentum", 0.95),
        muon_weight_decay=train_raw.get("muon_weight_decay", 0.1),
        muon_rms_rescale=train_raw.get("muon_rms_rescale", 0.18),
        adamw_beta1=train_raw.get("adamw_beta1", 0.9),
        adamw_beta2=train_raw.get("adamw_beta2", 0.95),
        adamw_eps=train_raw.get("adamw_eps", 1e-20),
        adamw_weight_decay=train_raw.get("adamw_weight_decay", 0.1),
        warmup_steps=train_raw.get("warmup_steps", 2000),
        peak_lr=train_raw.get("peak_lr", 2.7e-4),
        min_lr=train_raw.get("min_lr", 2.7e-5),
        batch_size=train_raw.get("batch_size", 4),
        grad_accum_steps=train_raw.get("grad_accum_steps", 8),
        context_stages=ctx_stages if ctx_stages else TrainingConfig.__dataclass_fields__['context_stages'].default_factory(),
        aux_loss_free=train_raw.get("aux_loss_free", True),
        bias_update_speed=train_raw.get("bias_update_speed", 0.001),
        balance_loss_weight=train_raw.get("balance_loss_weight", 0.0001),
        hash_routing_layers=train_raw.get("hash_routing_layers", 3),
        mtp_loss_weight=train_raw.get("mtp_loss_weight", 0.3),
        mtp_loss_weight_decay=train_raw.get("mtp_loss_weight_decay", 0.1),
        checkpoint=CheckpointConfig(**ckpt_raw),
        log_every_steps=train_raw.get("log_every_steps", 10),
        eval_every_steps=train_raw.get("eval_every_steps", 2000),
        wandb_project=train_raw.get("wandb_project", "hybrid-moe-7b"),
        wandb_run_name=train_raw.get("wandb_run_name"),
        dataset_name=train_raw.get("dataset_name", "HuggingFaceFW/fineweb-edu"),
        dataset_subset=train_raw.get("dataset_subset", "sample-10BT"),
        tokenizer_path=train_raw.get("tokenizer_path", "./tokenizer"),
        num_workers=train_raw.get("num_workers", 4),
    )

    return model_config, train_config
