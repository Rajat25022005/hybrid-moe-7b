#!/usr/bin/env python3
"""
Smoke test for the full Hybrid MoE model.
Verifies forward pass, backward pass, and gradient flow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest

from src.model.config import ModelConfig
from src.model.model import HybridMoEModel


class TestHybridMoEModel(unittest.TestCase):
    """Integration tests for the full model."""

    def setUp(self):
        """Create a small model config for testing."""
        self.config = ModelConfig(
            hidden_dim=128,
            num_layers=4,
            vocab_size=1000,
            max_seq_len=256,
            layer_types=["full_attn", "mamba2", "csa", "ffn_only"],
        )
        # Override sub-configs for small size
        self.config.attention.num_query_heads = 4
        self.config.attention.head_dim = 32
        self.config.attention.csa_hca_head_dim = 32
        self.config.attention.query_compress_dim = 64
        self.config.attention.num_output_groups = 2
        self.config.attention.group_output_dim = 64
        self.config.csa.indexer_heads = 4
        self.config.csa.indexer_head_dim = 16
        self.config.csa.topk = 8
        self.config.mamba2.state_dim = 16
        self.config.mamba2.head_dim = 32
        self.config.mamba2.num_heads = 4
        self.config.moe.num_routed_experts = 4
        self.config.moe.num_active_experts = 2
        self.config.moe.expert_intermediate_dim = 128
        self.config.moe.shared_expert_intermediate_dim = 128
        self.config.mhc.expansion = 2
        self.config.mhc.sinkhorn_iters = 5

    def test_forward_pass(self):
        """Test that forward pass produces correct output shapes."""
        model = HybridMoEModel(self.config, use_gradient_checkpointing=False)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        target_ids = torch.randint(0, self.config.vocab_size, (2, 32))

        output = model(input_ids, target_ids=target_ids, use_sparse=False)

        self.assertEqual(output["logits"].shape, (2, 32, self.config.vocab_size))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["mtp_loss"]))

    def test_backward_pass(self):
        """Test that backward pass produces gradients."""
        model = HybridMoEModel(self.config, use_gradient_checkpointing=False)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 16))
        target_ids = torch.randint(0, self.config.vocab_size, (1, 16))

        output = model(input_ids, target_ids=target_ids, use_sparse=False)
        loss = output["loss"]
        loss.backward()

        has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
        total = sum(1 for p in model.parameters() if p.requires_grad)
        self.assertGreater(has_grad, 0, "No parameters received gradients")
        print(f"Gradient flow: {has_grad}/{total} parameters have gradients")

    def test_param_count(self):
        """Test parameter counting."""
        model = HybridMoEModel(self.config, use_gradient_checkpointing=False)
        info = model.count_parameters()
        self.assertGreater(info["total"], 0)
        self.assertGreater(info["activated"], 0)
        self.assertLessEqual(info["activated"], info["total"])
        print(f"Params — Total: {info['total']:,}, Activated: {info['activated']:,}")

    def test_inference_no_target(self):
        """Test inference mode (no targets)."""
        model = HybridMoEModel(self.config, use_gradient_checkpointing=False)
        model.eval()
        input_ids = torch.randint(0, self.config.vocab_size, (1, 16))

        with torch.no_grad():
            output = model(input_ids, use_sparse=False)

        self.assertEqual(output["logits"].shape, (1, 16, self.config.vocab_size))


if __name__ == "__main__":
    unittest.main()
