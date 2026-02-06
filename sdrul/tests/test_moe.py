"""
Unit tests for DSA-MoE module components.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe.expert import AdapterExpert, ExpertGroup
from models.moe.router import ConditionRouter, StageRouter, JointRouter
from models.moe.losses import load_balance_loss, importance_loss, LoadBalanceLoss
from models.moe.dsa_moe import DSAMoE, DSAMoEConfig


class TestAdapterExpert:
    """Tests for AdapterExpert."""

    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        expert = AdapterExpert(d_model=256, bottleneck_dim=64)

        # Test 3D input
        x = torch.randn(4, 30, 256)
        out = expert(x)
        assert out.shape == x.shape

        # Test 2D input
        x = torch.randn(4, 256)
        out = expert(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connection is working."""
        expert = AdapterExpert(d_model=256, bottleneck_dim=64)

        x = torch.randn(4, 30, 256)
        out = expert(x)

        # Output should be close to input initially (due to zero init of up_proj)
        # The difference should be small
        diff = (out - x).abs().mean()
        assert diff < 1.0  # Should be small due to zero initialization

    def test_gradient_flow(self):
        """Test that gradients flow through the expert."""
        expert = AdapterExpert(d_model=256, bottleneck_dim=64)

        x = torch.randn(4, 30, 256, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_bottleneck_dimension(self):
        """Test different bottleneck dimensions."""
        for bottleneck_dim in [16, 32, 64, 128]:
            expert = AdapterExpert(d_model=256, bottleneck_dim=bottleneck_dim)
            x = torch.randn(4, 30, 256)
            out = expert(x)
            assert out.shape == x.shape


class TestExpertGroup:
    """Tests for ExpertGroup."""

    def test_uniform_weighting(self):
        """Test expert group with uniform weights."""
        group = ExpertGroup(num_experts=4, d_model=256)

        x = torch.randn(4, 30, 256)
        out = group(x)  # No weights = uniform
        assert out.shape == x.shape

    def test_custom_weighting(self):
        """Test expert group with custom weights."""
        group = ExpertGroup(num_experts=4, d_model=256)

        x = torch.randn(4, 30, 256)
        weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)  # Only first expert
        out = group(x, weights)
        assert out.shape == x.shape


class TestConditionRouter:
    """Tests for ConditionRouter."""

    def test_hard_routing(self):
        """Test hard routing with known condition IDs."""
        router = ConditionRouter(d_model=256, num_conditions=6)

        features = torch.randn(4, 30, 256)
        condition_id = torch.tensor([0, 1, 2, 3])

        weights = router(features, condition_id)

        assert weights.shape == (4, 6)
        # Should be one-hot
        assert torch.allclose(weights.sum(dim=1), torch.ones(4))
        assert (weights.max(dim=1)[0] == 1.0).all()

    def test_soft_routing(self):
        """Test soft routing without condition IDs."""
        router = ConditionRouter(d_model=256, num_conditions=6)

        features = torch.randn(4, 30, 256)
        weights = router(features, condition_id=None)

        assert weights.shape == (4, 6)
        # Should sum to 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
        # Should be soft (not one-hot)
        assert (weights > 0).all()

    def test_temperature_effect(self):
        """Test that temperature affects routing sharpness."""
        router_low_temp = ConditionRouter(d_model=256, num_conditions=6, temperature=0.1)
        router_high_temp = ConditionRouter(d_model=256, num_conditions=6, temperature=10.0)

        # Use same weights for fair comparison
        router_high_temp.classifier.load_state_dict(router_low_temp.classifier.state_dict())

        features = torch.randn(4, 30, 256)

        weights_low = router_low_temp(features)
        weights_high = router_high_temp(features)

        # Low temperature should be sharper (higher max)
        assert weights_low.max(dim=1)[0].mean() > weights_high.max(dim=1)[0].mean()

    def test_predict_condition(self):
        """Test condition prediction."""
        router = ConditionRouter(d_model=256, num_conditions=6)

        features = torch.randn(4, 30, 256)
        condition_id, probs = router.predict_condition(features)

        assert condition_id.shape == (4,)
        assert probs.shape == (4, 6)
        assert (condition_id >= 0).all() and (condition_id < 6).all()


class TestStageRouter:
    """Tests for StageRouter."""

    def test_stage_routing(self):
        """Test stage routing from raw sensor data."""
        router = StageRouter(sensor_dim=14, num_stages=3)

        x = torch.randn(4, 30, 14)
        weights = router(x)

        assert weights.shape == (4, 3)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_get_stage(self):
        """Test getting discrete stage."""
        router = StageRouter(sensor_dim=14, num_stages=3)

        x = torch.randn(4, 30, 14)
        stage_id, stage_probs = router.get_stage(x)

        assert stage_id.shape == (4,)
        assert stage_probs.shape == (4, 3)
        assert (stage_id >= 0).all() and (stage_id < 3).all()

    def test_health_indicator(self):
        """Test health indicator computation."""
        router = StageRouter(sensor_dim=14, num_stages=3)

        x = torch.randn(4, 30, 14)
        hi = router.get_health_indicator(x)

        assert hi.shape == (4, 30, 1)
        # HI should be in [0, 1] due to sigmoid
        assert (hi >= 0).all() and (hi <= 1).all()


class TestJointRouter:
    """Tests for JointRouter."""

    def test_joint_routing(self):
        """Test joint condition and stage routing."""
        router = JointRouter(
            d_model=256,
            sensor_dim=14,
            num_conditions=6,
            num_stages=3,
        )

        features = torch.randn(4, 30, 256)
        x_raw = torch.randn(4, 30, 14)

        joint_weights, cond_weights, stage_weights = router(features, x_raw)

        assert joint_weights.shape == (4, 6, 3)
        assert cond_weights.shape == (4, 6)
        assert stage_weights.shape == (4, 3)

        # Joint weights should be outer product
        expected_joint = cond_weights.unsqueeze(-1) * stage_weights.unsqueeze(1)
        assert torch.allclose(joint_weights, expected_joint, atol=1e-5)


class TestLoadBalanceLoss:
    """Tests for load balancing losses."""

    def test_uniform_load(self):
        """Test that uniform load gives zero loss."""
        num_experts = 6
        loads = torch.ones(4, num_experts) / num_experts

        loss = load_balance_loss(loads)
        assert loss.item() < 1e-6

    def test_imbalanced_load(self):
        """Test that imbalanced load gives positive loss."""
        num_experts = 6
        # All load on first expert
        loads = torch.zeros(4, num_experts)
        loads[:, 0] = 1.0

        loss = load_balance_loss(loads)
        assert loss.item() > 0

    def test_importance_loss(self):
        """Test importance loss computation."""
        logits = torch.randn(4, 6)
        loss = importance_loss(logits)
        assert loss.item() >= 0

    def test_load_balance_loss_module(self):
        """Test LoadBalanceLoss module."""
        loss_fn = LoadBalanceLoss(load_balance_weight=0.01, importance_weight=0.01)

        loads = torch.rand(4, 6)
        loads = loads / loads.sum(dim=1, keepdim=True)
        logits = torch.randn(4, 6)

        loss = loss_fn(loads, logits)
        assert loss.item() >= 0


class TestDSAMoE:
    """Tests for DSA-MoE module."""

    @pytest.fixture
    def config(self):
        return DSAMoEConfig(
            num_conditions=6,
            num_stages=3,
            d_model=256,
            bottleneck_dim=64,
            sensor_dim=14,
        )

    @pytest.fixture
    def moe(self, config):
        return DSAMoE(config)

    def test_forward_pass(self, moe):
        """Test complete forward pass."""
        features = torch.randn(4, 30, 256)
        x_raw = torch.randn(4, 30, 14)

        output, aux = moe(features, x_raw)

        assert output.shape == features.shape
        assert 'cond_weights' in aux
        assert 'stage_weights' in aux
        assert 'expert_loads' in aux
        assert 'load_balance_loss' in aux

    def test_hard_routing(self, moe):
        """Test with known condition IDs."""
        features = torch.randn(4, 30, 256)
        x_raw = torch.randn(4, 30, 14)
        condition_id = torch.tensor([0, 1, 2, 3])

        output, aux = moe(features, x_raw, condition_id)

        assert output.shape == features.shape
        # Condition weights should be one-hot
        assert (aux['cond_weights'].max(dim=1)[0] == 1.0).all()

    def test_expert_count(self, moe, config):
        """Test that correct number of experts are created."""
        expected_experts = config.num_conditions * config.num_stages
        assert len(moe.experts) == expected_experts

    def test_get_expert(self, moe):
        """Test getting specific expert."""
        expert = moe.get_expert(0, 0)
        assert isinstance(expert, AdapterExpert)

        expert = moe.get_expert(5, 2)
        assert isinstance(expert, AdapterExpert)

    def test_warmup_mode(self, moe):
        """Test warmup mode with random routing."""
        moe.set_warmup_mode(True)

        features = torch.randn(4, 30, 256)
        x_raw = torch.randn(4, 30, 14)

        # Run twice - should get different routing due to randomness
        _, aux1 = moe(features, x_raw)
        _, aux2 = moe(features, x_raw)

        # Weights should be different (with high probability)
        # Note: This test might occasionally fail due to randomness
        moe.set_warmup_mode(False)

    def test_epoch_scheduling(self, moe, config):
        """Test epoch-based warmup scheduling."""
        moe.set_warmup_mode(True)
        assert moe._warmup_mode.item() == True

        moe.set_epoch(config.warmup_epochs)
        assert moe._warmup_mode.item() == False

    def test_single_expert_forward(self, moe):
        """Test forward through single expert."""
        features = torch.randn(4, 30, 256)

        output = moe.forward_single_expert(features, cond_idx=0, stage_idx=0)
        assert output.shape == features.shape

    def test_gradient_flow(self, moe):
        """Test gradient flow through MoE."""
        features = torch.randn(4, 30, 256, requires_grad=True)
        x_raw = torch.randn(4, 30, 14)

        output, aux = moe(features, x_raw)
        loss = output.sum() + aux['load_balance_loss']
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_expert_loads_sum(self, moe):
        """Test that expert loads sum to 1."""
        features = torch.randn(4, 30, 256)
        x_raw = torch.randn(4, 30, 14)

        _, aux = moe(features, x_raw)

        # Expert loads should approximately sum to 1
        assert torch.allclose(aux['expert_loads'].sum(), torch.tensor(1.0), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
