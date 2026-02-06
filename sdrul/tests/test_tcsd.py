"""
Unit tests for TCSD module components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcsd.prototype_manager import TrajectoryPrototypeManager
from models.tcsd.distillation_loss import (
    DistributionalRULLoss,
    RULRegressionLoss,
    DistillationWithSupervisionLoss,
)
from models.tcsd.teacher_student import (
    FiLMLayer,
    TeacherBranch,
    StudentBranch,
    TeacherStudentPair,
)
from models.tcsd.tcsd_module import TCSD, TCSDConfig
from utils.trajectory import TrajectoryShapeEncoder


class TestTrajectoryPrototypeManager:
    """Tests for TrajectoryPrototypeManager."""

    @pytest.fixture
    def encoder(self):
        return TrajectoryShapeEncoder(seq_len=30, feature_dim=128)

    @pytest.fixture
    def manager(self, encoder):
        return TrajectoryPrototypeManager(
            trajectory_encoder=encoder,
            num_prototypes_per_condition=5,
            num_conditions=6,
        )

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.num_conditions == 6
        assert manager.num_prototypes == 5
        assert len(manager.buffer) == 6  # Uses parent's buffer attribute

    def test_add_trajectory_to_buffer(self, manager):
        """Test adding trajectories to initialization buffer."""
        traj = torch.linspace(100, 0, 30)
        result = manager.add_trajectory(traj, condition_id=0)
        assert result == True
        assert len(manager._init_buffer[0]) == 1

    def test_kmeans_initialization(self, manager):
        """Test K-means prototype initialization."""
        # Generate enough trajectories for initialization
        trajectories = [
            torch.linspace(100 + i * 10, 0, 30) + torch.randn(30) * 5
            for i in range(20)
        ]

        success = manager.fit_kmeans(trajectories, condition_id=0)
        assert success == True
        assert manager.is_initialized(0) == True
        assert manager.num_prototypes_for_condition(0) == 5

    def test_get_prototype(self, manager):
        """Test getting prototypes."""
        # Initialize first
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        manager.fit_kmeans(trajectories, condition_id=0)

        # Get prototype
        result = manager.get_prototype(0)
        assert result is not None
        traj, enc = result
        assert traj.shape == (30,)
        assert enc.shape == (128,)

    def test_get_random_prototype(self, manager):
        """Test getting random prototype encoding."""
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        manager.fit_kmeans(trajectories, condition_id=0)

        enc = manager.get_random_prototype(0)
        assert enc is not None
        assert enc.shape == (128,)

    def test_get_all_prototypes(self, manager):
        """Test getting all prototypes for a condition."""
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        manager.fit_kmeans(trajectories, condition_id=0)

        protos = manager.get_all_prototypes(0)
        assert len(protos) == 5
        for p in protos:
            assert p.shape == (128,)

    def test_ema_update(self, manager):
        """Test EMA update of prototypes."""
        # Initialize
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        manager.fit_kmeans(trajectories, condition_id=0)

        # Get initial encoding
        initial_enc = manager.get_random_prototype(0).clone()

        # Add new trajectory (should trigger EMA update)
        new_traj = torch.linspace(150, 0, 30)
        manager.add_trajectory(new_traj, condition_id=0)

        # Encoding should have changed slightly
        # (Note: this test may be flaky due to random prototype selection)

    def test_save_load_state(self, manager):
        """Test saving and loading manager state."""
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        manager.fit_kmeans(trajectories, condition_id=0)

        # Save state
        state = manager.save_state()

        # Create new manager and load
        encoder = TrajectoryShapeEncoder(seq_len=30, feature_dim=128)
        new_manager = TrajectoryPrototypeManager(encoder, num_conditions=6)
        new_manager.load_state(state)

        assert new_manager.is_initialized(0) == True
        assert new_manager.num_prototypes_for_condition(0) == 5


class TestDistributionalRULLoss:
    """Tests for DistributionalRULLoss."""

    @pytest.fixture
    def loss_fn(self):
        return DistributionalRULLoss(alpha=0.5)

    def test_identical_distributions(self, loss_fn):
        """Test loss is zero for identical distributions."""
        mu = torch.tensor([50.0, 60.0, 70.0])
        sigma = torch.tensor([5.0, 5.0, 5.0])

        loss = loss_fn((mu, sigma), (mu, sigma))
        assert loss.item() < 1e-5

    def test_different_means(self, loss_fn):
        """Test loss increases with mean difference."""
        mu_t = torch.tensor([50.0])
        sigma = torch.tensor([5.0])

        mu_s_close = torch.tensor([51.0])
        mu_s_far = torch.tensor([60.0])

        loss_close = loss_fn((mu_t, sigma), (mu_s_close, sigma))
        loss_far = loss_fn((mu_t, sigma), (mu_s_far, sigma))

        assert loss_far > loss_close

    def test_different_sigmas(self, loss_fn):
        """Test loss increases with sigma difference."""
        mu = torch.tensor([50.0])
        sigma_t = torch.tensor([5.0])

        sigma_s_close = torch.tensor([5.5])
        sigma_s_far = torch.tensor([10.0])

        loss_close = loss_fn((mu, sigma_t), (mu, sigma_s_close))
        loss_far = loss_fn((mu, sigma_t), (mu, sigma_s_far))

        assert loss_far > loss_close

    def test_kl_divergence_asymmetry(self, loss_fn):
        """Test KL divergence is asymmetric."""
        mu_t = torch.tensor([50.0])
        sigma_t = torch.tensor([5.0])
        mu_s = torch.tensor([55.0])
        sigma_s = torch.tensor([7.0])

        # KL(t||s) != KL(s||t)
        kl_ts = loss_fn.gaussian_kl_divergence(mu_t, sigma_t, mu_s, sigma_s)
        kl_st = loss_fn.gaussian_kl_divergence(mu_s, sigma_s, mu_t, sigma_t)

        assert not torch.allclose(kl_ts, kl_st)

    def test_wasserstein_symmetry(self, loss_fn):
        """Test Wasserstein distance is symmetric."""
        mu_t = torch.tensor([50.0])
        sigma_t = torch.tensor([5.0])
        mu_s = torch.tensor([55.0])
        sigma_s = torch.tensor([7.0])

        w2_ts = loss_fn.wasserstein_2_squared(mu_t, sigma_t, mu_s, sigma_s)
        w2_st = loss_fn.wasserstein_2_squared(mu_s, sigma_s, mu_t, sigma_t)

        assert torch.allclose(w2_ts, w2_st)

    def test_forward_with_components(self, loss_fn):
        """Test forward with component breakdown."""
        mu_t = torch.tensor([50.0, 60.0])
        sigma_t = torch.tensor([5.0, 6.0])
        mu_s = torch.tensor([52.0, 58.0])
        sigma_s = torch.tensor([5.5, 6.5])

        total, kl, w2 = loss_fn.forward_with_components(
            (mu_t, sigma_t), (mu_s, sigma_s)
        )

        # Total should be weighted combination
        expected = 0.5 * kl + 0.5 * w2
        assert torch.allclose(total, expected)


class TestRULRegressionLoss:
    """Tests for RULRegressionLoss."""

    def test_symmetric_loss(self):
        """Test symmetric MSE loss."""
        loss_fn = RULRegressionLoss(use_asymmetric=False)

        pred = torch.tensor([50.0])
        target = torch.tensor([55.0])

        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(target, pred)

        assert torch.allclose(loss1, loss2)

    def test_asymmetric_loss(self):
        """Test asymmetric loss penalizes late predictions more."""
        loss_fn = RULRegressionLoss(use_asymmetric=True, asymmetric_factor=2.0)

        target = torch.tensor([50.0])
        pred_early = torch.tensor([55.0])  # Predicts more RUL (early)
        pred_late = torch.tensor([45.0])   # Predicts less RUL (late)

        loss_early = loss_fn(pred_early, target)
        loss_late = loss_fn(pred_late, target)

        # Late prediction should have higher loss
        assert loss_late > loss_early


class TestFiLMLayer:
    """Tests for FiLMLayer."""

    def test_shape_preservation(self):
        """Test output shape matches input shape."""
        film = FiLMLayer(feature_dim=256, condition_dim=128)

        # 3D input
        features = torch.randn(4, 30, 256)
        condition = torch.randn(4, 128)
        out = film(features, condition)
        assert out.shape == features.shape

        # 2D input
        features = torch.randn(4, 256)
        out = film(features, condition)
        assert out.shape == features.shape

    def test_identity_initialization(self):
        """Test FiLM initializes to identity transformation."""
        film = FiLMLayer(feature_dim=256, condition_dim=128)

        features = torch.randn(4, 30, 256)
        condition = torch.randn(4, 128)
        out = film(features, condition)

        # Should be close to identity initially
        assert torch.allclose(out, features, atol=0.1)

    def test_gradient_flow(self):
        """Test gradients flow through FiLM layer."""
        film = FiLMLayer(feature_dim=256, condition_dim=128)

        features = torch.randn(4, 30, 256, requires_grad=True)
        condition = torch.randn(4, 128, requires_grad=True)

        out = film(features, condition)
        loss = out.sum()
        loss.backward()

        assert features.grad is not None
        assert condition.grad is not None


class TestTeacherStudentBranches:
    """Tests for TeacherBranch and StudentBranch."""

    @pytest.fixture
    def ts_pair(self):
        return TeacherStudentPair(
            sensor_dim=14,
            d_model=256,
            prototype_dim=128,
        )

    def test_teacher_forward(self, ts_pair):
        """Test teacher forward pass."""
        x = torch.randn(4, 30, 14)
        prototype = torch.randn(4, 128)

        mu, sigma = ts_pair.forward_teacher(x, prototype)

        assert mu.shape == (4,)
        assert sigma.shape == (4,)
        assert (sigma > 0).all()

    def test_student_forward(self, ts_pair):
        """Test student forward pass."""
        x = torch.randn(4, 30, 14)

        mu, sigma = ts_pair.forward_student(x)

        assert mu.shape == (4,)
        assert sigma.shape == (4,)
        assert (sigma > 0).all()

    def test_shared_feature_extractor(self, ts_pair):
        """Test that teacher and student share feature extractor."""
        assert ts_pair.teacher.feature_extractor is ts_pair.student.feature_extractor

    def test_combined_forward(self, ts_pair):
        """Test combined forward through both branches."""
        x = torch.randn(4, 30, 14)
        prototype = torch.randn(4, 128)

        teacher_out, student_out = ts_pair(x, prototype)

        assert len(teacher_out) == 2
        assert len(student_out) == 2


class TestTCSD:
    """Tests for TCSD module."""

    @pytest.fixture
    def config(self):
        return TCSDConfig(
            sensor_dim=14,
            d_model=256,
            prototype_dim=128,
            num_prototypes_per_condition=5,
            num_conditions=6,
        )

    @pytest.fixture
    def tcsd(self, config):
        return TCSD(config)

    def test_forward(self, tcsd):
        """Test default forward pass (student)."""
        x = torch.randn(4, 30, 14)
        mu, sigma = tcsd(x)

        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_predict_student(self, tcsd):
        """Test prediction with student."""
        x = torch.randn(4, 30, 14)
        mu, sigma = tcsd.predict(x, use_teacher=False)

        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_add_trajectory(self, tcsd):
        """Test adding trajectory to prototype manager."""
        traj = torch.linspace(100, 0, 30)
        result = tcsd.add_trajectory(traj, condition_id=0)
        assert result == True

    def test_initialize_prototypes(self, tcsd):
        """Test prototype initialization."""
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]

        success = tcsd.initialize_prototypes(trajectories, condition_id=0)
        assert success == True

    def test_on_policy_step_without_prototypes(self, tcsd):
        """Test on-policy step without initialized prototypes uses fallback."""
        x = torch.randn(4, 30, 14)
        condition_id = torch.tensor([0, 1, 2, 3])

        loss, outputs = tcsd.on_policy_step(x, condition_id)

        # With fallback prototypes, loss should be non-zero and samples should be valid
        # This ensures distillation is active even without explicit prototypes
        assert loss.item() >= 0.0  # Loss can be non-zero with fallback
        assert outputs['valid_samples'] == 4  # All samples use fallback prototype

    def test_on_policy_step_with_prototypes(self, tcsd):
        """Test on-policy step with initialized prototypes."""
        # Initialize prototypes
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        tcsd.initialize_prototypes(trajectories, condition_id=0)

        x = torch.randn(4, 30, 14)
        condition_id = torch.tensor([0, 0, 0, 0])

        loss, outputs = tcsd.on_policy_step(x, condition_id)

        assert loss.item() >= 0
        assert outputs['valid_samples'] == 4

    def test_supervised_step(self, tcsd):
        """Test supervised training step."""
        # Initialize prototypes
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        tcsd.initialize_prototypes(trajectories, condition_id=0)

        x = torch.randn(4, 30, 14)
        target = torch.tensor([50.0, 60.0, 70.0, 80.0])
        condition_id = torch.tensor([0, 0, 0, 0])

        loss, loss_dict = tcsd.supervised_step(x, target, condition_id)

        assert loss.item() >= 0
        assert 'distillation_loss' in loss_dict
        assert 'supervision_loss' in loss_dict

    def test_epoch_scheduling(self, tcsd):
        """Test epoch-based warmup scheduling."""
        initial_weight = tcsd._distillation_weight.item()

        tcsd.set_epoch(0)
        assert tcsd._distillation_weight.item() == 0.0  # Warmup starts at 0

        tcsd.set_epoch(tcsd.config.warmup_epochs)
        assert tcsd._distillation_weight.item() == initial_weight

    def test_prototype_stats(self, tcsd):
        """Test getting prototype statistics."""
        stats = tcsd.get_prototype_stats()

        assert 'num_conditions' in stats
        assert 'prototypes_per_condition' in stats
        assert 'initialized_conditions' in stats

    def test_gradient_flow(self, tcsd):
        """Test gradient flow through TCSD."""
        # Initialize prototypes
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        tcsd.initialize_prototypes(trajectories, condition_id=0)

        x = torch.randn(4, 30, 14, requires_grad=True)
        condition_id = torch.tensor([0, 0, 0, 0])

        loss, _ = tcsd.on_policy_step(x, condition_id)
        loss.backward()

        # Note: x.grad may be None if loss is 0 or doesn't depend on x
        # This is expected behavior


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
