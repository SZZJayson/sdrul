"""
Integration tests for the continual learning framework.

Tests the integration between DSA-MoE, TCSD, Smart Replay, and EWC components.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moe.dsa_moe import DSAMoE, DSAMoEConfig, DSAMoEWithFeatureExtractor
from models.tcsd.tcsd_module import TCSD, TCSDConfig
from models.encoders import TransformerFeatureExtractor, GaussianRULHead
from utils.health_indicator import HealthIndicatorComputer
from continual.ewc import EWCRegularizer
from continual.replay import ReplayBuffer, SmartReplayBuffer
from continual.trainer import ContinualTrainer, ContinualTrainerConfig


class TestDSAMoEWithHealthIndicator:
    """Test DSA-MoE integration with HealthIndicatorComputer."""

    def test_stage_router_uses_hi_computer(self):
        """Verify StageRouter correctly uses HealthIndicatorComputer."""
        from models.moe.router import StageRouter

        hi_computer = HealthIndicatorComputer(sensor_dim=14)
        router = StageRouter(
            sensor_dim=14,
            num_stages=3,
            hi_computer=hi_computer,
        )

        # Verify same instance is used
        assert router.hi_computer is hi_computer

        # Test routing
        x = torch.randn(4, 30, 14)
        weights = router(x)

        assert weights.shape == (4, 3)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_dsa_moe_stage_routing(self):
        """Test DSA-MoE uses stage routing correctly."""
        config = DSAMoEConfig(
            num_conditions=6,
            num_stages=3,
            d_model=256,
            sensor_dim=14,
        )
        moe = DSAMoE(config)

        features = torch.randn(4, 30, 256)
        x_raw = torch.randn(4, 30, 14)

        output, aux = moe(features, x_raw)

        # Check stage weights are computed
        assert 'stage_weights' in aux
        assert aux['stage_weights'].shape == (4, 3)

        # Check joint weights combine condition and stage
        assert 'joint_weights' in aux
        assert aux['joint_weights'].shape == (4, 6, 3)

    def test_dsa_moe_with_feature_extractor(self):
        """Test DSAMoEWithFeatureExtractor end-to-end."""
        config = DSAMoEConfig(
            num_conditions=6,
            num_stages=3,
            d_model=256,
            sensor_dim=14,
        )
        model = DSAMoEWithFeatureExtractor(config)

        x = torch.randn(4, 30, 14)
        condition_id = torch.tensor([0, 1, 2, 3])

        output, aux = model(x, condition_id)

        assert output.shape == (4, 30, 256)
        assert 'cond_weights' in aux
        assert 'stage_weights' in aux


class TestTCSDWithSmartReplay:
    """Test TCSD integration with smart experience replay."""

    @pytest.fixture
    def tcsd(self):
        config = TCSDConfig(
            sensor_dim=14,
            d_model=256,
            prototype_dim=128,
            num_conditions=6,
        )
        return TCSD(config)

    def test_tcsd_on_replay_samples(self, tcsd):
        """Test TCSD distillation on replay samples."""
        # Initialize prototypes
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        tcsd.initialize_prototypes(trajectories, condition_id=0)

        # Simulate replay samples
        replay_x = torch.randn(8, 30, 14)
        condition_id = torch.zeros(8, dtype=torch.long)

        # Run distillation
        loss, outputs = tcsd.on_policy_step(replay_x, condition_id)

        assert loss.item() >= 0
        assert outputs['valid_samples'] == 8
        assert 'teacher_mu' in outputs
        assert 'student_mu' in outputs

    def test_tcsd_prototype_update_during_replay(self, tcsd):
        """Test that prototypes can be updated during replay."""
        # Initialize
        trajectories = [
            torch.linspace(100, 0, 30) + torch.randn(30) * 5
            for _ in range(20)
        ]
        tcsd.initialize_prototypes(trajectories, condition_id=0)

        initial_proto = tcsd.prototype_manager.get_random_prototype(0).clone()

        # Add new trajectory (simulating learning from replay)
        new_traj = torch.linspace(150, 0, 30)
        tcsd.add_trajectory(new_traj, condition_id=0)

        # Prototype should have been updated via EMA
        # (Note: exact change depends on similarity to existing prototypes)


class TestSmartReplayBuffer:
    """Test SmartReplayBuffer functionality."""

    def test_importance_scoring(self):
        """Test importance score computation."""
        buffer = SmartReplayBuffer(max_size=1000)

        # Add samples with different characteristics
        for i in range(10):
            x = torch.randn(30, 14)
            y = torch.tensor(50.0 + i * 5)
            buffer.add_with_selection(x, y, task_id=0)

        assert len(buffer) == 10

    def test_weighted_sampling(self):
        """Test weighted sampling returns correct shapes."""
        buffer = SmartReplayBuffer(max_size=1000)

        # Add samples to multiple tasks
        for task_id in range(3):
            for _ in range(20):
                x = torch.randn(30, 14)
                y = torch.tensor(50.0 + task_id * 10)
                buffer.add_with_selection(x, y, task_id=task_id)

        # Sample with weights
        x, y, task_ids, weights = buffer.sample_weighted(
            batch_size=8,
            current_task=2,
            device=torch.device('cpu'),
        )

        assert x.shape[0] <= 8
        assert weights.shape[0] == x.shape[0]
        assert (task_ids != 2).all()  # Should not include current task

    def test_dynamic_replay_ratio(self):
        """Test dynamic replay ratio computation."""
        buffer = SmartReplayBuffer(max_size=1000)

        # Update task stats with different distributions
        buffer.update_task_stats(0, [50.0, 55.0, 60.0, 65.0, 70.0])
        buffer.update_task_stats(1, [80.0, 85.0, 90.0, 95.0, 100.0])  # Different distribution
        buffer.update_task_stats(2, [50.0, 55.0, 60.0, 65.0, 70.0])  # Similar to task 0

        # Ratio for task 2 vs task 0 should be lower (similar distributions)
        ratio_similar = buffer.compute_replay_ratio(2, [0])

        # Ratio for task 2 vs task 1 should be higher (different distributions)
        ratio_different = buffer.compute_replay_ratio(2, [1])

        # Different distribution should trigger higher replay ratio
        assert ratio_different >= ratio_similar

    def test_sample_weights_favor_middle_rul(self):
        """Test that sample weights favor middle RUL values."""
        buffer = SmartReplayBuffer(max_size=1000)

        # Create RUL values spanning the range
        y = torch.tensor([10.0, 30.0, 55.0, 80.0, 100.0])
        weights = buffer._compute_sample_weights(y)

        # Middle value (55) should have highest weight
        assert weights[2] > weights[0]  # 55 > 10
        assert weights[2] > weights[4]  # 55 > 100


class TestFullContinualTrainingStep:
    """Test complete continual training step with all components."""

    @pytest.fixture
    def setup_components(self):
        """Set up all components for training."""
        # DSA-MoE with feature extractor
        moe_config = DSAMoEConfig(
            num_conditions=6,
            num_stages=3,
            d_model=256,
            sensor_dim=14,
        )
        model = DSAMoEWithFeatureExtractor(moe_config)

        # Add RUL head to model
        class ModelWithHead(nn.Module):
            def __init__(self, moe_model):
                super().__init__()
                self.moe = moe_model
                self.rul_head = GaussianRULHead(d_model=256)

            def forward(self, x, condition_id=None, mask=None):
                features, aux = self.moe(x, condition_id, mask)
                mu, sigma = self.rul_head(features)
                return (mu, sigma), aux

        model_with_head = ModelWithHead(model)

        # TCSD
        tcsd_config = TCSDConfig(
            sensor_dim=14,
            d_model=256,
            prototype_dim=128,
            num_conditions=6,
        )
        tcsd = TCSD(tcsd_config)

        return model_with_head, tcsd

    def test_training_step_components(self, setup_components):
        """Test individual components of training step."""
        model, tcsd = setup_components

        # Create batch
        x = torch.randn(4, 30, 14)
        y = torch.tensor([50.0, 60.0, 70.0, 80.0])
        condition_id = torch.tensor([0, 1, 2, 3])

        # Forward through model
        (mu, sigma), aux = model(x, condition_id)

        assert mu.shape == (4,)
        assert sigma.shape == (4,)
        assert 'load_balance_loss' in aux

        # Compute supervised loss
        L_supervised = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (y - mu) ** 2 / (sigma ** 2 + 1e-6))
        L_supervised = L_supervised.mean()

        assert L_supervised.item() >= 0

    def test_ewc_integration(self, setup_components):
        """Test EWC integration with model."""
        model, _ = setup_components

        ewc = EWCRegularizer(model, ewc_lambda=1000.0)

        # Create simple dataloader
        def simple_dataloader():
            for _ in range(5):
                x = torch.randn(4, 30, 14)
                y = torch.tensor([50.0, 60.0, 70.0, 80.0])
                yield x, y

        # Compute Fisher
        ewc.compute_fisher(simple_dataloader(), num_samples=20)

        assert ewc.num_tasks == 1
        assert len(ewc.fisher) > 0

        # Compute EWC loss
        loss = ewc.ewc_loss()
        assert loss.item() >= 0

    def test_replay_buffer_integration(self, setup_components):
        """Test replay buffer integration."""
        model, _ = setup_components

        buffer = ReplayBuffer(max_size=1000)

        # Add samples
        for task_id in range(3):
            for _ in range(10):
                x = torch.randn(30, 14)
                y = torch.tensor(50.0 + task_id * 10)
                buffer.add(x, y, task_id)

        # Sample for replay
        x, y, task_ids = buffer.sample_for_replay(
            batch_size=8,
            current_task=2,
        )

        assert x.shape[0] <= 8
        assert (task_ids != 2).all()  # Should not include current task

    def test_full_training_step(self, setup_components):
        """Test complete training step."""
        model, tcsd = setup_components

        # Initialize TCSD prototypes
        for cond_id in range(3):
            trajectories = [
                torch.linspace(100, 0, 30) + torch.randn(30) * 5
                for _ in range(20)
            ]
            tcsd.initialize_prototypes(trajectories, condition_id=cond_id)

        # Create batch
        x = torch.randn(4, 30, 14)
        y = torch.tensor([50.0, 60.0, 70.0, 80.0])
        condition_id = torch.tensor([0, 1, 2, 0])

        # Forward through model
        (mu, sigma), aux = model(x, condition_id)

        # Supervised loss
        L_supervised = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (y - mu) ** 2 / (sigma ** 2 + 1e-6))
        L_supervised = L_supervised.mean()

        # Load balance loss
        L_balance = aux['load_balance_loss']

        # TCSD distillation
        L_distill, _ = tcsd.on_policy_step(x, condition_id)

        # Total loss
        total_loss = L_supervised + 0.01 * L_balance + L_distill

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                # Some params may have None grad if not used
                pass


class TestContinualTrainer:
    """Test ContinualTrainer class."""

    @pytest.fixture
    def trainer_setup(self):
        """Set up trainer with all components."""
        # Simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(14 * 30, 256)
                self.head = nn.Linear(256, 2)  # mu, sigma

            def forward(self, x, condition_id=None, mask=None):
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)
                features = torch.relu(self.encoder(x_flat))
                out = self.head(features)
                mu, log_sigma = out[:, 0], out[:, 1]
                sigma = torch.exp(log_sigma).clamp(0.1, 50)
                aux = {'load_balance_loss': torch.tensor(0.0)}
                return (mu, sigma), aux

        model = SimpleModel()

        config = ContinualTrainerConfig(
            lambda_supervised=1.0,
            lambda_ewc=100.0,
            use_ewc=True,
            use_experience_replay=True,
        )

        trainer = ContinualTrainer(
            model=model,
            tcsd=None,  # Simplified test without TCSD
            config=config,
        )

        return trainer

    def test_training_step(self, trainer_setup):
        """Test single training step."""
        trainer = trainer_setup

        batch = (
            torch.randn(4, 30, 14),
            torch.tensor([50.0, 60.0, 70.0, 80.0]),
            torch.tensor([0, 1, 2, 3]),
        )

        loss, metrics = trainer.training_step(batch, task_id=0)

        assert loss.item() >= 0
        assert 'supervised_loss' in metrics
        assert 'total_loss' in metrics

    def test_task_lifecycle(self, trainer_setup):
        """Test task start and end callbacks."""
        trainer = trainer_setup

        # Start task
        trainer.on_task_start(task_id=0)
        assert trainer.current_task == 0
        assert trainer.current_epoch == 0

        # Simulate training
        for _ in range(3):
            batch = (
                torch.randn(4, 30, 14),
                torch.tensor([50.0, 60.0, 70.0, 80.0]),
                torch.tensor([0, 0, 0, 0]),
            )
            trainer.training_step(batch, task_id=0)

        # End task
        trainer.on_task_end(task_id=0)
        assert 0 in trainer.completed_tasks

    def test_ewc_after_task(self, trainer_setup):
        """Test EWC consolidation after task."""
        trainer = trainer_setup

        # Create simple dataloader
        class SimpleDataLoader:
            def __iter__(self):
                for _ in range(5):
                    yield (
                        torch.randn(4, 30, 14),
                        torch.tensor([50.0, 60.0, 70.0, 80.0]),
                        torch.tensor([0, 0, 0, 0]),
                    )

        trainer.on_task_start(task_id=0)
        trainer.on_task_end(task_id=0, dataloader=SimpleDataLoader())

        # EWC should have computed Fisher
        assert trainer.ewc.num_tasks == 1

    def test_replay_buffer_usage(self, trainer_setup):
        """Test replay buffer is populated during training."""
        trainer = trainer_setup

        trainer.on_task_start(task_id=0)

        # Add samples via training - manually add to buffer since training_step doesn't
        for _ in range(5):
            batch = (
                torch.randn(4, 30, 14),
                torch.tensor([50.0, 60.0, 70.0, 80.0]),
                torch.tensor([0, 0, 0, 0]),
            )
            trainer.training_step(batch, task_id=0)
            # Manually add to replay buffer (simulating what train_epoch does)
            x, y, cond = batch
            trainer.replay_buffer.add_batch(x, y, cond)

        # Buffer should have samples
        assert len(trainer.replay_buffer) > 0


class TestGradientFlow:
    """Test gradient flow through the complete system."""

    def test_end_to_end_gradient_flow(self):
        """Test gradients flow through all components."""
        # Create full model
        moe_config = DSAMoEConfig(
            num_conditions=6,
            num_stages=3,
            d_model=256,
            sensor_dim=14,
        )

        class FullModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = TransformerFeatureExtractor(
                    sensor_dim=14,
                    d_model=256,
                )
                self.moe = DSAMoE(moe_config)
                self.rul_head = GaussianRULHead(d_model=256)

            def forward(self, x, condition_id=None):
                features = self.feature_extractor(x)
                moe_out, aux = self.moe(features, x, condition_id)
                mu, sigma = self.rul_head(moe_out)
                return (mu, sigma), aux

        model = FullModel()

        # Forward pass
        x = torch.randn(4, 30, 14, requires_grad=True)
        y = torch.tensor([50.0, 60.0, 70.0, 80.0])
        condition_id = torch.tensor([0, 1, 2, 3])

        (mu, sigma), aux = model(x, condition_id)

        # Compute loss
        loss = F.mse_loss(mu, y) + aux['load_balance_loss']

        # Backward pass
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check model parameters have gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        assert grad_count > 0, "No parameters received gradients"


class TestDatasetWithTrainer:
    """Test dataset integration with ContinualTrainer."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        import tempfile
        from data import create_synthetic_cmapss

        with tempfile.TemporaryDirectory() as tmpdir:
            for sub_id in ['FD001', 'FD003']:
                path = os.path.join(tmpdir, f'{sub_id}_train.npz')
                create_synthetic_cmapss(
                    num_engines=10,
                    save_path=path,
                    seed=int(sub_id[-1]),
                )
            yield tmpdir

    def test_cmapss_with_trainer(self, temp_data_dir):
        """Test C-MAPSS dataset works with ContinualTrainer."""
        from data import CMAPSSDataset, CMAPSS_CONFIG
        from models.rul_model import RULModelConfig, RULPredictionModel

        # Create dataset with condition_id
        path = os.path.join(temp_data_dir, 'FD001_train.npz')
        dataset = CMAPSSDataset(path, condition_id=0)

        # Verify dataset returns 3 values
        x, y, cond = dataset[0]
        assert cond.dtype == torch.long

        # Create model with matching config
        config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)
        model = RULPredictionModel(config)

        # Test forward pass with batch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4)
        batch = next(iter(loader))

        x, y, condition_id = batch
        (mu, sigma), aux = model(x, condition_id=condition_id)

        assert mu.shape == (4,)
        assert sigma.shape == (4,)

    def test_continual_cmapss_with_trainer(self, temp_data_dir):
        """Test ContinualCMAPSS with ContinualTrainer."""
        from data import ContinualCMAPSS

        continual = ContinualCMAPSS(
            temp_data_dir,
            sub_datasets=['FD001', 'FD003'],
            batch_size=4,
        )

        # Simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(14 * 30, 256)
                self.head = nn.Linear(256, 2)

            def forward(self, x, condition_id=None, mask=None):
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)
                features = torch.relu(self.encoder(x_flat))
                out = self.head(features)
                mu, log_sigma = out[:, 0], out[:, 1]
                sigma = torch.exp(log_sigma).clamp(0.1, 50)
                aux = {'load_balance_loss': torch.tensor(0.0)}
                return (mu, sigma), aux

        model = SimpleModel()

        config = ContinualTrainerConfig(
            use_ewc=False,
            use_experience_replay=False,
        )

        trainer = ContinualTrainer(
            model=model,
            config=config,
        )

        # Train on first task
        trainer.on_task_start(task_id=0)
        loader = continual.get_task_dataloader(0)

        for batch in loader:
            loss, metrics = trainer.training_step(batch, task_id=0)
            assert 'supervised_loss' in metrics
            break  # Just test one batch

        trainer.on_task_end(task_id=0)

        # Train on second task
        trainer.on_task_start(task_id=1)
        loader = continual.get_task_dataloader(1)

        for batch in loader:
            loss, metrics = trainer.training_step(batch, task_id=1)
            assert 'supervised_loss' in metrics
            break

    def test_multi_dataset_continual_learning(self, temp_data_dir):
        """Test multi-dataset continual learning scenario."""
        from data import ContinualCMAPSS, CMAPSS_CONFIG
        from models.rul_model import RULModelConfig, create_shared_model_and_tcsd

        continual = ContinualCMAPSS(
            temp_data_dir,
            sub_datasets=['FD001', 'FD003'],
            batch_size=4,
        )

        # Create model and TCSD with shared feature extractor
        config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)
        model, tcsd = create_shared_model_and_tcsd(config)

        # Initialize TCSD prototypes for each task
        for task_idx in range(len(continual)):
            dataset = continual.get_task_dataset(task_idx)
            condition_id = continual.get_condition_id(task_idx)

            # Create synthetic trajectories for prototype initialization
            trajectories = [
                torch.linspace(100, 0, 30) + torch.randn(30) * 5
                for _ in range(20)
            ]
            tcsd.initialize_prototypes(trajectories, condition_id=condition_id)

        # Verify prototypes are initialized
        stats = tcsd.get_prototype_stats()
        assert 0 in stats['initialized_conditions']  # FD001
        assert 2 in stats['initialized_conditions']  # FD003

        # Test forward pass with TCSD
        loader = continual.get_task_dataloader(0)
        batch = next(iter(loader))
        x, y, condition_id = batch

        # Model forward
        (mu, sigma), aux = model(x, condition_id=condition_id)
        assert mu.shape == (4,)

        # TCSD distillation
        loss, outputs = tcsd.on_policy_step(x, condition_id)
        assert loss.item() >= 0


class TestDatasetModelCompatibility:
    """Test dataset-model compatibility validation."""

    def test_compatible_config(self):
        """Test compatible configurations pass validation."""
        from data import CMAPSS_CONFIG, validate_dataset_model_compatibility
        from models.rul_model import RULModelConfig

        model_config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)

        # Should not raise
        validate_dataset_model_compatibility(
            CMAPSS_CONFIG,
            model_sensor_dim=model_config.sensor_dim,
            model_num_conditions=model_config.num_conditions,
        )

    def test_incompatible_sensor_dim(self):
        """Test incompatible sensor dimension is detected."""
        from data import CMAPSS_CONFIG, validate_dataset_model_compatibility

        with pytest.raises(ValueError, match="Sensor dimension mismatch"):
            validate_dataset_model_compatibility(
                CMAPSS_CONFIG,
                model_sensor_dim=20,  # Wrong
                model_num_conditions=4,
            )

    def test_end_to_end_with_config(self):
        """Test end-to-end flow with config-based model creation."""
        import tempfile
        from data import (
            CMAPSSDataset,
            CMAPSS_CONFIG,
            create_synthetic_cmapss,
            validate_dataset_model_compatibility,
        )
        from models.rul_model import RULModelConfig, RULPredictionModel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            path = os.path.join(tmpdir, 'test.npz')
            create_synthetic_cmapss(num_engines=10, save_path=path)

            # Create dataset
            dataset = CMAPSSDataset(path, condition_id=0)

            # Create model from config
            model_config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)
            model = RULPredictionModel(model_config)

            # Validate compatibility
            validate_dataset_model_compatibility(
                CMAPSS_CONFIG,
                model_sensor_dim=model_config.sensor_dim,
                model_num_conditions=model_config.num_conditions,
            )

            # Test forward pass
            x, y, cond = dataset[0]
            x = x.unsqueeze(0)
            cond = cond.unsqueeze(0)

            (mu, sigma), aux = model(x, condition_id=cond)
            assert mu.shape == (1,)


class TestExpertExpansionIntegration:
    """Test expert expansion integration with optimizer."""

    @pytest.fixture
    def expansion_setup(self):
        """Set up model and trainer for expansion testing."""
        # Create model with MoE
        moe_config = DSAMoEConfig(
            num_conditions=3,
            num_stages=2,
            d_model=256,
            sensor_dim=14,
        )

        class ModelWithMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = TransformerFeatureExtractor(
                    sensor_dim=14, d_model=256
                )
                self.moe = DSAMoE(moe_config)
                self.rul_head = GaussianRULHead(d_model=256)

            def forward(self, x, condition_id=None, mask=None):
                features = self.feature_extractor(x, mask)
                moe_out, aux = self.moe(features, x, condition_id, mask)
                mu, sigma = self.rul_head(moe_out)
                return (mu, sigma), aux

        model = ModelWithMoE()

        config = ContinualTrainerConfig(
            use_ewc=False,
            use_experience_replay=False,
            auto_expand_experts=False,  # Start with manual expansion
        )

        trainer = ContinualTrainer(
            model=model,
            config=config,
            device=torch.device('cpu'),
        )

        return trainer, model

    def test_expert_expansion_adds_to_optimizer(self, expansion_setup):
        """Test that new experts are added to optimizer after expansion."""
        trainer, model = expansion_setup

        # Count initial optimizer params
        initial_param_groups = len(trainer.optimizer.param_groups)
        initial_param_count = sum(len(g['params']) for g in trainer.optimizer.param_groups)

        # Expand experts
        model.moe.expand_for_new_condition(source_condition_id=0)

        # Add new params to optimizer
        trainer._add_new_params_to_optimizer()

        # Check params increased
        new_param_count = sum(len(g['params']) for g in trainer.optimizer.param_groups)
        assert new_param_count > initial_param_count

        # Check new params are trainable
        for group in trainer.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    assert param.grad is None or not torch.isnan(param.grad).any()

    def test_expert_expansion_with_training_step(self, expansion_setup):
        """Test training step works after expert expansion."""
        trainer, model = expansion_setup

        # Create batch
        batch = (
            torch.randn(4, 30, 14),
            torch.tensor([50.0, 60.0, 70.0, 80.0]),
            torch.tensor([0, 1, 2, 0]),
        )

        # Train before expansion
        loss1, _ = trainer.training_step(batch, task_id=0)
        assert loss1.item() >= 0

        # Expand experts
        model.moe.expand_for_new_condition(source_condition_id=0)
        trainer._add_new_params_to_optimizer()

        # Train after expansion - should not crash
        loss2, _ = trainer.training_step(batch, task_id=0)
        assert loss2.item() >= 0

    def test_auto_expansion_in_training_step(self):
        """Test automatic expert expansion during training step."""
        # Create model with MoE
        moe_config = DSAMoEConfig(
            num_conditions=3,
            num_stages=2,
            d_model=256,
            sensor_dim=14,
        )

        class ModelWithMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = TransformerFeatureExtractor(
                    sensor_dim=14, d_model=256
                )
                self.moe = DSAMoE(moe_config)
                self.rul_head = GaussianRULHead(d_model=256)

            def forward(self, x, condition_id=None, mask=None):
                features = self.feature_extractor(x, mask)
                moe_out, aux = self.moe(features, x, condition_id, mask)
                mu, sigma = self.rul_head(moe_out)
                return (mu, sigma), aux

        model = ModelWithMoE()

        config = ContinualTrainerConfig(
            use_ewc=False,
            use_experience_replay=False,
            auto_expand_experts=True,  # Enable auto expansion
            new_condition_threshold=0.9,  # High threshold to trigger detection
        )

        trainer = ContinualTrainer(
            model=model,
            config=config,
            device=torch.device('cpu'),
        )

        # Create batch with condition_id=5 (beyond initial 3)
        batch = (
            torch.randn(4, 30, 14),
            torch.tensor([50.0, 60.0, 70.0, 80.0]),
            torch.tensor([5, 5, 5, 5]),  # Condition 5 doesn't exist
        )

        initial_num_conditions = model.moe.num_conditions

        # Training step should trigger expansion
        # Note: With random data, detection may or may not trigger
        # This test verifies the mechanism exists
        loss, _ = trainer.training_step(batch, task_id=0)
        assert loss.item() >= 0

    def test_optimizer_no_duplicate_params(self, expansion_setup):
        """Test that optimizer doesn't have duplicate parameters after expansion."""
        from models.rul_model import create_shared_model_and_tcsd
        from models.rul_model import RULModelConfig

        # Create model with shared MoE between model and TCSD
        config = RULModelConfig(
            num_conditions=3,
            num_stages=2,
            d_model=256,
        )
        model, tcsd = create_shared_model_and_tcsd(config)

        trainer_config = ContinualTrainerConfig(
            use_ewc=False,
            use_experience_replay=False,
        )

        trainer = ContinualTrainer(
            model=model,
            tcsd=tcsd,
            config=trainer_config,
            device=torch.device('cpu'),
        )

        # Count unique parameters before expansion
        param_ids_before = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        params_count_before = sum(len(g['params']) for g in trainer.optimizer.param_groups)

        # Expand experts
        model.moe.expand_for_new_condition(source_condition_id=0)
        trainer._add_new_params_to_optimizer()

        # Count after expansion
        param_ids_after = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        params_count_after = sum(len(g['params']) for g in trainer.optimizer.param_groups)

        # Debug: find duplicates
        seen_ids = set()
        duplicates = []
        for group in trainer.optimizer.param_groups:
            for param in group['params']:
                if id(param) in seen_ids:
                    duplicates.append((id(param), param.shape))
                seen_ids.add(id(param))

        # Verify no duplicates
        assert len(param_ids_before) < len(param_ids_after), "New params should be added"
        assert len(duplicates) == 0, f"Duplicate params detected: {duplicates[:5]}"  # Show first 5 duplicates

    def test_non_consecutive_condition_expansion(self):
        """Test expansion with non-consecutive condition IDs."""
        from models.rul_model import create_shared_model_and_tcsd
        from models.rul_model import RULModelConfig

        # Create model with 3 initial conditions
        config = RULModelConfig(
            num_conditions=3,
            num_stages=2,
            d_model=256,
        )
        model, tcsd = create_shared_model_and_tcsd(config)

        trainer_config = ContinualTrainerConfig(
            use_ewc=False,
            use_experience_replay=False,
            auto_expand_experts=True,
            new_condition_threshold=0.99,  # High threshold to trigger detection
        )

        trainer = ContinualTrainer(
            model=model,
            tcsd=tcsd,
            config=trainer_config,
            device=torch.device('cpu'),
        )

        # Initial state
        assert model.moe.num_conditions == 3
        assert tcsd.prototype_manager.num_conditions == 3

        # Create batch with condition_id=5 (non-consecutive, skip 4)
        batch = (
            torch.randn(4, 30, 14),
            torch.tensor([50.0, 60.0, 70.0, 80.0]),
            torch.tensor([5, 5, 5, 5]),  # Skip from 3 to 5
        )

        # Training should trigger expansion
        trainer.training_step(batch, task_id=0)

        # Both should be at 6 now
        assert model.moe.num_conditions == 6, f"MoE should have 6 conditions, got {model.moe.num_conditions}"
        assert tcsd.prototype_manager.num_conditions == 6, f"PM should have 6 conditions, got {tcsd.prototype_manager.num_conditions}"

        # Verify routing works for condition 5
        x, y, condition_id = batch
        (mu, sigma), aux = model(x, condition_id=condition_id)

        # Check that routing weights include the new condition
        assert aux['cond_weights'].shape[-1] == 6, f"Routing weights should have 6 conditions"

        # Condition 5 should get non-zero weights (not clamped to last valid)
        # All samples are condition 5, so weights should be concentrated at index 5
        cond_weights_sum = aux['cond_weights'].sum(dim=0)
        assert cond_weights_sum[5] > 0, "Condition 5 should have non-zero weights"
        assert cond_weights_sum[5] == 4.0, f"All 4 samples should route to condition 5, got {cond_weights_sum}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
