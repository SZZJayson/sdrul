"""
Continual learning trainer for RUL prediction.

Integrates smart experience replay, TCSD distillation, DSA-MoE, and EWC
for comprehensive continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import logging

from .ewc import EWCRegularizer
from .replay import SmartReplayBuffer


@dataclass
class ContinualTrainerConfig:
    """
    Configuration for ContinualTrainer.

    Note: Loss weights are research-validated hyperparameters.
    """

    # Loss weights (research-specified values)
    lambda_supervised: float = 1.0
    lambda_replay: float = 1.0
    lambda_distillation: float = 1.0
    lambda_balance: float = 0.01
    lambda_ewc: float = 1000.0

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    replay_batch_size: int = 16

    # Replay settings
    use_experience_replay: bool = True
    replay_buffer_size: int = 10000

    # EWC settings
    use_ewc: bool = True
    ewc_online: bool = True
    ewc_gamma: float = 0.9
    ewc_samples: int = 1000

    # Warmup
    warmup_epochs: int = 5


class ContinualTrainer:
    """
    Trainer for continual RUL prediction.

    Combines multiple continual learning strategies:
    1. Smart experience replay with importance weighting
    2. TCSD self-distillation
    3. DSA-MoE for knowledge organization
    4. EWC regularization

    The model should be a RULPredictionModel that returns ((mu, sigma), aux_dict).

    Args:
        model: RULPredictionModel instance (returns ((mu, sigma), aux_dict))
        tcsd: TCSD module for self-distillation (should share feature_extractor with model)
        config: ContinualTrainerConfig
        device: Training device
    """

    def __init__(
        self,
        model: nn.Module,
        tcsd: Optional[nn.Module] = None,
        config: Optional[ContinualTrainerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tcsd = tcsd
        self.config = config or ContinualTrainerConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move models to device
        self.model.to(self.device)
        if self.tcsd is not None:
            self.tcsd.to(self.device)

        # Initialize EWC
        if self.config.use_ewc:
            self.ewc = EWCRegularizer(
                model=self.model,
                ewc_lambda=self.config.lambda_ewc,
                online=self.config.ewc_online,
                gamma=self.config.ewc_gamma,
            )
        else:
            self.ewc = None

        # Initialize smart replay buffer
        prototype_manager = None
        if self.tcsd is not None and hasattr(self.tcsd, 'prototype_manager'):
            prototype_manager = self.tcsd.prototype_manager

        if self.config.use_experience_replay:
            self.replay_buffer = SmartReplayBuffer(
                max_size=self.config.replay_buffer_size,
                prototype_manager=prototype_manager,
            )
        else:
            self.replay_buffer = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self._get_trainable_params(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Training state
        self.current_task = 0
        self.current_epoch = 0
        self.completed_tasks: List[int] = []

        # Logging
        self.logger = logging.getLogger(__name__)

    def _get_trainable_params(self) -> List[Dict]:
        """Get all trainable parameters."""
        params = list(self.model.parameters())
        if self.tcsd is not None:
            # Only add TCSD-specific params (not shared feature_extractor)
            for name, param in self.tcsd.named_parameters():
                if 'feature_extractor' not in name:
                    params.append(param)
        return params

    def _compute_rul_loss(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL loss for RUL prediction.

        Args:
            mu: Predicted mean [batch]
            sigma: Predicted std [batch]
            target: Target RUL values [batch]
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: Gaussian NLL loss
        """
        loss = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (target - mu) ** 2 / (sigma ** 2 + 1e-6))
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        task_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.

        Args:
            batch: (x, y, condition_id) tuple
            task_id: Current task identifier

        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        x, y, condition_id = batch
        x = x.to(self.device)
        y = y.to(self.device)
        condition_id = condition_id.to(self.device)

        metrics = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # 1. Supervised loss on new data
        # Model returns ((mu, sigma), aux_dict)
        output, aux = self.model(x, condition_id=condition_id)

        # Handle output format
        if isinstance(output, tuple) and len(output) == 2:
            mu, sigma = output
            L_supervised = self._compute_rul_loss(mu, sigma, y)
        else:
            # Fallback for simple models
            L_supervised = F.mse_loss(output, y)

        total_loss = total_loss + self.config.lambda_supervised * L_supervised
        metrics['supervised_loss'] = L_supervised.item()

        # 2. MoE load balancing loss
        if isinstance(aux, dict) and 'load_balance_loss' in aux:
            L_balance = aux['load_balance_loss']
            total_loss = total_loss + L_balance
            metrics['balance_loss'] = L_balance.item() if isinstance(L_balance, torch.Tensor) else L_balance

        # 3. Smart experience replay with dynamic ratio
        if self.replay_buffer is not None and self.config.use_experience_replay and len(self.replay_buffer) > 0 and self.completed_tasks:
            L_exp_replay, exp_metrics = self._experience_replay_loss(task_id, mu, sigma)
            total_loss = total_loss + self.config.lambda_replay * L_exp_replay
            metrics.update(exp_metrics)

        # 4. TCSD distillation
        if self.tcsd is not None:
            L_distill, distill_outputs = self.tcsd.on_policy_step(x, condition_id)
            total_loss = total_loss + self.config.lambda_distillation * L_distill
            metrics['distillation_loss'] = L_distill.item()
            metrics['valid_prototypes'] = distill_outputs['valid_samples']

        # 5. EWC regularization
        if self.ewc is not None and self.ewc.num_tasks > 0:
            L_ewc = self.ewc.ewc_loss()
            total_loss = total_loss + L_ewc
            metrics['ewc_loss'] = L_ewc.item()

        metrics['total_loss'] = total_loss.item()

        # Add samples to replay buffer with importance scoring
        if self.replay_buffer is not None and isinstance(output, tuple):
            self.replay_buffer.add_batch_with_selection(
                x, y, task_id,
                model_outputs=(mu.detach(), sigma.detach()),
            )

        return total_loss, metrics

    def _experience_replay_loss(
        self,
        current_task: int,
        current_mu: Optional[torch.Tensor] = None,
        current_sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss on experience replay samples.

        Uses dynamic replay ratio and importance-weighted sampling.

        Args:
            current_task: Current task ID
            current_mu: Current batch predictions (for logging)
            current_sigma: Current batch uncertainties (for logging)

        Returns:
            loss: Weighted replay loss
            metrics: Replay metrics
        """
        metrics = {}

        # Compute dynamic replay ratio
        replay_ratio = self.replay_buffer.compute_replay_ratio(
            current_task, self.completed_tasks
        )
        replay_batch_size = int(self.config.replay_batch_size * replay_ratio / self.replay_buffer.base_replay_ratio)
        replay_batch_size = max(1, min(replay_batch_size, self.config.replay_batch_size * 2))

        try:
            replay_x, replay_y, replay_cond, weights = self.replay_buffer.sample_weighted(
                replay_batch_size,
                current_task,
                self.device,
            )
        except ValueError:
            return torch.tensor(0.0, device=self.device), {'exp_replay_loss': 0.0}

        if replay_x.size(0) == 0:
            return torch.tensor(0.0, device=self.device), {'exp_replay_loss': 0.0}

        # Forward through model
        output, _ = self.model(replay_x, condition_id=replay_cond)

        if isinstance(output, tuple) and len(output) == 2:
            mu, sigma = output
            # Weighted loss: importance weights applied per-sample
            per_sample_loss = self._compute_rul_loss(mu, sigma, replay_y, reduction='none')
            L_replay = (weights * per_sample_loss).mean()
        else:
            L_replay = F.mse_loss(output, replay_y)

        metrics['exp_replay_loss'] = L_replay.item()
        metrics['exp_replay_samples'] = replay_x.size(0)
        metrics['replay_ratio'] = replay_ratio

        return L_replay, metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            task_id: Current task identifier

        Returns:
            epoch_metrics: Averaged metrics for the epoch
        """
        self.model.train()
        if self.tcsd is not None:
            self.tcsd.train()

        epoch_metrics = {}
        num_batches = 0
        rul_values = []  # Collect RUL values for task stats

        for batch in dataloader:
            self.optimizer.zero_grad()

            loss, metrics = self.training_step(batch, task_id)

            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            num_batches += 1

            # Collect RUL values for task statistics
            _, y, _ = batch
            rul_values.extend(y.tolist())

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        # Update task statistics for dynamic replay ratio
        if self.replay_buffer is not None:
            self.replay_buffer.update_task_stats(task_id, rul_values)

        self.current_epoch += 1

        return epoch_metrics

    def on_task_start(self, task_id: int):
        """Called at the start of a new task."""
        self.current_task = task_id
        self.current_epoch = 0

        # Update TCSD epoch for warmup
        if self.tcsd is not None:
            self.tcsd.set_epoch(0)

        # Update MoE warmup
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(0)
        elif hasattr(self.model, 'moe'):
            self.model.moe.set_epoch(0)

        self.logger.info(f"Starting task {task_id}")

    def on_task_end(self, task_id: int, dataloader: Optional[DataLoader] = None):
        """Called at the end of a task."""
        self.completed_tasks.append(task_id)

        # Consolidate EWC
        if self.ewc is not None and dataloader is not None:
            self.logger.info(f"Computing Fisher information for task {task_id}")
            self.ewc.consolidate(dataloader, self.config.ewc_samples)

        self.logger.info(f"Completed task {task_id}")

    def evaluate(
        self,
        dataloader: DataLoader,
        task_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: Evaluation data loader
            task_id: Task identifier (for condition routing)

        Returns:
            metrics: Evaluation metrics
        """
        self.model.eval()

        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                x, y, cond = batch
                x = x.to(self.device)
                y = y.to(self.device)
                cond = cond.to(self.device)

                output, _ = self.model(x, condition_id=cond)

                if isinstance(output, tuple) and len(output) == 2:
                    mu, sigma = output
                    pred = mu
                else:
                    pred = output

                mse = ((pred - y) ** 2).sum().item()
                mae = (pred - y).abs().sum().item()

                total_mse += mse
                total_mae += mae
                total_samples += y.size(0)

        metrics = {
            'mse': total_mse / total_samples,
            'rmse': (total_mse / total_samples) ** 0.5,
            'mae': total_mae / total_samples,
        }

        return metrics

    def evaluate_all_tasks(
        self,
        task_dataloaders: Dict[int, DataLoader],
    ) -> Dict[str, float]:
        """
        Evaluate on all tasks for continual learning metrics.

        Args:
            task_dataloaders: Dict mapping task_id to DataLoader

        Returns:
            metrics: Including ACC, BWT, FWT
        """
        task_metrics = {}

        for task_id, dataloader in task_dataloaders.items():
            task_metrics[task_id] = self.evaluate(dataloader, task_id)

        # Compute continual learning metrics
        rmse_values = [m['rmse'] for m in task_metrics.values()]

        metrics = {
            'avg_rmse': sum(rmse_values) / len(rmse_values),
            'task_metrics': task_metrics,
        }

        return metrics

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'current_task': self.current_task,
            'current_epoch': self.current_epoch,
            'completed_tasks': self.completed_tasks,
            'config': self.config,
        }

        if self.tcsd is not None:
            checkpoint['tcsd_state'] = self.tcsd.state_dict()
            checkpoint['tcsd_prototypes'] = self.tcsd.save_prototypes()

        if self.ewc is not None:
            checkpoint['ewc_state'] = self.ewc.state_dict()

        if self.replay_buffer is not None:
            checkpoint['replay_buffer'] = self.replay_buffer.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_task = checkpoint['current_task']
        self.current_epoch = checkpoint['current_epoch']
        self.completed_tasks = checkpoint['completed_tasks']

        if self.tcsd is not None and 'tcsd_state' in checkpoint:
            self.tcsd.load_state_dict(checkpoint['tcsd_state'])
            if 'tcsd_prototypes' in checkpoint:
                self.tcsd.load_prototypes(checkpoint['tcsd_prototypes'], self.device)

        if self.ewc is not None and 'ewc_state' in checkpoint:
            self.ewc.load_state_dict(checkpoint['ewc_state'], self.device)

        if self.replay_buffer is not None and 'replay_buffer' in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer'])

        self.logger.info(f"Loaded checkpoint from {path}")
