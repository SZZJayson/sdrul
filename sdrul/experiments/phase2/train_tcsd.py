"""
Training script for Trajectory-Conditioned Self-Distillation (TCSD) validation.

This script validates the TCSD module in a continual learning setting,
demonstrating that self-distillation can mitigate catastrophic forgetting
for RUL prediction across multiple operating conditions.

Phase 2 Goals:
1. Validate TCSD distillation mechanism
2. Compare with baseline (no distillation)
3. Measure forgetting mitigation via BWT metric
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data import (
    CMAPSSDataset,
    ContinualCMAPSS,
    create_synthetic_cmapss,
    CMAPSS_CONFIG,
)
from models.rul_model import RULModelConfig, RULPredictionModel, create_shared_model_and_tcsd
from models.tcsd import TCSD, TCSDConfig
from models.encoders import TransformerFeatureExtractor, GaussianRULHead
from continual import ContinualTrainer, ContinualTrainerConfig


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


class ContinualLearningMetrics:
    """
    Compute continual learning metrics: ACC, BWT, FWT.

    ACC (Average Accuracy): Average performance across all tasks
    BWT (Backward Transfer): Measures forgetting (negative = forgetting)
    FWT (Forward Transfer): Knowledge transfer to new tasks
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R[i,j] = performance on task j after training on task i
        self.R = np.zeros((num_tasks, num_tasks))
        self.task_trained = 0

    def update(self, task_trained: int, task_metrics: Dict[int, float]):
        """
        Update metrics after training on a task.

        Args:
            task_trained: Index of task just trained
            task_metrics: Dict mapping task_id -> RMSE
        """
        self.task_trained = task_trained
        for task_id, rmse in task_metrics.items():
            if task_id < self.num_tasks:
                self.R[task_trained, task_id] = rmse

    def compute_acc(self) -> float:
        """Compute Average Accuracy (lower RMSE is better)."""
        if self.task_trained == 0:
            return self.R[0, 0]
        # Average of final row (performance after all training)
        return np.mean(self.R[self.task_trained, :self.task_trained + 1])

    def compute_bwt(self) -> float:
        """
        Compute Backward Transfer.

        BWT = (1/T-1) * sum_{i=1}^{T-1} (R[T,i] - R[i,i])

        Negative BWT indicates forgetting.
        For RMSE: we want R[T,i] <= R[i,i], so negative means worse (forgetting)
        """
        if self.task_trained < 1:
            return 0.0

        T = self.task_trained
        bwt = 0.0
        for i in range(T):
            # R[T,i] - R[i,i]: if positive, performance degraded (forgetting)
            bwt += self.R[T, i] - self.R[i, i]

        return bwt / T

    def compute_fwt(self) -> float:
        """
        Compute Forward Transfer.

        FWT measures zero-shot performance on future tasks.
        """
        if self.task_trained < 1:
            return 0.0

        # For simplicity, use performance on task i before training on it
        # This requires storing pre-training metrics
        return 0.0  # Placeholder

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            'ACC': self.compute_acc(),
            'BWT': self.compute_bwt(),
            'FWT': self.compute_fwt(),
            'tasks_trained': self.task_trained + 1,
        }


def create_baseline_model(config: RULModelConfig, device: torch.device) -> nn.Module:
    """Create baseline model without TCSD (for comparison)."""
    return RULPredictionModel(config).to(device)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Returns:
        metrics: Dict with 'rmse', 'mae', 'mse'
    """
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y, cond = batch
            x = x.to(device)
            y = y.to(device)
            cond = cond.to(device)

            output, _ = model(x, condition_id=cond)

            if isinstance(output, tuple):
                mu, sigma = output
                pred = mu
            else:
                pred = output

            mse = ((pred - y) ** 2).sum().item()
            mae = (pred - y).abs().sum().item()

            total_mse += mse
            total_mae += mae
            total_samples += y.size(0)

    return {
        'mse': total_mse / total_samples,
        'rmse': (total_mse / total_samples) ** 0.5,
        'mae': total_mae / total_samples,
    }


def evaluate_all_tasks(
    model: nn.Module,
    task_loaders: Dict[int, DataLoader],
    device: torch.device,
) -> Dict[int, float]:
    """Evaluate model on all tasks, return RMSE per task."""
    results = {}
    for task_id, loader in task_loaders.items():
        metrics = evaluate_model(model, loader, device)
        results[task_id] = metrics['rmse']
    return results


def initialize_prototypes_from_data(
    tcsd: TCSD,
    dataloader: DataLoader,
    condition_id: int,
    num_trajectories: int = 50,
    device: torch.device = None,
) -> bool:
    """
    Initialize TCSD prototypes from dataset trajectories.

    Args:
        tcsd: TCSD module
        dataloader: DataLoader for the task
        condition_id: Condition ID for this task
        num_trajectories: Number of trajectories to use for initialization
        device: Device

    Returns:
        success: Whether initialization succeeded
    """
    # Collect real sensor sequences to extract degradation patterns
    all_sequences = []
    all_ruls = []

    for batch in dataloader:
        sequences, ruls, _ = batch
        all_sequences.append(sequences)
        all_ruls.extend(ruls.tolist())
        if len(all_ruls) >= num_trajectories * 2:
            break

    if not all_sequences:
        return False

    all_sequences = torch.cat(all_sequences, dim=0)

    # Create realistic degradation trajectories from actual sensor data
    # Use the mean sensor value across sensors as a proxy for health indicator
    trajectories = []
    seq_len = all_sequences.shape[1]

    for i in range(min(num_trajectories, len(all_sequences))):
        # Extract health indicator from sensor data (mean across sensors)
        sensor_seq = all_sequences[i]  # [seq_len, sensor_dim]
        hi_trajectory = sensor_seq.mean(dim=1)  # [seq_len]

        # Normalize to create a degradation pattern
        hi_min, hi_max = hi_trajectory.min(), hi_trajectory.max()
        if hi_max - hi_min > 1e-6:
            # Normalize and ensure monotonic decreasing (degradation)
            hi_norm = (hi_trajectory - hi_min) / (hi_max - hi_min + 1e-8)
            # Make it decreasing (higher HI = healthier, lower = degraded)
            trajectory = 1.0 - hi_norm
        else:
            # Fallback: create exponential decay based on RUL
            end_rul = all_ruls[i]
            t = torch.arange(seq_len).float()
            tau = seq_len / 2.0
            trajectory = torch.exp(-t / tau) * (end_rul / 125.0 + 0.5)

        trajectories.append(trajectory)

    if not trajectories:
        return False

    return tcsd.initialize_prototypes(trajectories, condition_id)


def train_task_with_tcsd(
    model: RULPredictionModel,
    tcsd: TCSD,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_id: int,
    condition_id: int,
    num_epochs: int,
    lambda_distill: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> List[float]:
    """
    Train on a single task with TCSD distillation.

    Args:
        model: RUL prediction model
        tcsd: TCSD module
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device
        task_id: Task index
        condition_id: Condition ID for this task
        num_epochs: Number of epochs
        lambda_distill: Weight for distillation loss
        logger: Logger

    Returns:
        losses: List of epoch losses
    """
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        tcsd.train()
        tcsd.set_epoch(epoch)

        total_loss = 0.0
        total_sup_loss = 0.0
        total_distill_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch+1}/{num_epochs}')

        for batch in pbar:
            x, y, cond = batch
            x = x.to(device)
            y = y.to(device)
            cond = cond.to(device)

            optimizer.zero_grad()

            # Forward through model
            (mu, sigma), aux = model(x, condition_id=cond)

            # Supervised loss (Gaussian NLL)
            L_sup = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (y - mu) ** 2 / (sigma ** 2 + 1e-6))
            L_sup = L_sup.mean()

            # Load balance loss from MoE
            L_balance = aux.get('load_balance_loss', torch.tensor(0.0, device=device))

            # TCSD distillation loss
            L_distill, distill_outputs = tcsd.on_policy_step(x, cond)

            # Total loss
            loss = L_sup + L_balance + lambda_distill * L_distill

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(tcsd.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_sup_loss += L_sup.item()
            total_distill_loss += L_distill.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sup': f'{L_sup.item():.4f}',
                'distill': f'{L_distill.item():.4f}',
            })

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)

        if logger:
            logger.info(
                f'Task {task_id} Epoch {epoch+1}: '
                f'loss={avg_loss:.4f}, sup={total_sup_loss/num_batches:.4f}, '
                f'distill={total_distill_loss/num_batches:.4f}'
            )

    return epoch_losses


def train_task_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_id: int,
    num_epochs: int,
    logger: Optional[logging.Logger] = None,
) -> List[float]:
    """
    Train on a single task without TCSD (baseline).
    """
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Baseline Task {task_id} Epoch {epoch+1}/{num_epochs}')

        for batch in pbar:
            x, y, cond = batch
            x = x.to(device)
            y = y.to(device)
            cond = cond.to(device)

            optimizer.zero_grad()

            (mu, sigma), aux = model(x, condition_id=cond)

            # Supervised loss
            L_sup = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (y - mu) ** 2 / (sigma ** 2 + 1e-6))
            L_sup = L_sup.mean()

            L_balance = aux.get('load_balance_loss', torch.tensor(0.0, device=device))

            loss = L_sup + L_balance

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)

        if logger:
            logger.info(f'Baseline Task {task_id} Epoch {epoch+1}: loss={avg_loss:.4f}')

    return epoch_losses


def run_continual_learning_experiment(
    args,
    logger: logging.Logger,
    use_tcsd: bool = True,
) -> Tuple[ContinualLearningMetrics, Dict]:
    """
    Run continual learning experiment.

    Args:
        args: Command line arguments
        logger: Logger
        use_tcsd: Whether to use TCSD (True) or baseline (False)

    Returns:
        metrics: ContinualLearningMetrics object
        history: Training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_type = "TCSD" if use_tcsd else "Baseline"
    logger.info(f'\n{"="*60}')
    logger.info(f'Running {experiment_type} Experiment')
    logger.info(f'{"="*60}')

    # Setup data
    continual_data = ContinualCMAPSS(
        data_dir=args.data_dir,
        sub_datasets=args.sub_datasets,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    num_tasks = len(continual_data)
    logger.info(f'Number of tasks: {num_tasks}')
    logger.info(f'Sub-datasets: {args.sub_datasets}')

    # Create model
    model_config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)

    if use_tcsd:
        model, tcsd = create_shared_model_and_tcsd(model_config)
        model = model.to(device)
        tcsd = tcsd.to(device)

        # Optimizer for both model and TCSD
        params = list(model.parameters())
        for name, param in tcsd.named_parameters():
            if 'feature_extractor' not in name:
                params.append(param)
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    else:
        model = create_baseline_model(model_config, device)
        tcsd = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Metrics tracking
    cl_metrics = ContinualLearningMetrics(num_tasks)
    history = {
        'task_losses': {},
        'task_metrics': {},
    }

    # Create evaluation loaders (use same as training for simplicity)
    eval_loaders = {
        i: continual_data.get_task_dataloader(i)
        for i in range(num_tasks)
    }

    # Continual learning loop
    for task_idx in range(num_tasks):
        task_id = task_idx
        condition_id = continual_data.get_condition_id(task_idx)

        logger.info(f'\n--- Task {task_idx}: {args.sub_datasets[task_idx]} (condition_id={condition_id}) ---')

        train_loader = continual_data.get_task_dataloader(task_idx)

        # Initialize prototypes for TCSD
        if use_tcsd:
            logger.info('Initializing TCSD prototypes...')
            success = initialize_prototypes_from_data(
                tcsd, train_loader, condition_id,
                num_trajectories=args.num_prototypes,
                device=device,
            )
            logger.info(f'Prototype initialization: {"success" if success else "failed"}')

            # Reset dataloader after prototype initialization
            train_loader = continual_data.get_task_dataloader(task_idx)

        # Train on task
        if use_tcsd:
            losses = train_task_with_tcsd(
                model, tcsd, train_loader, optimizer, device,
                task_id, condition_id, args.epochs_per_task,
                lambda_distill=args.lambda_distill,
                logger=logger,
            )
        else:
            losses = train_task_baseline(
                model, train_loader, optimizer, device,
                task_id, args.epochs_per_task,
                logger=logger,
            )

        history['task_losses'][task_idx] = losses

        # Evaluate on all tasks seen so far
        logger.info('Evaluating on all tasks...')
        task_rmse = evaluate_all_tasks(model, eval_loaders, device)

        for tid, rmse in task_rmse.items():
            logger.info(f'  Task {tid} ({args.sub_datasets[tid]}): RMSE = {rmse:.4f}')

        history['task_metrics'][task_idx] = task_rmse
        cl_metrics.update(task_idx, task_rmse)

        # Log current CL metrics
        summary = cl_metrics.get_summary()
        logger.info(f'CL Metrics after task {task_idx}: ACC={summary["ACC"]:.4f}, BWT={summary["BWT"]:.4f}')

    return cl_metrics, history


def main(args):
    """Main function."""
    # Setup
    logger = setup_logging(args.log_dir, args.experiment_name)

    logger.info('='*60)
    logger.info('Phase 2: TCSD Validation Experiment')
    logger.info('='*60)
    logger.info(f'Arguments: {vars(args)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Create data directory and synthetic data if needed
    os.makedirs(args.data_dir, exist_ok=True)

    for sub_id in args.sub_datasets:
        path = os.path.join(args.data_dir, f'{sub_id}_train.npz')
        if not os.path.exists(path):
            logger.info(f'Creating synthetic data for {sub_id}...')
            create_synthetic_cmapss(
                num_engines=args.num_engines,
                save_path=path,
                seed=int(sub_id[-1]) * 42,
            )

    # Run TCSD experiment
    tcsd_metrics, tcsd_history = run_continual_learning_experiment(
        args, logger, use_tcsd=True
    )

    # Run baseline experiment
    baseline_metrics, baseline_history = run_continual_learning_experiment(
        args, logger, use_tcsd=False
    )

    # Compare results
    logger.info('\n' + '='*60)
    logger.info('FINAL COMPARISON')
    logger.info('='*60)

    tcsd_summary = tcsd_metrics.get_summary()
    baseline_summary = baseline_metrics.get_summary()

    logger.info(f'\nTCSD Results:')
    logger.info(f'  ACC (avg RMSE): {tcsd_summary["ACC"]:.4f}')
    logger.info(f'  BWT (forgetting): {tcsd_summary["BWT"]:.4f}')

    logger.info(f'\nBaseline Results:')
    logger.info(f'  ACC (avg RMSE): {baseline_summary["ACC"]:.4f}')
    logger.info(f'  BWT (forgetting): {baseline_summary["BWT"]:.4f}')

    # Improvement
    acc_improvement = baseline_summary["ACC"] - tcsd_summary["ACC"]
    bwt_improvement = baseline_summary["BWT"] - tcsd_summary["BWT"]

    logger.info(f'\nImprovement (TCSD vs Baseline):')
    logger.info(f'  ACC improvement: {acc_improvement:.4f} ({"better" if acc_improvement > 0 else "worse"})')
    logger.info(f'  BWT improvement: {bwt_improvement:.4f} ({"less forgetting" if bwt_improvement > 0 else "more forgetting"})')

    # Save results
    results = {
        'args': vars(args),
        'tcsd': {
            'summary': tcsd_summary,
            'history': {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                       for k, vals in tcsd_history['task_losses'].items()},
        },
        'baseline': {
            'summary': baseline_summary,
            'history': {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                       for k, vals in baseline_history['task_losses'].items()},
        },
        'improvement': {
            'ACC': float(acc_improvement),
            'BWT': float(bwt_improvement),
        }
    }

    results_path = os.path.join(args.log_dir, f'{args.experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to: {results_path}')

    logger.info('\nExperiment completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: TCSD Validation')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--sub_datasets', type=str, nargs='+',
                        default=['FD001', 'FD003'],
                        help='Sub-datasets to use as tasks')
    parser.add_argument('--seq_len', type=int, default=30,
                        help='Input sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_engines', type=int, default=50,
                        help='Number of engines for synthetic data')

    # Model parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')

    # TCSD parameters
    parser.add_argument('--num_prototypes', type=int, default=50,
                        help='Number of trajectories for prototype initialization')
    parser.add_argument('--lambda_distill', type=float, default=1.0,
                        help='Weight for distillation loss')

    # Training parameters
    parser.add_argument('--epochs_per_task', type=int, default=10,
                        help='Number of epochs per task')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default='tcsd_validation',
                        help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='experiments/phase2/logs',
                        help='Log directory')

    args = parser.parse_args()
    main(args)
