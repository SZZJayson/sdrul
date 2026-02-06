"""
Training script for the complete SDFT-DIMIX framework.

Phase 3: Full Framework Integration
- DCD (Degradation-Conditioned Diffusion) for generative replay
- TCSD (Trajectory-Conditioned Self-Distillation) for adaptation
- DSA-MoE (Degradation-Stage-Aware Mixture of Experts) for knowledge organization
- EWC (Elastic Weight Consolidation) for regularization

This script validates the complete continual learning framework for RUL prediction.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from dataclasses import dataclass
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
from models.dcd import DegradationConditionedDiffusion, DCDConfig
from continual import ContinualTrainer, ContinualTrainerConfig
from continual.ewc import EWCRegularizer
from continual.replay import ReplayBuffer


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
    """Compute continual learning metrics: ACC, BWT, FWT."""

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.R = np.zeros((num_tasks, num_tasks))
        self.task_trained = 0

    def update(self, task_trained: int, task_metrics: Dict[int, float]):
        self.task_trained = task_trained
        for task_id, rmse in task_metrics.items():
            if task_id < self.num_tasks:
                self.R[task_trained, task_id] = rmse

    def compute_acc(self) -> float:
        if self.task_trained == 0:
            return self.R[0, 0]
        return np.mean(self.R[self.task_trained, :self.task_trained + 1])

    def compute_bwt(self) -> float:
        if self.task_trained < 1:
            return 0.0
        T = self.task_trained
        bwt = sum(self.R[T, i] - self.R[i, i] for i in range(T))
        return bwt / T

    def compute_fwt(self) -> float:
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        return {
            'ACC': self.compute_acc(),
            'BWT': self.compute_bwt(),
            'FWT': self.compute_fwt(),
            'tasks_trained': self.task_trained + 1,
        }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
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

            total_mse += ((pred - y) ** 2).sum().item()
            total_mae += (pred - y).abs().sum().item()
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
    """Evaluate model on all tasks."""
    return {
        task_id: evaluate_model(model, loader, device)['rmse']
        for task_id, loader in task_loaders.items()
    }


def initialize_prototypes_from_data(
    tcsd: TCSD,
    dataloader: DataLoader,
    condition_id: int,
    num_trajectories: int = 50,
) -> bool:
    """Initialize TCSD prototypes from dataset."""
    all_ruls = []
    for batch in dataloader:
        _, ruls, _ = batch
        all_ruls.extend(ruls.tolist())
        if len(all_ruls) >= num_trajectories * 2:
            break

    trajectories = []
    seq_len = 30
    for i in range(min(num_trajectories, len(all_ruls))):
        end_rul = all_ruls[i]
        start_rul = end_rul + seq_len
        trajectory = torch.linspace(start_rul, end_rul, seq_len)
        trajectory = trajectory + torch.randn(seq_len) * 2.0
        trajectories.append(trajectory)

    if not trajectories:
        return False
    return tcsd.initialize_prototypes(trajectories, condition_id)


def train_dcd_on_task(
    dcd: DegradationConditionedDiffusion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    condition_id: int,
    num_epochs: int,
    logger: logging.Logger,
) -> List[float]:
    """Pre-train DCD on a task's data."""
    losses = []
    dcd.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x, y, cond = batch
            x = x.to(device)
            y = y.to(device)

            # Create RUL trajectory from scalar RUL
            batch_size = x.size(0)
            seq_len = x.size(1)
            rul_traj = y.unsqueeze(-1).expand(-1, seq_len)

            # Override condition_id for this task
            cond_tensor = torch.full((batch_size,), condition_id, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = dcd.compute_loss(x, rul_traj, cond_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dcd.parameters(), max_norm=1.0)
            optimizer.step()
            dcd.update_ema()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            logger.info(f'  DCD Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}')

    return losses


class FullFrameworkTrainer:
    """
    Complete SDFT-DIMIX framework trainer.

    Integrates:
    - RULPredictionModel (with DSA-MoE)
    - TCSD for self-distillation
    - DCD for generative replay
    - EWC for regularization
    - Experience replay buffer
    """

    def __init__(
        self,
        model: RULPredictionModel,
        tcsd: TCSD,
        dcd: DegradationConditionedDiffusion,
        config: 'FullFrameworkConfig',
        device: torch.device,
        logger: logging.Logger,
    ):
        self.model = model.to(device)
        self.tcsd = tcsd.to(device)
        self.dcd = dcd.to(device)
        self.config = config
        self.device = device
        self.logger = logger

        # EWC regularizer
        self.ewc = EWCRegularizer(
            model=model,
            ewc_lambda=config.lambda_ewc,
            online=True,
            gamma=0.9,
        ) if config.use_ewc else None

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.replay_buffer_size,
        ) if config.use_experience_replay else None

        # Optimizer for model and TCSD
        params = list(model.parameters())
        for name, param in tcsd.named_parameters():
            if 'feature_extractor' not in name:
                params.append(param)
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

        # DCD optimizer (separate)
        self.dcd_optimizer = torch.optim.AdamW(
            dcd.get_training_parameters(),
            lr=config.dcd_learning_rate,
        )

        # Training state
        self.completed_tasks: List[int] = []
        self.prototype_buffer: Dict[int, List[torch.Tensor]] = {}

    def train_task(
        self,
        task_id: int,
        condition_id: int,
        train_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, List[float]]:
        """Train on a single task with full framework."""
        history = {
            'total_loss': [],
            'supervised_loss': [],
            'distillation_loss': [],
            'replay_loss': [],
            'ewc_loss': [],
            'balance_loss': [],
        }

        # Phase 1: Pre-train DCD on this task's data
        if self.config.use_generative_replay and self.config.dcd_pretrain_epochs > 0:
            self.logger.info(f'Pre-training DCD on task {task_id}...')
            train_dcd_on_task(
                self.dcd, train_loader, self.dcd_optimizer,
                self.device, condition_id,
                self.config.dcd_pretrain_epochs, self.logger,
            )

        # Phase 2: Initialize TCSD prototypes
        self.logger.info(f'Initializing TCSD prototypes for task {task_id}...')
        initialize_prototypes_from_data(
            self.tcsd, train_loader, condition_id,
            num_trajectories=self.config.num_prototypes,
        )

        # Store prototypes for DCD replay
        protos = self.tcsd.prototype_manager.get_all_prototypes(condition_id)
        if protos:
            self.prototype_buffer[condition_id] = [p.cpu() for p in protos]

        # Phase 3: Main training loop
        for epoch in range(num_epochs):
            self.model.train()
            self.tcsd.train()
            self.tcsd.set_epoch(epoch)

            epoch_metrics = {k: 0.0 for k in history.keys()}
            num_batches = 0

            pbar = tqdm(train_loader, desc=f'Task {task_id} Epoch {epoch+1}/{num_epochs}')

            for batch in pbar:
                x, y, cond = batch
                x = x.to(self.device)
                y = y.to(self.device)
                cond = cond.to(self.device)

                self.optimizer.zero_grad()

                # 1. Supervised loss on new data
                (mu, sigma), aux = self.model(x, condition_id=cond)
                L_sup = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (y - mu) ** 2 / (sigma ** 2 + 1e-6))
                L_sup = L_sup.mean()

                # 2. Load balance loss from MoE
                L_balance = aux.get('load_balance_loss', torch.tensor(0.0, device=self.device))

                # 3. TCSD distillation loss
                L_distill, _ = self.tcsd.on_policy_step(x, cond)

                # 4. Generative replay loss (from DCD)
                L_gen_replay = torch.tensor(0.0, device=self.device)
                if self.config.use_generative_replay and self.completed_tasks and self.prototype_buffer:
                    L_gen_replay = self._compute_generative_replay_loss()

                # 5. Experience replay loss
                L_exp_replay = torch.tensor(0.0, device=self.device)
                if self.config.use_experience_replay and self.replay_buffer and len(self.replay_buffer) > 0:
                    L_exp_replay = self._compute_experience_replay_loss(task_id)

                # 6. EWC regularization
                L_ewc = torch.tensor(0.0, device=self.device)
                if self.ewc and self.ewc.num_tasks > 0:
                    L_ewc = self.ewc.ewc_loss()

                # Total loss
                total_loss = (
                    self.config.lambda_supervised * L_sup +
                    L_balance +
                    self.config.lambda_distillation * L_distill +
                    self.config.lambda_replay * (L_gen_replay + L_exp_replay) +
                    L_ewc
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.tcsd.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Add to replay buffer
                if self.replay_buffer:
                    self.replay_buffer.add_batch(x.cpu(), y.cpu(), cond.cpu())

                # Track metrics
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['supervised_loss'] += L_sup.item()
                epoch_metrics['distillation_loss'] += L_distill.item()
                epoch_metrics['replay_loss'] += (L_gen_replay + L_exp_replay).item()
                epoch_metrics['ewc_loss'] += L_ewc.item()
                epoch_metrics['balance_loss'] += L_balance.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{total_loss.item():.2f}',
                    'sup': f'{L_sup.item():.2f}',
                    'dist': f'{L_distill.item():.4f}',
                })

            # Average metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= num_batches
                history[k].append(epoch_metrics[k])

            self.logger.info(
                f'Task {task_id} Epoch {epoch+1}: '
                f'loss={epoch_metrics["total_loss"]:.4f}, '
                f'sup={epoch_metrics["supervised_loss"]:.4f}, '
                f'dist={epoch_metrics["distillation_loss"]:.4f}, '
                f'replay={epoch_metrics["replay_loss"]:.4f}'
            )

        return history

    def _compute_generative_replay_loss(self) -> torch.Tensor:
        """Compute loss on DCD-generated replay samples."""
        # Select random past condition
        past_conditions = [c for c in self.prototype_buffer.keys() if c in [
            cid for cid in self.completed_tasks
        ]]

        if not past_conditions:
            return torch.tensor(0.0, device=self.device)

        # Generate replay samples
        try:
            replay_x, replay_cond = self.dcd.generate_replay_batch(
                {c: self.prototype_buffer[c] for c in past_conditions[:2]},
                samples_per_condition=self.config.replay_batch_size // max(1, len(past_conditions)),
                use_ema=True,
            )
        except Exception as e:
            self.logger.warning(f'DCD replay generation failed: {e}')
            return torch.tensor(0.0, device=self.device)

        replay_x = replay_x.to(self.device)
        replay_cond = replay_cond.to(self.device)

        # Forward through model
        (mu, sigma), _ = self.model(replay_x, condition_id=replay_cond)

        # Use TCSD distillation on replay samples
        L_replay, _ = self.tcsd.on_policy_step(replay_x, replay_cond)

        return L_replay

    def _compute_experience_replay_loss(self, current_task: int) -> torch.Tensor:
        """Compute loss on experience replay samples."""
        try:
            replay_x, replay_y, replay_cond = self.replay_buffer.sample_for_replay(
                self.config.replay_batch_size,
                current_task,
                self.device,
            )
        except (ValueError, RuntimeError):
            return torch.tensor(0.0, device=self.device)

        if replay_x.size(0) == 0:
            return torch.tensor(0.0, device=self.device)

        (mu, sigma), _ = self.model(replay_x, condition_id=replay_cond)
        L_replay = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (replay_y - mu) ** 2 / (sigma ** 2 + 1e-6))
        return L_replay.mean()

    def on_task_end(self, task_id: int, condition_id: int, dataloader: DataLoader):
        """Called after completing a task."""
        self.completed_tasks.append(condition_id)

        # Consolidate EWC
        if self.ewc:
            self.logger.info(f'Computing Fisher information for task {task_id}...')
            self.ewc.consolidate(dataloader, num_samples=self.config.ewc_samples)

        # Update DCD EMA
        if self.dcd:
            self.dcd.update_ema()


@dataclass
class FullFrameworkConfig:
    """Configuration for full framework training."""

    # Loss weights
    lambda_supervised: float = 1.0
    lambda_distillation: float = 1.0
    lambda_replay: float = 0.5
    lambda_ewc: float = 1000.0

    # Learning rates
    learning_rate: float = 1e-4
    dcd_learning_rate: float = 1e-4

    # Replay settings
    use_generative_replay: bool = True
    use_experience_replay: bool = True
    replay_buffer_size: int = 5000
    replay_batch_size: int = 16

    # DCD settings
    dcd_pretrain_epochs: int = 5

    # EWC settings
    use_ewc: bool = True
    ewc_samples: int = 500

    # TCSD settings
    num_prototypes: int = 50


def run_experiment(
    args,
    logger: logging.Logger,
    method: str = 'full',
) -> Tuple[ContinualLearningMetrics, Dict]:
    """
    Run continual learning experiment.

    Args:
        args: Command line arguments
        logger: Logger
        method: 'full', 'tcsd_only', 'ewc_only', 'replay_only', 'baseline'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'\n{"="*60}')
    logger.info(f'Running {method.upper()} Experiment')
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

    # Create models
    model_config = RULModelConfig.from_dataset_config(CMAPSS_CONFIG)
    model, tcsd = create_shared_model_and_tcsd(model_config)

    # Create DCD
    dcd_config = DCDConfig(
        sensor_dim=CMAPSS_CONFIG.sensor_dim,
        seq_len=args.seq_len,
        num_timesteps=100,  # Reduced for faster training
    )
    dcd = DegradationConditionedDiffusion(dcd_config, num_conditions=CMAPSS_CONFIG.num_conditions)

    # Configure based on method
    config = FullFrameworkConfig(
        lambda_supervised=1.0,
        lambda_distillation=args.lambda_distill if method in ['full', 'tcsd_only'] else 0.0,
        lambda_replay=args.lambda_replay if method in ['full', 'replay_only'] else 0.0,
        lambda_ewc=args.lambda_ewc if method in ['full', 'ewc_only'] else 0.0,
        use_generative_replay=(method in ['full', 'replay_only']),
        use_experience_replay=(method in ['full', 'replay_only']),
        use_ewc=(method in ['full', 'ewc_only']),
        dcd_pretrain_epochs=args.dcd_epochs if method in ['full', 'replay_only'] else 0,
        num_prototypes=args.num_prototypes,
        learning_rate=args.learning_rate,
    )

    # Create trainer
    trainer = FullFrameworkTrainer(
        model=model,
        tcsd=tcsd,
        dcd=dcd,
        config=config,
        device=device,
        logger=logger,
    )

    # Metrics tracking
    cl_metrics = ContinualLearningMetrics(num_tasks)
    history = {'task_losses': {}, 'task_metrics': {}}

    eval_loaders = {i: continual_data.get_task_dataloader(i) for i in range(num_tasks)}

    # Continual learning loop
    for task_idx in range(num_tasks):
        condition_id = continual_data.get_condition_id(task_idx)
        logger.info(f'\n--- Task {task_idx}: {args.sub_datasets[task_idx]} (cond={condition_id}) ---')

        train_loader = continual_data.get_task_dataloader(task_idx)

        # Train on task
        task_history = trainer.train_task(
            task_idx, condition_id, train_loader, args.epochs_per_task
        )
        history['task_losses'][task_idx] = task_history

        # End task
        trainer.on_task_end(task_idx, condition_id, train_loader)

        # Evaluate
        logger.info('Evaluating on all tasks...')
        task_rmse = evaluate_all_tasks(model, eval_loaders, device)
        for tid, rmse in task_rmse.items():
            logger.info(f'  Task {tid}: RMSE = {rmse:.4f}')

        history['task_metrics'][task_idx] = task_rmse
        cl_metrics.update(task_idx, task_rmse)

        summary = cl_metrics.get_summary()
        logger.info(f'CL Metrics: ACC={summary["ACC"]:.4f}, BWT={summary["BWT"]:.4f}')

    return cl_metrics, history


def main(args):
    """Main function."""
    logger = setup_logging(args.log_dir, args.experiment_name)

    logger.info('='*60)
    logger.info('Phase 3: Full Framework (SDFT-DIMIX) Experiment')
    logger.info('='*60)
    logger.info(f'Arguments: {vars(args)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Create data
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

    # Run experiments
    results = {}

    # Full framework
    full_metrics, full_history = run_experiment(args, logger, method='full')
    results['full'] = {
        'summary': full_metrics.get_summary(),
        'history': {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                   for k, v in full_history.items()},
    }

    # Baseline (no CL techniques)
    baseline_metrics, baseline_history = run_experiment(args, logger, method='baseline')
    results['baseline'] = {
        'summary': baseline_metrics.get_summary(),
        'history': {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                   for k, v in baseline_history.items()},
    }

    # Ablations (optional)
    if args.run_ablations:
        for method in ['tcsd_only', 'ewc_only', 'replay_only']:
            logger.info(f'\nRunning ablation: {method}')
            metrics, history = run_experiment(args, logger, method=method)
            results[method] = {
                'summary': metrics.get_summary(),
                'history': {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                           for k, v in history.items()},
            }

    # Final comparison
    logger.info('\n' + '='*60)
    logger.info('FINAL COMPARISON')
    logger.info('='*60)

    for method, data in results.items():
        summary = data['summary']
        logger.info(f'\n{method.upper()}:')
        logger.info(f'  ACC (avg RMSE): {summary["ACC"]:.4f}')
        logger.info(f'  BWT (forgetting): {summary["BWT"]:.4f}')

    # Improvement over baseline
    if 'full' in results and 'baseline' in results:
        full_acc = results['full']['summary']['ACC']
        base_acc = results['baseline']['summary']['ACC']
        full_bwt = results['full']['summary']['BWT']
        base_bwt = results['baseline']['summary']['BWT']

        logger.info(f'\nFull Framework vs Baseline:')
        logger.info(f'  ACC improvement: {base_acc - full_acc:.4f} ({"better" if base_acc > full_acc else "worse"})')
        logger.info(f'  BWT improvement: {base_bwt - full_bwt:.4f} ({"less forgetting" if base_bwt > full_bwt else "more forgetting"})')

    # Save results
    results['args'] = vars(args)
    results_path = os.path.join(args.log_dir, f'{args.experiment_name}_results.json')

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    logger.info(f'\nResults saved to: {results_path}')

    logger.info('\nExperiment completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3: Full Framework Training')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--sub_datasets', type=str, nargs='+',
                        default=['FD001', 'FD003'])
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_engines', type=int, default=50)

    # Model parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    # Loss weights
    parser.add_argument('--lambda_distill', type=float, default=1.0)
    parser.add_argument('--lambda_replay', type=float, default=0.5)
    parser.add_argument('--lambda_ewc', type=float, default=1000.0)

    # Component settings
    parser.add_argument('--num_prototypes', type=int, default=50)
    parser.add_argument('--dcd_epochs', type=int, default=5)

    # Training parameters
    parser.add_argument('--epochs_per_task', type=int, default=10)
    parser.add_argument('--run_ablations', action='store_true',
                        help='Run ablation experiments')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default='full_framework')
    parser.add_argument('--log_dir', type=str, default='experiments/phase3/logs')

    args = parser.parse_args()
    main(args)
