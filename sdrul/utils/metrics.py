"""
Evaluation metrics for continual learning and diffusion quality.

This module implements:
- Continual learning metrics: ACC, BWT, FWT
- Diffusion quality metrics: MPR, TSS, PR
- RUL prediction metrics: RMSE, MAPE, etc.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


def continual_learning_metrics(
    results_per_task: torch.Tensor,
    task_order: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute continual learning metrics: ACC, BWT, FWT.

    Args:
        results_per_task: Performance matrix [num_tasks, num_tasks]
                         results_per_task[i, j] = performance on task j
                         after learning task i
        task_order: Optional task order for computation

    Returns:
        metrics: Dictionary containing ACC, BWT, FWT
    """
    num_tasks = results_per_task.shape[0]

    if task_order is None:
        task_order = list(range(num_tasks))

    # ACC (Average Accuracy): Average performance across all tasks
    # Usually computed as average of diagonal elements of final column
    acc = results_per_task[-1, :].mean().item()

    # BWT (Backward Transfer): Measure of forgetting
    # BWT = average performance drop on old tasks after learning new tasks
    bwt = 0.0
    for k in range(num_tasks - 1):
        # Performance on task k after learning final task vs after learning task k
        bwt += results_per_task[-1, k] - results_per_task[k, k]
    bwt = bwt / (num_tasks - 1)

    # FWT (Forward Transfer): Knowledge transfer from old to new tasks
    # FWT = average benefit on task k from previous tasks
    fwt = 0.0
    for k in range(1, num_tasks):
        # Performance on task k before learning it vs after learning it
        # (Assuming results_per_task[0, k] is available from random init)
        if k > 0:
            fwt += results_per_task[k-1, k] - results_per_task[k, k]
    fwt = fwt / max(num_tasks - 1, 1)

    return {
        'ACC': acc,
        'BWT': bwt,
        'FWT': fwt,
    }


def diffusion_quality_metrics(
    generated_sequences: torch.Tensor,
    real_sequences: torch.Tensor,
    model_predictions_real: Optional[torch.Tensor] = None,
    model_predictions_gen: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute diffusion generation quality metrics for RUL prediction.

    Args:
        generated_sequences: Generated RUL sequences [batch, seq_len]
        real_sequences: Real RUL sequences [batch, seq_len]
        model_predictions_real: Model predictions on real data [batch]
        model_predictions_gen: Model predictions on generated data [batch]

    Returns:
        metrics: Dictionary containing MPR, TSS, PR, and other metrics
    """
    metrics = {}

    # MPR (Monotonicity Preservation Rate)
    metrics['MPR'] = monotonicity_preservation_rate(generated_sequences)

    # TSS (Trend Similarity Score)
    metrics['TSS'] = trend_similarity_score(generated_sequences, real_sequences)

    # PR (Predictive Retention)
    if model_predictions_real is not None and model_predictions_gen is not None:
        metrics['PR'] = predictive_retention(
            model_predictions_real,
            model_predictions_gen,
            real_sequences,
        )

    # Distribution metrics
    metrics['MMD'] = maximum_mean_discrepancy(generated_sequences, real_sequences)

    # Statistical metrics
    metrics['mean_diff'] = torch.abs(generated_sequences.mean() - real_sequences.mean()).item()
    metrics['std_diff'] = torch.abs(generated_sequences.std() - real_sequences.std()).item()

    return metrics


def monotonicity_preservation_rate(
    sequences: torch.Tensor,
    threshold: float = 0.1
) -> float:
    """
    Compute MPR: fraction of sequences that are monotonic (non-increasing).

    Args:
        sequences: RUL sequences [batch, seq_len] or [batch, seq_len, sensor_dim]
        threshold: Allowed violation rate

    Returns:
        mpr: Monotonicity preservation rate in [0, 1]
    """
    if sequences.dim() == 3:
        # 3D tensor: [batch, seq_len, sensor_dim]
        # Compute MPR per sensor and average
        batch_size, seq_len, sensor_dim = sequences.shape

        # Compute differences along time dimension
        diffs = sequences[:, 1:, :] - sequences[:, :-1, :]  # [batch, seq_len-1, sensor_dim]

        # Count non-increasing pairs per sensor
        monotonic_pairs = (diffs <= threshold).float()  # [batch, seq_len-1, sensor_dim]

        # A sequence is monotonic if all time steps are non-increasing
        # Check per sensor, then average across sensors
        monotonic_per_sensor = monotonic_pairs.all(dim=1).float()  # [batch, sensor_dim]
        monotonic_sequences = monotonic_per_sensor.mean(dim=1)  # [batch]

        return monotonic_sequences.mean().item()
    elif sequences.dim() == 2:
        # 2D tensor: [batch, seq_len]
        # Compute differences
        diffs = sequences[:, 1:] - sequences[:, :-1]

        # Count non-increasing pairs (diff <= 0)
        monotonic_pairs = (diffs <= threshold).float()

        # A sequence is monotonic if all pairs are non-increasing
        monotonic_sequences = monotonic_pairs.all(dim=1).float()

        return monotonic_sequences.mean().item()
    else:
        # Single sequence [seq_len]
        diffs = sequences[1:] - sequences[:-1]
        return (diffs <= threshold).float().all().item()


def trend_similarity_score(
    generated: torch.Tensor,
    real: torch.Tensor,
    method: str = 'dtw'
) -> float:
    """
    Compute TSS: average trend similarity between generated and real sequences.

    Args:
        generated: Generated sequences [batch, seq_len] or [batch, seq_len, sensor_dim]
        real: Real sequences [batch, seq_len] or [batch, seq_len, sensor_dim]
        method: 'dtw' or 'correlation'

    Returns:
        tss: Trend similarity score (lower is better for DTW)
    """
    # Handle 3D tensors by flattening sensor dimension
    if generated.dim() == 3:
        generated = generated.flatten(start_dim=1)  # [batch, seq_len * sensor_dim]
    if real.dim() == 3:
        real = real.flatten(start_dim=1)  # [batch, seq_len * sensor_dim]

    batch_size = generated.shape[0]

    if method == 'dtw':
        scores = []
        for i in range(batch_size):
            score = _dtw_distance(generated[i], real[i])
            scores.append(score)
        return np.mean(scores)

    elif method == 'correlation':
        # Normalize by standard deviation
        gen_norm = (generated - generated.mean(dim=1, keepdim=True)) / (generated.std(dim=1, keepdim=True) + 1e-8)
        real_norm = (real - real.mean(dim=1, keepdim=True)) / (real.std(dim=1, keepdim=True) + 1e-8)

        # Compute correlation
        correlations = F.cosine_similarity(gen_norm, real_norm, dim=1)
        return (1 - correlations).mean().item()

    else:
        raise ValueError(f"Unknown method: {method}")


def _dtw_distance(s1: torch.Tensor, s2: torch.Tensor) -> float:
    """Compute DTW distance between two sequences."""
    n, m = len(s1), len(s2)

    # Use numpy for efficiency
    s1_np = s1.detach().cpu().numpy()
    s2_np = s2.detach().cpu().numpy()

    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s1_np[i - 1] - s2_np[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return np.sqrt(dtw_matrix[n, m])


def predictive_retention(
    predictions_real: torch.Tensor,
    predictions_gen: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute PR: how well model performance is retained when training on generated data.

    Args:
        predictions_real: Model predictions on real data [batch]
        predictions_gen: Model predictions on generated data [batch]
        targets: Ground truth RUL values [batch]

    Returns:
        pr: Predictive retention score in [0, 1]
    """
    # Compute RMSE on real data
    rmse_real = F.mse_loss(predictions_real, targets).sqrt()

    # Compute RMSE on generated data
    rmse_gen = F.mse_loss(predictions_gen, targets).sqrt()

    # PR = 1 - |rmse_gen - rmse_real| / rmse_real
    pr = 1 - torch.abs(rmse_gen - rmse_real) / (rmse_real + 1e-8)

    return pr.item()


def maximum_mean_discrepancy(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = 'rbf',
    sigma: float = 1.0
) -> float:
    """
    Compute MMD between two sets of sequences.

    Args:
        x: First set of sequences [batch, seq_len]
        y: Second set of sequences [batch, seq_len]
        kernel: 'rbf' or 'linear'
        sigma: RBF kernel bandwidth

    Returns:
        mmd: MMD distance
    """
    if x.dim() == 2:
        # Flatten sequences
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)

    xx = torch.cdist(x, x)
    yy = torch.cdist(y, y)
    xy = torch.cdist(x, y)

    if kernel == 'rbf':
        xx = torch.exp(-xx / (2 * sigma ** 2))
        yy = torch.exp(-yy / (2 * sigma ** 2))
        xy = torch.exp(-xy / (2 * sigma ** 2))

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd.item()


def rul_prediction_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute RUL prediction metrics.

    Args:
        predictions: Predicted RUL values [batch]
        targets: Ground truth RUL values [batch]
        threshold: Optional threshold for binary classification metrics

    Returns:
        metrics: Dictionary containing RMSE, MAE, MAPE, etc.
    """
    metrics = {}

    # RMSE
    metrics['RMSE'] = F.mse_loss(predictions, targets).sqrt().item()

    # MAE
    metrics['MAE'] = F.l1_loss(predictions, targets).item()

    # MAPE (Mean Absolute Percentage Error)
    mape = torch.abs((targets - predictions) / (targets + 1e-8)).mean() * 100
    metrics['MAPE'] = mape.item()

    # R² score
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    metrics['R2'] = r2.item()

    # Binary metrics if threshold provided
    if threshold is not None:
        pred_binary = (predictions <= threshold).float()
        target_binary = (targets <= threshold).float()

        # Accuracy
        metrics['Accuracy'] = (pred_binary == target_binary).float().mean().item()

        # Precision, Recall, F1
        tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
        fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
        fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics['Precision'] = precision.item()
        metrics['Recall'] = recall.item()
        metrics['F1'] = f1.item()

    return metrics


def uncertainty_calibration(
    predictions_mean: torch.Tensor,
    predictions_std: torch.Tensor,
    targets: torch.Tensor,
    confidence_levels: List[float] = [0.68, 0.95],
) -> Dict[str, float]:
    """
    Evaluate uncertainty calibration.

    Args:
        predictions_mean: Mean predictions [batch]
        predictions_std: Std predictions [batch]
        targets: Ground truth [batch]
        confidence_levels: Confidence levels to check

    Returns:
        metrics: Calibration metrics
    """
    metrics = {}

    for level in confidence_levels:
        # Compute z-score for confidence level
        z_score = 1.96 if level == 0.95 else 1.0

        # Compute prediction intervals
        lower = predictions_mean - z_score * predictions_std
        upper = predictions_mean + z_score * predictions_std

        # Check coverage
        in_interval = (targets >= lower) & (targets <= upper)
        coverage = in_interval.float().mean().item()

        metrics[f'Coverage_{int(level*100)}'] = coverage

    # Average calibration error
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute predicted confidence (as inverse of normalized std)
    normalized_std = predictions_std / (predictions_std.mean() + 1e-8)
    confidences = 1.0 / (1.0 + normalized_std)

    # Compute accuracy (whether target is in interval)
    errors = torch.abs(targets - predictions_mean)
    accuracies = (errors <= predictions_std).float()

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * torch.abs(accuracy_in_bin - avg_confidence_in_bin)

    metrics['ECE'] = ece.item()

    return metrics


class MetricsTracker:
    """
    Tracker for continual learning metrics across tasks.
    """

    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        self.task_order = []

    def update(
        self,
        task_id: int,
        current_task: int,
        metrics: Dict[str, float]
    ):
        """
        Update metrics after learning a task.

        Args:
            task_id: ID of the task to evaluate on
            current_task: ID of the task just learned
            metrics: Dictionary of metric values
        """
        if current_task not in self.task_order:
            self.task_order.append(current_task)

        for key, value in metrics.items():
            self.results[key][(current_task, task_id)] = value

    def get_results_matrix(self, metric: str = 'RMSE') -> torch.Tensor:
        """
        Get results matrix for a specific metric.

        Args:
            metric: Metric name

        Returns:
            matrix: Results matrix [num_tasks, num_tasks]
        """
        num_tasks = len(self.task_order)
        matrix = torch.zeros(num_tasks, num_tasks)

        for i, learned_task in enumerate(self.task_order):
            for j, eval_task in enumerate(self.task_order):
                key = (learned_task, eval_task)
                if key in self.results[metric]:
                    matrix[i, j] = self.results[metric][key]

        return matrix

    def compute_cl_metrics(self, metric: str = 'RMSE') -> Dict[str, float]:
        """
        Compute continual learning metrics for a given metric.

        Args:
            metric: Metric name (e.g., 'RMSE', 'MAE')

        Returns:
            cl_metrics: ACC, BWT, FWT computed on the metric
        """
        results_matrix = self.get_results_matrix(metric)

        # Note: For error metrics, lower is better
        # So we invert the sign for BWT/FWT interpretation

        acc = results_matrix[-1].mean().item()

        bwt = 0.0
        num_tasks = results_matrix.shape[0]
        for k in range(num_tasks - 1):
            bwt += results_matrix[-1, k] - results_matrix[k, k]
        bwt = bwt / max(num_tasks - 1, 1)

        fwt = 0.0
        for k in range(1, num_tasks):
            fwt += results_matrix[k-1, k] - results_matrix[k, k]
        fwt = fwt / max(num_tasks - 1, 1)

        return {
            f'ACC_{metric}': acc,
            f'BWT_{metric}': bwt,
            f'FWT_{metric}': fwt,
        }

    def summary(self) -> Dict[str, float]:
        """Get summary of all continual learning metrics."""
        summary = {}
        for metric in ['RMSE', 'MAE', 'MAPE']:
            if any((i, i) in self.results[metric] for i in self.task_order):
                summary.update(self.compute_cl_metrics(metric))
        return summary
