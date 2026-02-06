"""
Trajectory shape encoding for degradation-aware diffusion models.

This module encodes degradation trajectory shapes for use as conditioning
information in diffusion-based generative replay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class TrajectoryShapeEncoder(nn.Module):
    """
    Encodes degradation trajectory shapes into latent representations.

    The encoding captures:
    - First-order difference: degradation rate
    - Second-order difference: degradation acceleration
    - Normalized shape: removes absolute RUL value effects

    Args:
        seq_len: Length of trajectory sequences
        feature_dim: Dimension of output encoding
        num_kernels: Number of convolution kernels for shape feature extraction
    """

    def __init__(
        self,
        seq_len: int = 30,
        feature_dim: int = 128,
        num_kernels: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Convolutional layers for shape feature extraction
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(3, num_kernels, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])

        # Shape feature aggregation
        self.aggregate = nn.Sequential(
            nn.Linear(num_kernels * 4, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
        )

        # Statistical features
        self.stat_encoder = nn.Sequential(
            nn.Linear(6, feature_dim // 4),
            nn.GELU(),
        )

        # Final encoding layer
        self.final_encoder = nn.Sequential(
            nn.Linear(feature_dim // 2 + feature_dim // 4, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

    def compute_differences(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute first and second order differences of trajectory.

        Args:
            trajectory: Shape [batch, seq_len] or [batch, seq_len, 1]

        Returns:
            diff1: First-order difference (degradation rate)
            diff2: Second-order difference (degradation acceleration)
        """
        if trajectory.dim() == 3:
            trajectory = trajectory.squeeze(-1)

        # First-order difference: rate of change
        diff1 = trajectory[:, 1:] - trajectory[:, :-1]

        # Pad to maintain sequence length
        diff1 = F.pad(diff1, (1, 0), mode='replicate')

        # Second-order difference: acceleration
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        diff2 = F.pad(diff2, (1, 0), mode='replicate')

        return diff1, diff2

    def normalize_shape(
        self, trajectory: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Normalize trajectory to unit scale, preserving shape.

        Normalization: (RUL - RUL_final) / (RUL_initial - RUL_final)
        This maps trajectory from [initial, 0] to [1, 0] approximately.

        Args:
            trajectory: Shape [batch, seq_len]
            eps: Small constant for numerical stability

        Returns:
            normalized: Normalized trajectory [batch, seq_len]
        """
        if trajectory.dim() == 3:
            trajectory = trajectory.squeeze(-1)

        # Get initial and final values
        initial = trajectory[:, 0:1]
        final = trajectory[:, -1:]

        # Normalize to [0, 1] range (approximately)
        range_val = initial - final + eps
        normalized = (trajectory - final) / range_val

        return normalized

    def extract_statistical_features(
        self,
        trajectory: torch.Tensor,
        diff1: torch.Tensor,
        diff2: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract statistical features from trajectory and its differences.

        Args:
            trajectory: Original trajectory [batch, seq_len]
            diff1: First-order difference [batch, seq_len]
            diff2: Second-order difference [batch, seq_len]

        Returns:
            stats: Statistical features [batch, 6]
        """
        if trajectory.dim() == 3:
            trajectory = trajectory.squeeze(-1)

        # Mean and std of degradation rate
        mean_rate = diff1.mean(dim=1)
        std_rate = diff1.std(dim=1) + 1e-8

        # Mean and std of acceleration
        mean_accel = diff2.mean(dim=1)
        std_accel = diff2.std(dim=1) + 1e-8

        # Monotonicity (fraction of negative diffs for RUL)
        monotonicity = (diff1 < 0).float().mean(dim=1)

        # Total degradation amount
        total_degradation = trajectory[:, 0] - trajectory[:, -1]

        stats = torch.stack([
            mean_rate, std_rate, mean_accel, std_accel,
            monotonicity, total_degradation
        ], dim=1)

        return stats

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory shape into latent representation.

        Args:
            trajectory: RUL trajectory [batch, seq_len] or [batch, seq_len, 1]

        Returns:
            encoding: Shape encoding [batch, feature_dim]
        """
        if trajectory.dim() == 3:
            trajectory = trajectory.squeeze(-1)

        # Compute differences
        diff1, diff2 = self.compute_differences(trajectory)

        # Normalize shape
        normalized = self.normalize_shape(trajectory)

        # Stack for conv input: [batch, 3, seq_len]
        conv_input = torch.stack([normalized, diff1, diff2], dim=1)

        # Multi-scale convolution features
        conv_features = []
        for conv in self.conv_layers:
            feat = F.relu(conv(conv_input))  # [batch, num_kernels, seq_len]
            feat = feat.max(dim=2)[0]  # Global max pooling
            conv_features.append(feat)

        conv_features = torch.cat(conv_features, dim=1)  # [batch, num_kernels * 4]
        conv_encoding = self.aggregate(conv_features)

        # Statistical features
        stats = self.extract_statistical_features(trajectory, diff1, diff2)
        stat_encoding = self.stat_encoder(stats)

        # Combine features
        combined = torch.cat([conv_encoding, stat_encoding], dim=1)
        encoding = self.final_encoder(combined)

        return encoding

    def encode_batch(
        self,
        trajectories: List[torch.Tensor],
        padding_value: float = -999.0
    ) -> torch.Tensor:
        """
        Encode a batch of variable-length trajectories.

        Args:
            trajectories: List of trajectories, each [seq_len] or [seq_len, 1]
            padding_value: Value used for padding (will be masked)

        Returns:
            encodings: [batch, feature_dim]
            mask: [batch, max_seq_len] indicating valid positions
        """
        # Find max length
        lengths = [t.shape[0] for t in trajectories]
        max_len = max(lengths)

        # Pad and stack
        batch_size = len(trajectories)
        padded = torch.zeros(batch_size, max_len, device=trajectories[0].device)

        for i, traj in enumerate(trajectories):
            if traj.dim() == 1:
                traj = traj.unsqueeze(-1)
            padded[i, :traj.shape[0]] = traj.squeeze(-1)

        # Create mask
        mask = torch.arange(max_len, device=padded.device)[None, :] < torch.tensor(lengths, device=padded.device)[:, None]

        return self.forward(padded * mask.float()), mask


def extract_degradation_features(
    rul_sequence: torch.Tensor,
    normalize: bool = True
) -> dict:
    """
    Extract comprehensive degradation features from RUL sequence.

    Args:
        rul_sequence: RUL values [seq_len] or [batch, seq_len]
        normalize: Whether to normalize features

    Returns:
        features: Dictionary of degradation features
    """
    if rul_sequence.dim() == 1:
        rul_sequence = rul_sequence.unsqueeze(0)

    batch_size, seq_len = rul_sequence.shape

    # First-order difference
    diff1 = torch.zeros_like(rul_sequence)
    diff1[:, 1:] = rul_sequence[:, 1:] - rul_sequence[:, :-1]
    diff1[:, 0] = diff1[:, 1]

    # Second-order difference
    diff2 = torch.zeros_like(rul_sequence)
    diff2[:, 1:] = diff1[:, 1:] - diff1[:, :-1]
    diff2[:, 0] = diff2[:, 1]

    features = {
        'rul_sequence': rul_sequence,
        'diff1': diff1,  # Degradation rate
        'diff2': diff2,  # Degradation acceleration
        'mean_rate': diff1.mean(dim=1),
        'std_rate': diff1.std(dim=1),
        'mean_accel': diff2.mean(dim=1),
        'monotonicity': (diff1 < 0).float().mean(dim=1),
        'total_degradation': rul_sequence[:, 0] - rul_sequence[:, -1],
    }

    return features


def compute_trajectory_similarity(
    traj1: torch.Tensor,
    traj2: torch.Tensor,
    metric: str = 'dtw'
) -> torch.Tensor:
    """
    Compute similarity between two trajectories.

    Args:
        traj1: First trajectory [seq_len] or [batch, seq_len]
        traj2: Second trajectory [seq_len] or [batch, seq_len]
        metric: 'dtw', 'euclidean', or 'cosine'

    Returns:
        similarity: Similarity score (lower = more similar for distances)
    """
    if metric == 'euclidean':
        if traj1.dim() == 1:
            traj1, traj2 = traj1.unsqueeze(0), traj2.unsqueeze(0)
        return F.pairwise_distance(traj1, traj2)

    elif metric == 'cosine':
        if traj1.dim() == 1:
            traj1, traj2 = traj1.unsqueeze(0), traj2.unsqueeze(0)
        return 1 - F.cosine_similarity(traj1, traj2, dim=1)

    elif metric == 'dtw':
        return _dynamic_time_warping(traj1, traj2)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def _dynamic_time_warping(
    s1: torch.Tensor,
    s2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Dynamic Time Warping distance between sequences.

    Args:
        s1: First sequence [seq_len1]
        s2: Second sequence [seq_len2]

    Returns:
        dtw_distance: DTW distance
    """
    n, m = len(s1), len(s2)

    # Initialize cost matrix
    dtw_matrix = torch.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    return torch.sqrt(dtw_matrix[n, m])
