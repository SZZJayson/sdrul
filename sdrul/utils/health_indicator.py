"""
Health Indicator computation for RUL prediction.

This module provides methods to compute health indicators from sensor data,
which are used for degradation stage routing and assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class HealthIndicatorComputer(nn.Module):
    """
    Computes health indicators from multi-sensor time series data.

    The health indicator (HI) represents the current degradation state:
    - HI ≈ 1.0: Healthy / Early stage
    - HI ≈ 0.5: Mid degradation
    - HI ≈ 0.0: Near failure / Late stage

    Args:
        sensor_dim: Number of sensor channels
        hidden_dim: Hidden dimension for feature extraction
        output_dim: Output HI dimension (default 1 for scalar HI)
    """

    def __init__(
        self,
        sensor_dim: int = 14,
        hidden_dim: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Temporal aggregation (attention-based)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            batch_first=True
        )

        # HI prediction head
        self.hi_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Sigmoid()  # HI in [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute health indicator from sensor data.

        Args:
            x: Sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            hi: Health indicator [batch, seq_len, output_dim] or [batch, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Extract features per time step
        features = self.feature_extractor(x)  # [batch, seq_len, hidden_dim//2]

        # Temporal attention for context
        attended_features, _ = self.temporal_attention(
            features, features, features,
            key_padding_mask=~mask if mask is not None else None
        )

        # Compute HI per time step
        hi = self.hi_head(attended_features)  # [batch, seq_len, output_dim]

        return hi

    def compute_stage_probabilities(
        self,
        hi: torch.Tensor,
        num_stages: int = 3,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Convert health indicator to stage probabilities.

        Args:
            hi: Health indicator [batch, seq_len, 1] or [batch, 1]
            num_stages: Number of degradation stages
            temperature: Softmax temperature

        Returns:
            stage_probs: Stage probabilities [batch, seq_len, num_stages]
        """
        if hi.dim() == 2:
            hi = hi.unsqueeze(-1)

        # Create stage boundaries (equally spaced)
        boundaries = torch.linspace(0, 1, num_stages + 1, device=hi.device)

        # Compute distances to each stage center
        stage_centers = (boundaries[1:] + boundaries[:-1]) / 2
        distances = torch.abs(hi - stage_centers.view(1, 1, -1))

        # Convert to probabilities (lower distance = higher probability)
        stage_logits = -distances / temperature
        stage_probs = F.softmax(stage_logits, dim=-1)

        return stage_probs

    def get_current_stage(
        self,
        x: torch.Tensor,
        num_stages: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current degradation stage from sensor data.

        Args:
            x: Sensor data [batch, seq_len, sensor_dim]
            num_stages: Number of degradation stages

        Returns:
            stage_id: Most likely stage [batch]
            stage_probs: Stage probabilities [batch, num_stages]
        """
        # Compute HI (use last time step)
        hi = self.forward(x)  # [batch, seq_len, 1]
        hi_last = hi[:, -1:, :]  # [batch, 1, 1]

        # Convert to stage probabilities
        stage_probs = self.compute_stage_probabilities(hi_last, num_stages)
        stage_probs = stage_probs.squeeze(1)  # [batch, num_stages]

        # Get most likely stage
        stage_id = torch.argmax(stage_probs, dim=1)

        return stage_id, stage_probs


class SelfSupervisedHI(nn.Module):
    """
    Self-supervised health indicator learning.

    Uses contrastive learning to learn HI representations without labels.
    """

    def __init__(
        self,
        sensor_dim: int = 14,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Prediction head for HI
        self.hi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Sensor data [batch * seq_len, sensor_dim] (flattened)

        Returns:
            hi: Health indicator [batch * seq_len, 1]
            projection: Projected features [batch * seq_len, projection_dim]
        """
        features = self.encoder(x)
        projection = self.projection(features)
        hi = self.hi_head(features)
        return hi, projection

    def contrastive_loss(
        self,
        projections: torch.Tensor,
        time_indices: torch.Tensor,
        window_size: int = 5
    ) -> torch.Tensor:
        """
        Compute temporal contrastive loss.

        Nearby time points should have similar representations;
        distant time points should have different representations.

        Args:
            projections: Projected features [N, projection_dim]
            time_indices: Time indices for each sample [N]
            window_size: Window for positive pairs

        Returns:
            loss: Contrastive loss
        """
        # Normalize projections
        projections = F.normalize(projections, dim=1)

        # Compute similarity matrix
        similarity_matrix = projections @ projections.T  # [N, N]

        # Create positive/negative mask based on time distance
        time_diff = torch.abs(time_indices.unsqueeze(1) - time_indices.unsqueeze(0))
        positive_mask = (time_diff <= window_size) & (time_diff > 0)
        negative_mask = time_diff > window_size

        # InfoNCE loss
        exp_sim = torch.exp(similarity_matrix / self.temperature)

        # Positive pairs sum
        positive_sum = (exp_sim * positive_mask.float()).sum(dim=1)

        # All pairs sum (excluding self)
        all_sum = (exp_sim * (1 - torch.eye(len(projections), device=projections.device))).sum(dim=1)

        # Loss (negative log likelihood of positive pairs)
        loss = -torch.log(positive_sum / (all_sum + 1e-8) + 1e-8)

        return loss.mean()


def compute_monotonicity_score(rul_sequence: torch.Tensor) -> float:
    """
    Compute monotonicity score of RUL sequence.

    Score = proportion of consecutive pairs that are non-increasing
    (since RUL should decrease over time).

    Args:
        rul_sequence: RUL values [seq_len] or [batch, seq_len]

    Returns:
        score: Monotonicity score in [0, 1]
    """
    if rul_sequence.dim() == 1:
        rul_sequence = rul_sequence.unsqueeze(0)

    # Compute differences
    diffs = rul_sequence[:, 1:] - rul_sequence[:, :-1]

    # Count non-increasing pairs (diff <= 0 allows for plateaus)
    monotonic = (diffs <= 0).float().mean(dim=1)

    return monotonic.mean().item()


def compute_trend_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalize: bool = True
) -> float:
    """
    Compute trend similarity between predicted and target RUL sequences.

    Uses normalized cross-correlation of first differences.

    Args:
        pred: Predicted RUL [seq_len] or [batch, seq_len]
        target: Target RUL [seq_len] or [batch, seq_len]
        normalize: Whether to normalize correlation

    Returns:
        similarity: Trend similarity score
    """
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    # Compute first differences (trends)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]

    # Normalize
    if normalize:
        pred_diff = pred_diff / (pred_diff.std(dim=1, keepdim=True) + 1e-8)
        target_diff = target_diff / (target_diff.std(dim=1, keepdim=True) + 1e-8)

    # Compute correlation
    batch_size = pred.shape[0]
    correlations = []
    for i in range(batch_size):
        corr = torch.corrcoef(torch.stack([pred_diff[i], target_diff[i]]))[0, 1]
        correlations.append(corr if not torch.isnan(corr) else torch.tensor(0.0))

    return torch.stack(correlations).mean().item()
