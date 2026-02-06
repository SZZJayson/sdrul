"""
Routing modules for DSA-MoE.

Provides condition-based and degradation-stage-based routing for
directing inputs to appropriate experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.health_indicator import HealthIndicatorComputer


class ConditionRouter(nn.Module):
    """
    Router for operating condition-based expert selection.

    Supports both hard routing (when condition_id is known) and
    soft routing (learned classification when condition is unknown).

    Args:
        d_model: Input feature dimension (default: 256)
        num_conditions: Number of operating conditions (default: 6)
        temperature: Softmax temperature for soft routing (default: 1.0)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_conditions: int = 6,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.temperature = temperature

        # Classifier for soft routing
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_conditions),
        )

    def forward(
        self,
        features: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute routing weights for conditions.

        Args:
            features: Input features [batch, seq_len, d_model] or [batch, d_model]
            condition_id: Optional known condition IDs [batch]

        Returns:
            weights: Condition routing weights [batch, num_conditions]
        """
        batch_size = features.size(0)

        if condition_id is not None:
            # Hard routing: one-hot encoding
            weights = F.one_hot(
                condition_id.long(),
                num_classes=self.num_conditions
            ).float()
        else:
            # Soft routing: learned classification
            if features.dim() == 3:
                # Pool sequence features (use mean)
                features = features.mean(dim=1)  # [batch, d_model]

            logits = self.classifier(features)  # [batch, num_conditions]
            weights = F.softmax(logits / self.temperature, dim=-1)

        return weights

    def predict_condition(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict operating condition from features.

        Args:
            features: Input features [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            condition_id: Predicted condition [batch]
            probs: Condition probabilities [batch, num_conditions]
        """
        if features.dim() == 3:
            features = features.mean(dim=1)

        logits = self.classifier(features)
        probs = F.softmax(logits / self.temperature, dim=-1)
        condition_id = torch.argmax(probs, dim=-1)

        return condition_id, probs


class StageRouter(nn.Module):
    """
    Router for degradation-stage-based expert selection.

    Wraps HealthIndicatorComputer to convert raw sensor data
    into stage routing weights.

    Args:
        sensor_dim: Number of sensor channels (default: 14)
        hidden_dim: Hidden dimension for HI computation (default: 128)
        num_stages: Number of degradation stages (default: 3)
        temperature: Softmax temperature (default: 1.0)
        hi_computer: Optional pre-trained HealthIndicatorComputer
    """

    def __init__(
        self,
        sensor_dim: int = 14,
        hidden_dim: int = 128,
        num_stages: int = 3,
        temperature: float = 1.0,
        hi_computer: Optional[HealthIndicatorComputer] = None,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.temperature = temperature

        # Use provided HI computer or create new one
        if hi_computer is not None:
            self.hi_computer = hi_computer
        else:
            self.hi_computer = HealthIndicatorComputer(
                sensor_dim=sensor_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute routing weights based on degradation stage.

        Args:
            x: Raw sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            weights: Stage routing weights [batch, num_stages]
        """
        # Compute health indicator
        hi = self.hi_computer(x, mask)  # [batch, seq_len, 1]

        # Use last time step HI
        hi_last = hi[:, -1, :]  # [batch, 1]

        # Convert to stage probabilities
        stage_probs = self.hi_computer.compute_stage_probabilities(
            hi_last.unsqueeze(1),  # [batch, 1, 1]
            num_stages=self.num_stages,
            temperature=self.temperature,
        )

        return stage_probs.squeeze(1)  # [batch, num_stages]

    def get_stage(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current degradation stage.

        Args:
            x: Raw sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask

        Returns:
            stage_id: Most likely stage [batch]
            stage_probs: Stage probabilities [batch, num_stages]
        """
        stage_probs = self.forward(x, mask)
        stage_id = torch.argmax(stage_probs, dim=-1)
        return stage_id, stage_probs

    def get_health_indicator(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the raw health indicator values.

        Args:
            x: Raw sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask

        Returns:
            hi: Health indicator [batch, seq_len, 1]
        """
        return self.hi_computer(x, mask)


class JointRouter(nn.Module):
    """
    Joint router combining condition and stage routing.

    Produces a 2D routing weight matrix for the expert grid.

    Args:
        d_model: Feature dimension for condition routing
        sensor_dim: Sensor dimension for stage routing
        num_conditions: Number of operating conditions
        num_stages: Number of degradation stages
        temperature: Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 256,
        sensor_dim: int = 14,
        num_conditions: int = 6,
        num_stages: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_conditions = num_conditions
        self.num_stages = num_stages

        self.condition_router = ConditionRouter(
            d_model=d_model,
            num_conditions=num_conditions,
            temperature=temperature,
        )

        self.stage_router = StageRouter(
            sensor_dim=sensor_dim,
            num_stages=num_stages,
            temperature=temperature,
        )

    def forward(
        self,
        features: torch.Tensor,
        x_raw: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute joint routing weights.

        Args:
            features: Extracted features [batch, seq_len, d_model]
            x_raw: Raw sensor data [batch, seq_len, sensor_dim]
            condition_id: Optional known condition IDs [batch]
            mask: Optional attention mask

        Returns:
            joint_weights: Combined weights [batch, num_conditions, num_stages]
            cond_weights: Condition weights [batch, num_conditions]
            stage_weights: Stage weights [batch, num_stages]
        """
        # Get individual routing weights
        cond_weights = self.condition_router(features, condition_id)
        stage_weights = self.stage_router(x_raw, mask)

        # Compute joint weights via outer product
        # [batch, num_conditions, 1] * [batch, 1, num_stages]
        joint_weights = cond_weights.unsqueeze(-1) * stage_weights.unsqueeze(1)

        return joint_weights, cond_weights, stage_weights
