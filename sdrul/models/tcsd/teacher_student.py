"""
Teacher and Student branches for TCSD.

Implements FiLM-based condition injection and the teacher-student
architecture for trajectory-conditioned self-distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Applies affine transformation conditioned on external input:
    output = gamma * features + beta

    where gamma and beta are generated from the conditioning input.

    Args:
        feature_dim: Dimension of features to modulate
        condition_dim: Dimension of conditioning input
        hidden_dim: Hidden dimension for gamma/beta generation (default: None, uses feature_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        if hidden_dim is None:
            hidden_dim = feature_dim

        # Gamma generator (scale)
        self.gamma_gen = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Beta generator (shift)
        self.beta_gen = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Initialize to identity transformation
        nn.init.zeros_(self.gamma_gen[-1].weight)
        nn.init.ones_(self.gamma_gen[-1].bias)
        nn.init.zeros_(self.beta_gen[-1].weight)
        nn.init.zeros_(self.beta_gen[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            features: Input features [batch, seq_len, feature_dim] or [batch, feature_dim]
            condition: Conditioning input [batch, condition_dim]

        Returns:
            modulated: Modulated features, same shape as input
        """
        # Generate gamma and beta
        gamma = self.gamma_gen(condition)  # [batch, feature_dim]
        beta = self.beta_gen(condition)    # [batch, feature_dim]

        # Handle sequence dimension
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)  # [batch, 1, feature_dim]
            beta = beta.unsqueeze(1)    # [batch, 1, feature_dim]

        # Apply modulation
        return gamma * features + beta


class TeacherBranch(nn.Module):
    """
    Teacher branch for TCSD.

    Conditioned on trajectory prototypes via FiLM layers.
    Shares feature extractor with student but has additional
    conditioning mechanism.

    Args:
        feature_extractor: Shared feature extractor (TransformerFeatureExtractor)
        rul_head: RUL prediction head (GaussianRULHead)
        prototype_dim: Dimension of prototype embeddings (default: 128)
        d_model: Feature dimension (default: 256)
        num_film_layers: Number of FiLM layers (default: 2)
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        rul_head: nn.Module,
        prototype_dim: int = 128,
        d_model: int = 256,
        num_film_layers: int = 2,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rul_head = rul_head
        self.prototype_dim = prototype_dim
        self.d_model = d_model

        # FiLM layers for prototype conditioning
        self.film_layers = nn.ModuleList([
            FiLMLayer(d_model, prototype_dim)
            for _ in range(num_film_layers)
        ])

        # Layer norms between FiLM layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_film_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        prototype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with prototype conditioning.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            prototype: Prototype embedding [batch, prototype_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            mu: Predicted RUL mean [batch]
            sigma: Predicted RUL std [batch]
        """
        # Extract features (shared with student)
        features = self.feature_extractor(x, mask)  # [batch, seq_len, d_model]

        # Apply FiLM conditioning
        for film, norm in zip(self.film_layers, self.layer_norms):
            features = film(features, prototype)
            features = norm(features)

        # Predict RUL distribution
        mu, sigma = self.rul_head(features)

        return mu, sigma

    def get_conditioned_features(
        self,
        x: torch.Tensor,
        prototype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get conditioned features without RUL prediction.

        Useful for analysis and visualization.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            prototype: Prototype embedding [batch, prototype_dim]
            mask: Optional attention mask

        Returns:
            features: Conditioned features [batch, seq_len, d_model]
        """
        features = self.feature_extractor(x, mask)

        for film, norm in zip(self.film_layers, self.layer_norms):
            features = film(features, prototype)
            features = norm(features)

        return features


class StudentBranch(nn.Module):
    """
    Student branch for TCSD.

    Shares feature extractor AND RUL head with teacher.
    Makes predictions without prototype conditioning.
    Learns to match teacher's conditioned predictions through distillation.

    Optionally integrates with DSA-MoE for multi-condition knowledge organization.

    Args:
        feature_extractor: Shared feature extractor (same instance as teacher)
        rul_head: Shared RUL prediction head (same instance as teacher)
        moe: Optional shared MoE module (same instance as RULPredictionModel)
        d_model: Feature dimension (default: 256)
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        rul_head: nn.Module,
        moe: Optional[nn.Module] = None,
        d_model: int = 256,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor  # Shared with teacher
        self.rul_head = rul_head  # Shared with teacher
        self.moe = moe  # Shared with RULPredictionModel
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass without conditioning.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            condition_id: Optional operating condition IDs [batch] (for MoE routing)
            mask: Optional attention mask [batch, seq_len]

        Returns:
            mu: Predicted RUL mean [batch]
            sigma: Predicted RUL std [batch]
        """
        # Extract features (shared with teacher)
        features = self.feature_extractor(x, mask)  # [batch, seq_len, d_model]

        if self.moe is not None:
            # Apply MoE transformation
            moe_output, _ = self.moe(features, x, condition_id, mask)
            mu, sigma = self.rul_head(moe_output)
        else:
            # Direct prediction without MoE
            mu, sigma = self.rul_head(features)

        return mu, sigma

    def get_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get student features without RUL prediction.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask

        Returns:
            features: Student features [batch, seq_len, d_model]
        """
        return self.feature_extractor(x, mask)


class TeacherStudentPair(nn.Module):
    """
    Combined teacher-student pair for TCSD.

    Manages shared components and provides unified interface.
    Both teacher and student share the same feature_extractor and rul_head.
    The only difference is teacher applies FiLM conditioning.

    Args:
        feature_extractor: Shared feature extractor (can be external for sharing with DSA-MoE)
        moe: Optional shared MoE module (for student branch integration)
        sensor_dim: Input sensor dimension (default: 14, only used if feature_extractor is None)
        d_model: Feature dimension (default: 256)
        prototype_dim: Prototype embedding dimension (default: 128)
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        moe: Optional[nn.Module] = None,
        sensor_dim: int = 14,
        d_model: int = 256,
        prototype_dim: int = 128,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.d_model = d_model
        self.prototype_dim = prototype_dim

        # Import here to avoid circular imports
        from models.encoders import TransformerFeatureExtractor, GaussianRULHead

        # Shared feature extractor (can be passed from outside for sharing with DSA-MoE)
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = TransformerFeatureExtractor(
                sensor_dim=sensor_dim,
                d_model=d_model,
            )

        # Separate RUL heads for teacher and student to avoid gradient conflicts
        # Teacher head: learns to predict with prototype conditioning
        self.teacher_rul_head = GaussianRULHead(d_model=d_model)
        # Student head: learns to match teacher without conditioning
        self.student_rul_head = GaussianRULHead(d_model=d_model)

        # Teacher and student branches (share feature_extractor but separate RUL heads)
        self.teacher = TeacherBranch(
            feature_extractor=self.feature_extractor,
            rul_head=self.teacher_rul_head,
            prototype_dim=prototype_dim,
            d_model=d_model,
        )

        self.student = StudentBranch(
            feature_extractor=self.feature_extractor,
            rul_head=self.student_rul_head,
            moe=moe,  # Pass shared MoE to student
            d_model=d_model,
        )

        # For backward compatibility
        self.rul_head = self.student_rul_head

    def forward_teacher(
        self,
        x: torch.Tensor,
        prototype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through teacher with prototype conditioning."""
        return self.teacher(x, prototype, mask)

    def forward_student(
        self,
        x: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through student without conditioning."""
        return self.student(x, condition_id, mask)

    def forward(
        self,
        x: torch.Tensor,
        prototype: Optional[torch.Tensor] = None,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward through both teacher and student.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            prototype: Prototype embedding [batch, prototype_dim] (required for teacher)
            condition_id: Optional operating condition IDs [batch] (for student MoE routing)
            mask: Optional attention mask

        Returns:
            teacher_output: (mu_t, sigma_t)
            student_output: (mu_s, sigma_s)
        """
        student_output = self.student(x, condition_id, mask)

        if prototype is not None:
            teacher_output = self.teacher(x, prototype, mask)
        else:
            # If no prototype, teacher output equals student output
            teacher_output = student_output

        return teacher_output, student_output
