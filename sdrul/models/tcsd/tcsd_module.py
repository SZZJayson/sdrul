"""
TCSD (Trajectory-Conditioned Self-Distillation) main module.

Integrates prototype management, teacher-student architecture,
and distillation loss for continual RUL prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass

from .prototype_manager import TrajectoryPrototypeManager
from .teacher_student import TeacherBranch, StudentBranch, TeacherStudentPair
from .distillation_loss import DistributionalRULLoss, DistillationWithSupervisionLoss

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.trajectory import TrajectoryShapeEncoder

if TYPE_CHECKING:
    from data.config import DatasetConfig


@dataclass
class TCSDConfig:
    """
    Configuration for TCSD module.

    Note: These are research-validated hyperparameters.
    Do not change without re-validation.
    """

    # Model architecture
    sensor_dim: int = 14
    d_model: int = 256
    prototype_dim: int = 128  # Research-specified value

    # Prototype management (research-specified values)
    num_prototypes_per_condition: int = 5
    num_conditions: int = 6
    ema_decay: float = 0.9

    # Distillation (research-specified values)
    alpha_kl: float = 0.5
    distillation_weight: float = 1.0

    # Training
    warmup_distillation: bool = True
    warmup_epochs: int = 5

    @classmethod
    def from_dataset_config(
        cls,
        dataset_config: 'DatasetConfig',
        **kwargs,
    ) -> 'TCSDConfig':
        """
        Create TCSDConfig from a DatasetConfig.

        Args:
            dataset_config: Dataset configuration
            **kwargs: Additional config overrides

        Returns:
            config: TCSDConfig instance
        """
        return cls(
            sensor_dim=dataset_config.sensor_dim,
            num_conditions=dataset_config.num_conditions,
            **kwargs,
        )


class TCSD(nn.Module):
    """
    Trajectory-Conditioned Self-Distillation module.

    Implements self-distillation for RUL prediction where:
    - Teacher: conditioned on trajectory prototypes
    - Student: learns to match teacher without explicit conditioning

    This enables knowledge transfer from prototype-conditioned predictions
    to unconditional predictions, supporting continual learning.

    Args:
        config: TCSDConfig instance
        feature_extractor: Optional shared feature extractor (for sharing with DSA-MoE)
        moe: Optional shared MoE module (for student branch integration)
    """

    def __init__(
        self,
        config: TCSDConfig,
        feature_extractor: Optional[nn.Module] = None,
        moe: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # Trajectory encoder for prototype management
        self.trajectory_encoder = TrajectoryShapeEncoder(
            feature_dim=config.prototype_dim,
        )

        # Prototype manager (extends PrototypeBuffer)
        self.prototype_manager = TrajectoryPrototypeManager(
            trajectory_encoder=self.trajectory_encoder,
            num_prototypes_per_condition=config.num_prototypes_per_condition,
            num_conditions=config.num_conditions,
            ema_decay=config.ema_decay,
        )

        # Teacher-student pair (can share feature_extractor and MoE with DSA-MoE)
        self.ts_pair = TeacherStudentPair(
            feature_extractor=feature_extractor,
            moe=moe,  # Pass shared MoE to student
            sensor_dim=config.sensor_dim,
            d_model=config.d_model,
            prototype_dim=config.prototype_dim,
        )

        # Distillation loss
        self.distillation_loss = DistributionalRULLoss(
            alpha=config.alpha_kl,
        )

        # Combined loss for supervised + distillation
        self.combined_loss = DistillationWithSupervisionLoss(
            distillation_weight=config.distillation_weight,
            alpha_kl=config.alpha_kl,
        )

        # Training state
        self.register_buffer('_current_epoch', torch.tensor(0))
        self.register_buffer('_distillation_weight', torch.tensor(config.distillation_weight))

        # Default prototype for fallback when no prototypes available
        self.register_buffer('_default_prototype', torch.randn(config.prototype_dim) * 0.1)

    @property
    def feature_extractor(self) -> nn.Module:
        """Access shared feature extractor."""
        return self.ts_pair.feature_extractor

    @property
    def teacher(self) -> TeacherBranch:
        """Access teacher branch."""
        return self.ts_pair.teacher

    @property
    def student(self) -> StudentBranch:
        """Access student branch."""
        return self.ts_pair.student

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self._current_epoch.fill_(epoch)

        if self.config.warmup_distillation:
            # Gradually increase distillation weight
            warmup_factor = min(1.0, epoch / self.config.warmup_epochs)
            self._distillation_weight.fill_(
                self.config.distillation_weight * warmup_factor
            )

    def add_trajectory(
        self,
        trajectory: torch.Tensor,
        condition_id: int,
    ) -> bool:
        """
        Add a trajectory to prototype manager.

        Args:
            trajectory: RUL trajectory [seq_len]
            condition_id: Operating condition ID

        Returns:
            added: Whether trajectory was added
        """
        return self.prototype_manager.add_trajectory(trajectory, condition_id)

    def initialize_prototypes(
        self,
        trajectories: List[torch.Tensor],
        condition_id: int,
    ) -> bool:
        """
        Initialize prototypes for a condition using K-means.

        Args:
            trajectories: List of RUL trajectories
            condition_id: Operating condition ID

        Returns:
            success: Whether initialization was successful
        """
        device = next(self.parameters()).device
        return self.prototype_manager.fit_kmeans(trajectories, condition_id, device)

    def on_policy_step(
        self,
        x: torch.Tensor,
        condition_id: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        On-policy training step for self-distillation.

        Uses random prototype from the condition to condition teacher,
        then distills to student.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            condition_id: Operating condition IDs [batch]
            mask: Optional attention mask

        Returns:
            loss: Distillation loss
            outputs: Dictionary with predictions and metrics
        """
        batch_size = x.size(0)
        device = x.device

        # Get prototypes for each sample
        prototypes = []
        valid_mask = []

        for i in range(batch_size):
            cond = condition_id[i].item()
            proto = self.prototype_manager.get_random_prototype(int(cond))

            if proto is not None:
                prototypes.append(proto.to(device))
                valid_mask.append(True)
            else:
                # Use mean prototype from any available condition as fallback
                fallback_proto = self.prototype_manager.get_mean_prototype()
                if fallback_proto is not None:
                    prototypes.append(fallback_proto.to(device))
                    valid_mask.append(True)  # Still valid with fallback
                else:
                    # Last resort: use learned default prototype
                    prototypes.append(self._default_prototype.to(device))
                    valid_mask.append(True)  # Enable distillation with default

        prototypes = torch.stack(prototypes)  # [batch, prototype_dim]
        valid_mask = torch.tensor(valid_mask, device=device)

        # Forward through teacher and student
        teacher_output = self.teacher(x, prototypes, mask)
        student_output = self.student(x, condition_id, mask)  # Pass condition_id for MoE routing

        # Compute distillation loss - always compute if we have valid prototypes
        if valid_mask.any():
            # Filter to valid samples
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            teacher_valid = (
                teacher_output[0][valid_indices],
                teacher_output[1][valid_indices],
            )
            student_valid = (
                student_output[0][valid_indices],
                student_output[1][valid_indices],
            )

            loss = self.distillation_loss(teacher_valid, student_valid)
            # Ensure minimum distillation weight to prevent zero loss
            effective_weight = max(self._distillation_weight.item(), 0.1)
            loss = loss * effective_weight
        else:
            loss = torch.tensor(0.0, device=device)

        outputs = {
            'teacher_mu': teacher_output[0],
            'teacher_sigma': teacher_output[1],
            'student_mu': student_output[0],
            'student_sigma': student_output[1],
            'valid_samples': valid_mask.sum().item(),
            'distillation_weight': self._distillation_weight.item(),
        }

        return loss, outputs

    def supervised_step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        condition_id: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Supervised training step with distillation.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            target: Target RUL values [batch]
            condition_id: Operating condition IDs [batch]
            mask: Optional attention mask

        Returns:
            loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        batch_size = x.size(0)
        device = x.device

        # Get prototypes
        prototypes = []
        for i in range(batch_size):
            cond = condition_id[i].item()
            proto = self.prototype_manager.get_random_prototype(int(cond))
            if proto is not None:
                prototypes.append(proto.to(device))
            else:
                prototypes.append(torch.zeros(self.config.prototype_dim, device=device))

        prototypes = torch.stack(prototypes)

        # Forward
        teacher_output = self.teacher(x, prototypes, mask)
        student_output = self.student(x, condition_id, mask)  # Pass condition_id for MoE routing

        # Combined loss
        loss, loss_dict = self.combined_loss(
            student_output,
            teacher_output,
            target,
        )

        # Scale distillation component
        loss_dict['distillation_weight'] = self._distillation_weight.item()

        return loss, loss_dict

    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_teacher: bool = False,
        condition_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make RUL predictions.

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask
            use_teacher: Whether to use teacher (requires condition_id)
            condition_id: Operating condition IDs (required if use_teacher=True)

        Returns:
            mu: Predicted RUL mean [batch]
            sigma: Predicted RUL std [batch]
        """
        if use_teacher and condition_id is not None:
            batch_size = x.size(0)
            device = x.device

            prototypes = []
            for i in range(batch_size):
                cond = condition_id[i].item()
                proto = self.prototype_manager.get_random_prototype(int(cond))
                if proto is not None:
                    prototypes.append(proto.to(device))
                else:
                    prototypes.append(torch.zeros(self.config.prototype_dim, device=device))

            prototypes = torch.stack(prototypes)
            return self.teacher(x, prototypes, mask)
        else:
            return self.student(x, condition_id, mask)  # Pass condition_id for MoE routing

    def forward(
        self,
        x: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Default forward pass (student prediction).

        Args:
            x: Input sensor data [batch, seq_len, sensor_dim]
            condition_id: Optional operating condition IDs [batch] (for MoE routing)
            mask: Optional attention mask

        Returns:
            mu: Predicted RUL mean [batch]
            sigma: Predicted RUL std [batch]
        """
        return self.student(x, condition_id, mask)

    def get_prototype_stats(self) -> Dict[str, any]:
        """Get statistics about prototype manager."""
        stats = {
            'num_conditions': self.prototype_manager.num_conditions,
            'prototypes_per_condition': {},
            'initialized_conditions': [],
        }

        for cond_id in range(self.prototype_manager.num_conditions):
            n_protos = self.prototype_manager.num_prototypes_for_condition(cond_id)
            stats['prototypes_per_condition'][cond_id] = n_protos
            if self.prototype_manager.is_initialized(cond_id):
                stats['initialized_conditions'].append(cond_id)

        return stats

    def save_prototypes(self) -> Dict:
        """Save prototype manager state."""
        return self.prototype_manager.save_state()

    def load_prototypes(self, state: Dict, device: Optional[torch.device] = None):
        """Load prototype manager state."""
        if device is None:
            device = next(self.parameters()).device
        self.prototype_manager.load_state(state, device)
