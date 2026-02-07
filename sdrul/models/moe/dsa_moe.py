"""
Degradation-Stage-Aware Mixture of Experts (DSA-MoE) module.

Implements a 2D expert matrix organized by operating conditions and
degradation stages for effective multi-condition knowledge organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .expert import AdapterExpert
from .router import ConditionRouter, StageRouter, JointRouter
from .losses import load_balance_loss

if TYPE_CHECKING:
    from data.config import DatasetConfig


@dataclass
class DSAMoEConfig:
    """Configuration for DSA-MoE module."""

    # Expert configuration
    num_conditions: int = 6
    num_stages: int = 3
    d_model: int = 256
    bottleneck_dim: int = 64

    # Router configuration
    sensor_dim: int = 14
    temperature: float = 1.0

    # Training configuration
    load_balance_weight: float = 0.01
    dropout: float = 0.1

    # Warmup configuration
    warmup_epochs: int = 5
    warmup_random_routing: bool = True

    @classmethod
    def from_dataset_config(
        cls,
        dataset_config: 'DatasetConfig',
        num_stages: int = 3,
        **kwargs,
    ) -> 'DSAMoEConfig':
        """
        Create DSAMoEConfig from a DatasetConfig.

        Args:
            dataset_config: Dataset configuration
            num_stages: Number of degradation stages
            **kwargs: Additional config overrides

        Returns:
            config: DSAMoEConfig instance
        """
        return cls(
            sensor_dim=dataset_config.sensor_dim,
            num_conditions=dataset_config.num_conditions,
            num_stages=num_stages,
            **kwargs,
        )


class DSAMoE(nn.Module):
    """
    Degradation-Stage-Aware Mixture of Experts.

    Organizes experts in a 2D matrix: [num_conditions] x [num_stages]
    Each expert specializes in a specific (condition, stage) combination.

    Features:
    - Hard routing when condition_id is known
    - Soft routing when condition is unknown
    - Stage routing based on health indicator
    - Load balancing to prevent expert collapse

    Args:
        config: DSAMoEConfig instance
    """

    def __init__(self, config: DSAMoEConfig):
        super().__init__()
        self.config = config
        self.num_conditions = config.num_conditions
        self.num_stages = config.num_stages
        self.num_experts = config.num_conditions * config.num_stages

        # Create 2D expert matrix
        # Stored as flat ModuleList, indexed as [cond_idx * num_stages + stage_idx]
        self.experts = nn.ModuleList([
            AdapterExpert(
                d_model=config.d_model,
                bottleneck_dim=config.bottleneck_dim,
                dropout=config.dropout,
            )
            for _ in range(self.num_experts)
        ])

        # Routers
        self.condition_router = ConditionRouter(
            d_model=config.d_model,
            num_conditions=config.num_conditions,
            temperature=config.temperature,
        )

        self.stage_router = StageRouter(
            sensor_dim=config.sensor_dim,
            num_stages=config.num_stages,
            temperature=config.temperature,
        )

        # Training state
        self.register_buffer('_warmup_mode', torch.tensor(False))
        self.register_buffer('_current_epoch', torch.tensor(0))

    def get_expert(self, cond_idx: int, stage_idx: int) -> AdapterExpert:
        """Get expert for specific (condition, stage) combination."""
        flat_idx = cond_idx * self.num_stages + stage_idx
        return self.experts[flat_idx]

    def set_warmup_mode(self, enabled: bool):
        """Enable/disable warmup mode with random routing."""
        self._warmup_mode.fill_(enabled)

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self._current_epoch.fill_(epoch)
        if epoch >= self.config.warmup_epochs:
            self.set_warmup_mode(False)

    def forward(
        self,
        features: torch.Tensor,
        x_raw: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply DSA-MoE transformation.

        Args:
            features: Extracted features [batch, seq_len, d_model]
            x_raw: Raw sensor data [batch, seq_len, sensor_dim]
            condition_id: Optional known condition IDs [batch]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            output: Transformed features [batch, seq_len, d_model]
            aux_outputs: Dictionary containing:
                - cond_weights: Condition routing weights [batch, num_conditions]
                - stage_weights: Stage routing weights [batch, num_stages]
                - expert_loads: Expert load distribution [num_experts]
                - load_balance_loss: Load balancing loss (scalar)
        """
        batch_size = features.size(0)

        # Get routing weights
        if self._warmup_mode.item() and self.config.warmup_random_routing:
            # Random routing during warmup
            cond_weights = torch.rand(batch_size, self.num_conditions, device=features.device)
            cond_weights = F.softmax(cond_weights, dim=-1)
            stage_weights = torch.rand(batch_size, self.num_stages, device=features.device)
            stage_weights = F.softmax(stage_weights, dim=-1)
        else:
            # Normal routing
            cond_weights = self.condition_router(features, condition_id)
            stage_weights = self.stage_router(x_raw, mask)

        # Compute joint weights [batch, num_conditions, num_stages]
        joint_weights = cond_weights.unsqueeze(-1) * stage_weights.unsqueeze(1)

        # Flatten to [batch, num_experts]
        flat_weights = joint_weights.view(batch_size, -1)

        # Compute weighted expert outputs
        output = torch.zeros_like(features)
        for cond_idx in range(self.num_conditions):
            for stage_idx in range(self.num_stages):
                expert = self.get_expert(cond_idx, stage_idx)
                expert_out = expert(features)

                # Weight for this expert
                weight = joint_weights[:, cond_idx, stage_idx]  # [batch]
                weight = weight.view(-1, 1, 1)  # [batch, 1, 1]

                output = output + weight * expert_out

        # Compute auxiliary outputs
        expert_loads = flat_weights.mean(dim=0)  # [num_experts]
        lb_loss = load_balance_loss(flat_weights) * self.config.load_balance_weight

        aux_outputs = {
            'cond_weights': cond_weights,
            'stage_weights': stage_weights,
            'expert_loads': expert_loads,
            'load_balance_loss': lb_loss,
            'joint_weights': joint_weights,
        }

        return output, aux_outputs

    def forward_single_expert(
        self,
        features: torch.Tensor,
        cond_idx: int,
        stage_idx: int,
    ) -> torch.Tensor:
        """
        Apply a single expert (for debugging/analysis).

        Args:
            features: Input features [batch, seq_len, d_model]
            cond_idx: Condition index
            stage_idx: Stage index

        Returns:
            output: Expert output [batch, seq_len, d_model]
        """
        expert = self.get_expert(cond_idx, stage_idx)
        return expert(features)

    def get_expert_utilization(self) -> torch.Tensor:
        """
        Get expert utilization statistics.

        Returns:
            utilization: Expert utilization [num_conditions, num_stages]
        """
        # This would be computed from training statistics
        # Placeholder returning uniform distribution
        return torch.ones(self.num_conditions, self.num_stages) / self.num_experts

    def prune_unused_experts(self, threshold: float = 0.01):
        """
        Identify experts with low utilization for potential pruning.

        Args:
            threshold: Utilization threshold below which experts are considered unused

        Returns:
            unused_experts: List of (cond_idx, stage_idx) tuples for unused experts
        """
        utilization = self.get_expert_utilization()
        unused = []
        for cond_idx in range(self.num_conditions):
            for stage_idx in range(self.num_stages):
                if utilization[cond_idx, stage_idx] < threshold:
                    unused.append((cond_idx, stage_idx))
        return unused

    def expand_for_new_condition(
        self,
        source_condition_id: int,
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Expand expert matrix for a new operating condition.

        Clones experts from the most similar existing condition as initialization.

        Args:
            source_condition_id: ID of condition to clone experts from
            device: Device for new experts (defaults to existing experts' device)

        Returns:
            new_condition_id: ID assigned to the new condition
        """
        new_condition_id = self.num_conditions

        # Determine device and dtype from existing experts
        if device is None:
            device = next(self.experts.parameters()).device
        dtype = next(self.experts.parameters()).dtype

        # Create new experts for all stages of the new condition
        new_experts = nn.ModuleList([
            AdapterExpert(
                d_model=self.config.d_model,
                bottleneck_dim=self.config.bottleneck_dim,
                dropout=self.config.dropout,
            )
            for _ in range(self.num_stages)
        ])

        # Move new experts to correct device and dtype before cloning
        for expert in new_experts:
            expert.to(device=device, dtype=dtype)

        # Clone weights from source condition
        self._clone_expert_row(source_condition_id, new_experts)

        # Add new experts to the module list
        for expert in new_experts:
            self.experts.append(expert)

        # Expand condition router
        self.condition_router.expand_classifier(
            new_num_conditions=self.num_conditions + 1,
            source_condition_id=source_condition_id,
        )

        # Update counts
        self.num_conditions += 1
        self.num_experts = self.num_conditions * self.num_stages

        return new_condition_id

    def _clone_expert_row(
        self,
        source_cond_idx: int,
        target_experts: nn.ModuleList,
    ):
        """
        Clone weights from source condition's experts to target experts.

        Args:
            source_cond_idx: Source condition index to clone from
            target_experts: ModuleList of new experts to initialize
        """
        with torch.no_grad():
            for stage_idx in range(self.num_stages):
                source_expert = self.get_expert(source_cond_idx, stage_idx)
                target_expert = target_experts[stage_idx]

                # Copy all parameters
                for (name, src_param), (_, tgt_param) in zip(
                    source_expert.named_parameters(),
                    target_expert.named_parameters()
                ):
                    tgt_param.copy_(src_param)

    def detect_new_condition(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[bool, int, float]:
        """
        Detect whether input represents a new operating condition.

        Delegates to condition router's detection method.

        Args:
            features: Input features [batch, seq_len, d_model]
            threshold: Confidence threshold

        Returns:
            is_new: Whether this appears to be a new condition
            predicted_condition: ID of most similar existing condition
            confidence: Confidence score
        """
        return self.condition_router.detect_new_condition(features, threshold)


class DSAMoEWithFeatureExtractor(nn.Module):
    """
    DSA-MoE with integrated feature extractor.

    Convenience wrapper that combines TransformerFeatureExtractor with DSA-MoE.

    Args:
        config: DSAMoEConfig instance
        feature_extractor: Optional pre-built feature extractor
    """

    def __init__(
        self,
        config: DSAMoEConfig,
        feature_extractor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # Feature extractor
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            # Import here to avoid circular imports
            from models.encoders import TransformerFeatureExtractor
            self.feature_extractor = TransformerFeatureExtractor(
                sensor_dim=config.sensor_dim,
                d_model=config.d_model,
            )

        # DSA-MoE
        self.moe = DSAMoE(config)

    def forward(
        self,
        x: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features and apply DSA-MoE.

        Args:
            x: Raw sensor data [batch, seq_len, sensor_dim]
            condition_id: Optional known condition IDs [batch]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            output: Transformed features [batch, seq_len, d_model]
            aux_outputs: Auxiliary outputs from DSA-MoE
        """
        # Extract features
        features = self.feature_extractor(x, mask)

        # Apply MoE
        output, aux_outputs = self.moe(features, x, condition_id, mask)

        return output, aux_outputs
