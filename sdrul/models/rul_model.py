"""
Unified RUL Prediction Model.

Combines TransformerFeatureExtractor, DSA-MoE, and GaussianRULHead
into a single model with consistent interface.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .encoders import TransformerFeatureExtractor, GaussianRULHead
from .moe import DSAMoE, DSAMoEConfig

if TYPE_CHECKING:
    from data.config import DatasetConfig


@dataclass
class RULModelConfig:
    """
    Configuration for unified RUL prediction model.

    Note: These are research-validated hyperparameters.
    Do not change without re-validation.
    """

    # Feature extractor
    sensor_dim: int = 14
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # DSA-MoE (research-specified values)
    num_conditions: int = 6
    num_stages: int = 3
    bottleneck_dim: int = 64
    moe_temperature: float = 1.0
    load_balance_weight: float = 0.01

    # RUL head
    min_sigma: float = 0.1
    max_sigma: float = 50.0

    @classmethod
    def from_dataset_config(
        cls,
        dataset_config: 'DatasetConfig',
        num_stages: int = 3,
        **kwargs,
    ) -> 'RULModelConfig':
        """
        Create RULModelConfig from a DatasetConfig.

        Args:
            dataset_config: Dataset configuration
            num_stages: Number of degradation stages
            **kwargs: Additional config overrides

        Returns:
            config: RULModelConfig instance
        """
        return cls(
            sensor_dim=dataset_config.sensor_dim,
            num_conditions=dataset_config.num_conditions,
            num_stages=num_stages,
            **kwargs,
        )


class RULPredictionModel(nn.Module):
    """
    Unified RUL Prediction Model.

    Chains: TransformerFeatureExtractor -> DSA-MoE -> GaussianRULHead

    This model provides a consistent interface returning ((mu, sigma), aux_dict)
    for use with ContinualTrainer and other components.

    The feature_extractor is designed to be shared with TCSD module.

    Args:
        config: RULModelConfig instance
        feature_extractor: Optional pre-built feature extractor (for sharing with TCSD)
    """

    def __init__(
        self,
        config: Optional[RULModelConfig] = None,
        feature_extractor: Optional[TransformerFeatureExtractor] = None,
    ):
        super().__init__()
        self.config = config or RULModelConfig()

        # Feature extractor (can be shared with TCSD)
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = TransformerFeatureExtractor(
                sensor_dim=self.config.sensor_dim,
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                num_layers=self.config.num_encoder_layers,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
            )

        # DSA-MoE
        moe_config = DSAMoEConfig(
            num_conditions=self.config.num_conditions,
            num_stages=self.config.num_stages,
            d_model=self.config.d_model,
            bottleneck_dim=self.config.bottleneck_dim,
            sensor_dim=self.config.sensor_dim,
            temperature=self.config.moe_temperature,
            load_balance_weight=self.config.load_balance_weight,
            dropout=self.config.dropout,
        )
        self.moe = DSAMoE(moe_config)

        # RUL prediction head
        self.rul_head = GaussianRULHead(
            d_model=self.config.d_model,
            min_sigma=self.config.min_sigma,
            max_sigma=self.config.max_sigma,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete model.

        Args:
            x: Raw sensor data [batch, seq_len, sensor_dim]
            condition_id: Optional operating condition IDs [batch]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            output: ((mu, sigma), aux_dict) where:
                - mu: Predicted RUL mean [batch]
                - sigma: Predicted RUL std [batch]
                - aux_dict: Dictionary containing routing weights and losses
        """
        # Extract features
        features = self.feature_extractor(x, mask)  # [batch, seq_len, d_model]

        # Apply DSA-MoE
        moe_output, aux = self.moe(features, x, condition_id, mask)

        # Predict RUL distribution
        mu, sigma = self.rul_head(moe_output)

        return (mu, sigma), aux

    def get_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get features without MoE or RUL prediction.

        Useful for TCSD which needs raw features.

        Args:
            x: Raw sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask

        Returns:
            features: Extracted features [batch, seq_len, d_model]
        """
        return self.feature_extractor(x, mask)

    def set_warmup_mode(self, enabled: bool):
        """Enable/disable MoE warmup mode."""
        self.moe.set_warmup_mode(enabled)

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.moe.set_epoch(epoch)


def create_shared_model_and_tcsd(
    config: Optional[RULModelConfig] = None,
) -> Tuple['RULPredictionModel', 'TCSD']:
    """
    Create RUL model and TCSD with shared feature extractor and MoE.

    This is the recommended way to instantiate both components
    to ensure they share the same feature extractor and MoE.

    Args:
        config: RULModelConfig instance

    Returns:
        model: RULPredictionModel instance
        tcsd: TCSD instance with shared feature extractor and MoE
    """
    from .tcsd import TCSD, TCSDConfig

    config = config or RULModelConfig()

    # Create shared feature extractor
    shared_feature_extractor = TransformerFeatureExtractor(
        sensor_dim=config.sensor_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )

    # Create RUL model with shared feature extractor
    model = RULPredictionModel(
        config=config,
        feature_extractor=shared_feature_extractor,
    )

    # Create TCSD config
    tcsd_config = TCSDConfig(
        sensor_dim=config.sensor_dim,
        d_model=config.d_model,
        prototype_dim=128,  # Research-specified value
        num_prototypes_per_condition=5,  # Research-specified value
        num_conditions=config.num_conditions,
        ema_decay=0.9,  # Research-specified value
        alpha_kl=0.5,  # Research-specified value
    )

    # Create TCSD with shared feature extractor and MoE
    tcsd = TCSD(
        config=tcsd_config,
        feature_extractor=shared_feature_extractor,
        moe=model.moe,  # Share MoE with student branch
    )

    return model, tcsd
