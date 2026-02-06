"""
Dataset configuration for RUL prediction.

Provides unified configuration classes for different datasets,
enabling parameterized model creation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """
    Unified configuration for RUL prediction datasets.

    This configuration is used to parameterize model creation,
    ensuring consistency between data and model dimensions.

    Attributes:
        name: Dataset identifier
        sensor_dim: Number of sensor channels
        num_conditions: Number of operating conditions/tasks
        seq_len: Default sequence length for windowing
        max_rul: Maximum RUL value for clipping
        condition_mapping: Optional mapping from sub-dataset names to condition IDs
    """

    name: str
    sensor_dim: int
    num_conditions: int
    seq_len: int = 30
    max_rul: int = 125
    condition_mapping: Optional[Dict[str, int]] = None

    def get_condition_id(self, sub_dataset: str) -> int:
        """
        Get condition ID for a sub-dataset.

        Args:
            sub_dataset: Sub-dataset name (e.g., 'FD001')

        Returns:
            condition_id: Integer condition ID

        Raises:
            KeyError: If sub_dataset not in mapping
        """
        if self.condition_mapping is None:
            raise ValueError(f"No condition mapping defined for {self.name}")
        return self.condition_mapping[sub_dataset]

    def get_sub_datasets(self) -> List[str]:
        """Get list of available sub-datasets."""
        if self.condition_mapping is None:
            return []
        return list(self.condition_mapping.keys())


# C-MAPSS Dataset Configuration
# 14 sensors after removing constant ones (indices 2,3,4,7,8,9,11,12,13,14,15,17,20,21)
# 4 sub-datasets: FD001 (1 condition), FD002 (6 conditions), FD003 (1 condition), FD004 (6 conditions)
# For continual learning, we treat each sub-dataset as a separate condition/task
CMAPSS_CONFIG = DatasetConfig(
    name='cmapss',
    sensor_dim=14,
    num_conditions=4,  # FD001, FD002, FD003, FD004 as separate tasks
    seq_len=30,
    max_rul=125,
    condition_mapping={
        'FD001': 0,
        'FD002': 1,
        'FD003': 2,
        'FD004': 3,
    },
)

# Extended C-MAPSS configuration for fine-grained operating conditions
# FD002 and FD004 have 6 operating conditions each
CMAPSS_EXTENDED_CONFIG = DatasetConfig(
    name='cmapss_extended',
    sensor_dim=14,
    num_conditions=6,  # Maximum operating conditions within any sub-dataset
    seq_len=30,
    max_rul=125,
    condition_mapping={
        'FD001': 0,  # Single condition
        'FD002_OC1': 0, 'FD002_OC2': 1, 'FD002_OC3': 2,
        'FD002_OC4': 3, 'FD002_OC5': 4, 'FD002_OC6': 5,
        'FD003': 0,  # Single condition
        'FD004_OC1': 0, 'FD004_OC2': 1, 'FD004_OC3': 2,
        'FD004_OC4': 3, 'FD004_OC5': 4, 'FD004_OC6': 5,
    },
)

# N-CMAPSS Dataset Configuration
# New C-MAPSS with real flight profiles
# Units 2, 5, 10, 14, 15 are commonly used
# More sensors than C-MAPSS (exact number depends on preprocessing)
NCMAPSS_CONFIG = DatasetConfig(
    name='ncmapss',
    sensor_dim=20,  # Typical after preprocessing
    num_conditions=4,  # Units as separate conditions
    seq_len=50,  # Longer sequences for flight profiles
    max_rul=200,  # Higher max RUL
    condition_mapping={
        'DS02': 0,  # Unit 2
        'DS05': 1,  # Unit 5
        'DS10': 2,  # Unit 10
        'DS14': 3,  # Units 14/15 combined
    },
)


def get_dataset_config(name: str) -> DatasetConfig:
    """
    Get dataset configuration by name.

    Args:
        name: Dataset name ('cmapss', 'cmapss_extended', 'ncmapss')

    Returns:
        config: DatasetConfig instance

    Raises:
        ValueError: If dataset name not recognized
    """
    configs = {
        'cmapss': CMAPSS_CONFIG,
        'cmapss_extended': CMAPSS_EXTENDED_CONFIG,
        'ncmapss': NCMAPSS_CONFIG,
    }

    if name not in configs:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(configs.keys())}"
        )

    return configs[name]


def validate_dataset_model_compatibility(
    dataset_config: DatasetConfig,
    model_sensor_dim: int,
    model_num_conditions: int,
) -> bool:
    """
    Validate that dataset and model configurations are compatible.

    Args:
        dataset_config: Dataset configuration
        model_sensor_dim: Model's expected sensor dimension
        model_num_conditions: Model's expected number of conditions

    Returns:
        compatible: True if compatible

    Raises:
        ValueError: If incompatible with details
    """
    errors = []

    if dataset_config.sensor_dim != model_sensor_dim:
        errors.append(
            f"Sensor dimension mismatch: dataset={dataset_config.sensor_dim}, "
            f"model={model_sensor_dim}"
        )

    if dataset_config.num_conditions > model_num_conditions:
        errors.append(
            f"Condition count mismatch: dataset has {dataset_config.num_conditions} "
            f"conditions but model only supports {model_num_conditions}"
        )

    if errors:
        raise ValueError(
            f"Dataset-model incompatibility for {dataset_config.name}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return True
