"""
Data loading and processing modules.
"""

from .base import (
    BaseRULDataset,
    RULDatasetWrapper,
    ConcatRULDataset,
)

from .config import (
    DatasetConfig,
    CMAPSS_CONFIG,
    CMAPSS_EXTENDED_CONFIG,
    NCMAPSS_CONFIG,
    get_dataset_config,
    validate_dataset_model_compatibility,
)

from .cmapss import (
    CMAPSSDataset,
    CMAPSSDataModule,
    ContinualCMAPSS,
    create_synthetic_cmapss,
    get_cmapss_statistics,
)

from .ncmapss import (
    NCMAPSSDataset,
    ContinualNCMAPSS,
    create_synthetic_ncmapss,
)

__all__ = [
    # Base classes
    'BaseRULDataset',
    'RULDatasetWrapper',
    'ConcatRULDataset',
    # Config
    'DatasetConfig',
    'CMAPSS_CONFIG',
    'CMAPSS_EXTENDED_CONFIG',
    'NCMAPSS_CONFIG',
    'get_dataset_config',
    'validate_dataset_model_compatibility',
    # C-MAPSS
    'CMAPSSDataset',
    'CMAPSSDataModule',
    'ContinualCMAPSS',
    'create_synthetic_cmapss',
    'get_cmapss_statistics',
    # N-CMAPSS
    'NCMAPSSDataset',
    'ContinualNCMAPSS',
    'create_synthetic_ncmapss',
]
