"""
C-MAPSS dataset processing for RUL prediction.

This module handles downloading, preprocessing, and loading of the NASA C-MAPSS
turbofan engine degradation dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict
import os

from .base import BaseRULDataset


class CMAPSSDataset(BaseRULDataset):
    """
    C-MAPSS dataset for RUL prediction.

    The C-MAPSS dataset contains run-to-failure degradation trajectories of
    turbofan engines under different operating conditions.

    Returns (sequence, rul, condition_id) tuples for compatibility with
    the continual learning framework.

    Args:
        data_path: Path to processed .npz file
        seq_len: Length of input sequences
        overlap: Overlap between consecutive sequences (0 to 1)
        rul_only: If True, only return sequences with remaining RUL > 0
        max_rul: Maximum RUL value for clipping (censoring)
        normalize: Whether to normalize sensor data
        split: 'train' or 'test'
        condition_id: Operating condition ID for this dataset (default: 0)
    """

    SENSOR_COLUMNS = [
        'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
        'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
        'Nf_dmd', 'PCfR_dmd', 'W31', 'W32'
    ]

    # Remove constant sensors (based on C-MAPSS analysis)
    SENSOR_COLUMNS_USED = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

    def __init__(
        self,
        data_path: str,
        seq_len: int = 30,
        overlap: float = 0.5,
        rul_only: bool = True,
        max_rul: int = 125,
        normalize: bool = True,
        split: str = 'train',
        condition_id: int = 0,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.overlap = overlap
        self.rul_only = rul_only
        self.max_rul = max_rul
        self.normalize = normalize
        self.split = split
        self.condition_id = condition_id

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and preprocess data from npz file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path)

        if self.split == 'train':
            self.sensor_data = data['x_train']
            self.rul_data = data['y_train']
            self.engine_ids = data.get('engine_ids_train', None)
        else:
            self.sensor_data = data['x_test']
            self.rul_data = data['y_test']
            self.engine_ids = data.get('engine_ids_test', None)

        # Normalize sensor data
        if self.normalize:
            self.sensor_data = self._normalize_sensors(self.sensor_data)

        # Create sequences
        self.sequences, self.ruls, self.engine_ids_seq = self._create_sequences()

    def _normalize_sensors(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize sensor data using min-max normalization per sensor.

        Args:
            data: Raw sensor data [num_samples, num_sensors]

        Returns:
            normalized: Normalized sensor data
        """
        # Avoid division by zero
        min_vals = data.min(axis=0, keepdims=True)
        max_vals = data.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals + 1e-8

        normalized = (data - min_vals) / range_vals
        return normalized.astype(np.float32)

    def _create_sequences(
        self
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding window sequences from data.

        Returns:
            sequences: Sensor sequences [num_sequences, seq_len, num_sensors]
            ruls: RUL values [num_sequences]
            engine_ids: Engine IDs for each sequence [num_sequences]
        """
        sequences = []
        ruls = []
        engine_ids_seq = [] if self.engine_ids is not None else None

        if self.engine_ids is not None:
            unique_engines = np.unique(self.engine_ids)
        else:
            # Single engine
            unique_engines = [0]
            self.engine_ids = np.zeros(len(self.sensor_data), dtype=int)

        step = int(self.seq_len * (1 - self.overlap))

        for engine_id in unique_engines:
            # Get data for this engine
            engine_mask = self.engine_ids == engine_id
            engine_data = self.sensor_data[engine_mask]
            engine_rul = self.rul_data[engine_mask]

            # Skip if too short
            if len(engine_data) < self.seq_len:
                continue

            # Create sliding windows
            for i in range(0, len(engine_data) - self.seq_len + 1, step):
                seq = engine_data[i:i + self.seq_len]
                rul = engine_rul[i + self.seq_len - 1]

                # Filter by RUL if specified
                if self.rul_only and rul <= 0:
                    continue

                sequences.append(seq)
                ruls.append(rul)
                if engine_ids_seq is not None:
                    engine_ids_seq.append(engine_id)

        sequences = np.array(sequences, dtype=np.float32)
        ruls = np.array(ruls, dtype=np.float32)
        ruls = np.clip(ruls, 0, self.max_rul)  # Clip RUL

        if engine_ids_seq is not None:
            engine_ids_seq = np.array(engine_ids_seq, dtype=int)

        return sequences, ruls, engine_ids_seq

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            sequence: Sensor data [seq_len, num_sensors]
            rul: Remaining useful life (scalar)
            condition_id: Operating condition ID (scalar, dtype=long)
        """
        sequence = torch.from_numpy(self.sequences[idx])
        rul = torch.tensor(self.ruls[idx], dtype=torch.float32)
        cond = torch.tensor(self.condition_id, dtype=torch.long)

        return sequence, rul, cond

    def get_sensor_dim(self) -> int:
        """Get sensor dimension."""
        return self.sequences.shape[-1] if len(self.sequences) > 0 else 14

    def get_condition_id(self) -> int:
        """Get the condition ID for this dataset."""
        return self.condition_id

    def get_full_trajectory(
        self,
        engine_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get full trajectory for an engine.

        Args:
            engine_id: Engine ID

        Returns:
            data: Full sensor data [trajectory_len, num_sensors]
            rul: Full RUL trajectory [trajectory_len]
        """
        if self.engine_ids is None:
            return None

        engine_mask = self.engine_ids == engine_id
        if not engine_mask.any():
            return None

        data = torch.from_numpy(self.sensor_data[engine_mask])
        rul = torch.from_numpy(self.rul_data[engine_mask])

        return data, rul


class CMAPSSDataModule:
    """
    Data module for C-MAPSS dataset with train/val/test splits.

    Args:
        data_dir: Directory containing processed C-MAPSS data
        sub_dataset: Sub-dataset ID (FD001, FD002, FD003, FD004)
        seq_len: Input sequence length
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
    """

    def __init__(
        self,
        data_dir: str,
        sub_dataset: str = 'FD001',
        seq_len: int = 30,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.data_dir = data_dir
        self.sub_dataset = sub_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Paths
        self.train_path = os.path.join(data_dir, f'{sub_dataset}_train.npz')
        self.test_path = os.path.join(data_dir, f'{sub_dataset}_test.npz')

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, val_split: float = 0.2):
        """
        Set up train/val/test datasets.

        Args:
            val_split: Fraction of training data to use for validation
        """
        # Load training data
        full_train = CMAPSSDataset(
            self.train_path,
            seq_len=self.seq_len,
            split='train',
        )

        # Split into train/val
        train_size = int((1 - val_split) * len(full_train))
        val_size = len(full_train) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_train, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Load test data
        self.test_dataset = CMAPSSDataset(
            self.test_path,
            seq_len=self.seq_len,
            split='test',
        )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_sensor_dim(self) -> int:
        """Get sensor dimension."""
        if self.train_dataset is not None:
            return self.train_dataset.dataset.sensor_data.shape[1]
        # Default C-MAPSS uses 14 sensors after removing constant ones
        return 14


def create_synthetic_cmapss(
    num_engines: int = 100,
    num_sensors: int = 14,
    max_cycles: int = 300,
    min_cycles: int = 100,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create synthetic C-MAPSS-like data for testing.

    Args:
        num_engines: Number of engines to simulate
        num_sensors: Number of sensor channels
        max_cycles: Maximum cycles per engine
        min_cycles: Minimum cycles per engine
        save_path: Optional path to save the data
        seed: Random seed

    Returns:
        data: Dictionary with 'x', 'y', 'engine_ids'
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_sensor_data = []
    all_rul = []
    all_engine_ids = []

    for engine_id in range(num_engines):
        # Random failure time
        failure_time = np.random.randint(min_cycles, max_cycles)

        # Generate degradation trajectory (exponential decay)
        cycles = np.arange(1, failure_time + 1)
        degradation = 1 - (cycles / failure_time) ** np.random.uniform(0.5, 2.0)
        degradation = np.clip(degradation, 0, 1)

        # Generate sensor data with degradation trend + noise
        sensor_data = np.zeros((failure_time, num_sensors))

        for sensor in range(num_sensors):
            # Base value (random per sensor)
            base_value = np.random.uniform(-1, 1)

            # Sensor-specific degradation response
            degradation_response = np.random.uniform(0.5, 2.0)

            # Sensor trend (some increase, some decrease with degradation)
            trend_direction = 1 if np.random.random() > 0.5 else -1

            # Sensor data
            sensor_data[:, sensor] = (
                base_value +
                trend_direction * degradation * degradation_response +
                np.random.randn(failure_time) * 0.05
            )

        # RUL (remaining cycles)
        rul = failure_time - cycles

        all_sensor_data.append(sensor_data)
        all_rul.append(rul)
        all_engine_ids.append(np.full(failure_time, engine_id))

    # Concatenate all data
    x = np.concatenate(all_sensor_data, axis=0)
    y = np.concatenate(all_rul, axis=0)
    engine_ids = np.concatenate(all_engine_ids, axis=0)

    # Split into train and test (80/20)
    split_idx = int(len(x) * 0.8)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    engine_ids_train = engine_ids[:split_idx]
    engine_ids_test = engine_ids[split_idx:]

    # Use keys compatible with CMAPSSDataset
    data = {
        'x_train': x_train,
        'y_train': y_train,
        'engine_ids_train': engine_ids_train,
        'x_test': x_test,
        'y_test': y_test,
        'engine_ids_test': engine_ids_test,
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **data)

    return data


class ContinualCMAPSS:
    """
    C-MAPSS dataset for continual learning experiments.

    Simulates continual learning by treating different sub-datasets
    or different operating conditions as sequential tasks.

    Each sub-dataset is assigned a unique condition_id for use with
    the continual learning framework.

    Args:
        data_dir: Directory containing processed C-MAPSS data
        sub_datasets: List of sub-dataset IDs (e.g., ['FD001', 'FD003'])
        seq_len: Input sequence length
        batch_size: Batch size for dataloaders
    """

    # Default condition mapping for C-MAPSS sub-datasets
    CONDITION_MAPPING = {
        'FD001': 0,
        'FD002': 1,
        'FD003': 2,
        'FD004': 3,
    }

    def __init__(
        self,
        data_dir: str,
        sub_datasets: List[str] = ['FD001', 'FD003'],
        seq_len: int = 30,
        batch_size: int = 64,
        condition_mapping: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = data_dir
        self.sub_datasets = sub_datasets
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Use provided mapping or default
        self.condition_mapping = condition_mapping or self.CONDITION_MAPPING

        # Load all datasets with appropriate condition_ids
        self.datasets = {}
        for sub_id in sub_datasets:
            path = os.path.join(data_dir, f'{sub_id}_train.npz')
            if not os.path.exists(path):
                # Create synthetic data
                data = create_synthetic_cmapss(
                    num_engines=50,
                    save_path=path,
                    seed=int(sub_id[-1]),
                )

            # Get condition_id for this sub-dataset
            condition_id = self.condition_mapping.get(sub_id, 0)

            self.datasets[sub_id] = CMAPSSDataset(
                path,
                seq_len=seq_len,
                split='train',
                condition_id=condition_id,
            )

    def get_task_dataset(self, task_idx: int) -> CMAPSSDataset:
        """Get dataset for a specific task (sub-dataset)."""
        sub_id = self.sub_datasets[task_idx]
        return self.datasets[sub_id]

    def get_task_dataloader(self, task_idx: int) -> DataLoader:
        """Get dataloader for a specific task."""
        dataset = self.get_task_dataset(task_idx)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def get_all_task_loaders(self) -> List[DataLoader]:
        """Get dataloaders for all tasks."""
        return [self.get_task_dataloader(i) for i in range(len(self.sub_datasets))]

    def get_condition_id(self, task_idx: int) -> int:
        """Get condition_id for a specific task."""
        sub_id = self.sub_datasets[task_idx]
        return self.condition_mapping.get(sub_id, 0)

    def get_num_conditions(self) -> int:
        """Get total number of unique conditions."""
        return len(set(self.condition_mapping[s] for s in self.sub_datasets))

    def __len__(self) -> int:
        return len(self.sub_datasets)


def get_cmapss_statistics(
    data_path: str,
) -> Dict[str, np.ndarray]:
    """
    Compute statistics for C-MAPSS dataset normalization.

    Args:
        data_path: Path to C-MAPSS .npz file

    Returns:
        stats: Dictionary with 'mean', 'std', 'min', 'max' per sensor
    """
    data = np.load(data_path)
    x = data['x_train'] if 'x_train' in data else data['x']

    stats = {
        'mean': x.mean(axis=0),
        'std': x.std(axis=0),
        'min': x.min(axis=0),
        'max': x.max(axis=0),
    }

    return stats
