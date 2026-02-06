"""
N-CMAPSS dataset processing for RUL prediction.

This module handles loading and preprocessing of the NASA N-CMAPSS
(New C-MAPSS) turbofan engine degradation dataset with real flight profiles.

N-CMAPSS provides more realistic degradation data compared to C-MAPSS,
with actual flight conditions and more sensor channels.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict
import os

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from .base import BaseRULDataset


class NCMAPSSDataset(BaseRULDataset):
    """
    N-CMAPSS dataset for RUL prediction.

    The N-CMAPSS dataset contains run-to-failure degradation trajectories
    with real flight profiles. Each unit (DS02, DS05, etc.) represents
    different operating conditions.

    Returns (sequence, rul, condition_id) tuples for compatibility with
    the continual learning framework.

    Args:
        data_path: Path to processed .npz or .h5 file
        unit_id: Unit identifier (e.g., 'DS02', 'DS05')
        seq_len: Length of input sequences
        overlap: Overlap between consecutive sequences (0 to 1)
        max_rul: Maximum RUL value for clipping
        normalize: Whether to normalize sensor data
        split: 'train', 'val', or 'test'
        condition_id: Operating condition ID for this dataset
    """

    # N-CMAPSS sensor columns (typical configuration)
    # Actual columns depend on preprocessing
    SENSOR_COLUMNS = [
        'alt', 'Mach', 'TRA', 'T2', 'T24', 'T30', 'T48', 'T50',
        'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50',
        'Nf', 'Nc', 'Wf', 'phi', 'BPR'
    ]

    # Default sensor indices to use (after removing constant/noisy ones)
    DEFAULT_SENSOR_INDICES = list(range(20))

    # Unit to condition_id mapping
    UNIT_MAPPING = {
        'DS02': 0,
        'DS05': 1,
        'DS10': 2,
        'DS14': 3,
        'DS15': 3,  # Combined with DS14
    }

    def __init__(
        self,
        data_path: str,
        unit_id: str = 'DS02',
        seq_len: int = 50,
        overlap: float = 0.5,
        max_rul: int = 200,
        normalize: bool = True,
        split: str = 'train',
        condition_id: Optional[int] = None,
        sensor_indices: Optional[List[int]] = None,
    ):
        self.data_path = data_path
        self.unit_id = unit_id
        self.seq_len = seq_len
        self.overlap = overlap
        self.max_rul = max_rul
        self.normalize = normalize
        self.split = split
        self.sensor_indices = sensor_indices or self.DEFAULT_SENSOR_INDICES

        # Set condition_id from unit mapping if not provided
        if condition_id is not None:
            self.condition_id = condition_id
        else:
            self.condition_id = self.UNIT_MAPPING.get(unit_id, 0)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and preprocess data from file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        ext = os.path.splitext(self.data_path)[1].lower()

        if ext == '.npz':
            self._load_npz()
        elif ext in ['.h5', '.hdf5']:
            self._load_h5()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Normalize sensor data
        if self.normalize:
            self.sensor_data = self._normalize_sensors(self.sensor_data)

        # Create sequences
        self.sequences, self.ruls, self.engine_ids_seq = self._create_sequences()

    def _load_npz(self):
        """Load data from npz file."""
        data = np.load(self.data_path, allow_pickle=True)

        key_prefix = f'{self.split}_' if self.split != 'train' else ''

        # Try different key formats
        x_key = f'x_{self.split}' if f'x_{self.split}' in data else 'x_train'
        y_key = f'y_{self.split}' if f'y_{self.split}' in data else 'y_train'
        id_key = f'engine_ids_{self.split}' if f'engine_ids_{self.split}' in data else 'engine_ids_train'

        self.sensor_data = data[x_key]
        self.rul_data = data[y_key]
        self.engine_ids = data.get(id_key, None)

        # Select sensor columns
        if self.sensor_data.shape[-1] > len(self.sensor_indices):
            self.sensor_data = self.sensor_data[:, self.sensor_indices]

    def _load_h5(self):
        """Load data from HDF5 file."""
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required to load HDF5 files. "
                "Install with: pip install h5py"
            )

        with h5py.File(self.data_path, 'r') as f:
            # N-CMAPSS HDF5 structure varies by source
            # Common structure: /unit_id/X_s, /unit_id/Y_s, etc.
            if self.unit_id in f:
                group = f[self.unit_id]
                self.sensor_data = np.array(group['X_s'])
                self.rul_data = np.array(group['Y_s']).flatten()
                self.engine_ids = np.array(group.get('unit', np.zeros(len(self.rul_data))))
            else:
                # Fallback: try root level
                self.sensor_data = np.array(f['X_s'])
                self.rul_data = np.array(f['Y_s']).flatten()
                self.engine_ids = np.array(f.get('unit', np.zeros(len(self.rul_data))))

            # Select sensor columns
            if self.sensor_data.shape[-1] > len(self.sensor_indices):
                self.sensor_data = self.sensor_data[:, self.sensor_indices]

    def _normalize_sensors(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize sensor data using min-max normalization per sensor.

        Args:
            data: Raw sensor data [num_samples, num_sensors]

        Returns:
            normalized: Normalized sensor data
        """
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
            unique_engines = [0]
            self.engine_ids = np.zeros(len(self.sensor_data), dtype=int)

        step = max(1, int(self.seq_len * (1 - self.overlap)))

        for engine_id in unique_engines:
            engine_mask = self.engine_ids == engine_id
            engine_data = self.sensor_data[engine_mask]
            engine_rul = self.rul_data[engine_mask]

            if len(engine_data) < self.seq_len:
                continue

            for i in range(0, len(engine_data) - self.seq_len + 1, step):
                seq = engine_data[i:i + self.seq_len]
                rul = engine_rul[i + self.seq_len - 1]

                sequences.append(seq)
                ruls.append(rul)
                if engine_ids_seq is not None:
                    engine_ids_seq.append(engine_id)

        if not sequences:
            # Return empty arrays with correct shape
            return (
                np.zeros((0, self.seq_len, len(self.sensor_indices)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=int) if engine_ids_seq is not None else None,
            )

        sequences = np.array(sequences, dtype=np.float32)
        ruls = np.array(ruls, dtype=np.float32)
        ruls = np.clip(ruls, 0, self.max_rul)

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
        return self.sequences.shape[-1] if len(self.sequences) > 0 else len(self.sensor_indices)

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

    def get_all_trajectories(
        self
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get all trajectories in the dataset."""
        if self.engine_ids is None:
            return None

        trajectories = []
        for engine_id in np.unique(self.engine_ids):
            traj = self.get_full_trajectory(engine_id)
            if traj is not None:
                trajectories.append(traj)

        return trajectories if trajectories else None


class ContinualNCMAPSS:
    """
    N-CMAPSS dataset for continual learning experiments.

    Treats different units as sequential tasks for continual learning.

    Args:
        data_dir: Directory containing processed N-CMAPSS data
        units: List of unit IDs (e.g., ['DS02', 'DS05', 'DS10'])
        seq_len: Input sequence length
        batch_size: Batch size for dataloaders
    """

    def __init__(
        self,
        data_dir: str,
        units: List[str] = ['DS02', 'DS05', 'DS10'],
        seq_len: int = 50,
        batch_size: int = 64,
        condition_mapping: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = data_dir
        self.units = units
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Use provided mapping or default
        self.condition_mapping = condition_mapping or NCMAPSSDataset.UNIT_MAPPING

        # Load all datasets
        self.datasets = {}
        for unit_id in units:
            path = os.path.join(data_dir, f'{unit_id}.npz')
            if os.path.exists(path):
                condition_id = self.condition_mapping.get(unit_id, 0)
                self.datasets[unit_id] = NCMAPSSDataset(
                    path,
                    unit_id=unit_id,
                    seq_len=seq_len,
                    condition_id=condition_id,
                )

    def get_task_dataset(self, task_idx: int) -> NCMAPSSDataset:
        """Get dataset for a specific task (unit)."""
        unit_id = self.units[task_idx]
        return self.datasets[unit_id]

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
        return [self.get_task_dataloader(i) for i in range(len(self.units))]

    def get_condition_id(self, task_idx: int) -> int:
        """Get condition_id for a specific task."""
        unit_id = self.units[task_idx]
        return self.condition_mapping.get(unit_id, 0)

    def __len__(self) -> int:
        return len(self.units)


def create_synthetic_ncmapss(
    num_engines: int = 50,
    num_sensors: int = 20,
    max_cycles: int = 500,
    min_cycles: int = 200,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create synthetic N-CMAPSS-like data for testing.

    Args:
        num_engines: Number of engines to simulate
        num_sensors: Number of sensor channels
        max_cycles: Maximum cycles per engine
        min_cycles: Minimum cycles per engine
        save_path: Optional path to save the data
        seed: Random seed

    Returns:
        data: Dictionary with 'x_train', 'y_train', 'engine_ids_train'
    """
    np.random.seed(seed)

    all_sensor_data = []
    all_rul = []
    all_engine_ids = []

    for engine_id in range(num_engines):
        failure_time = np.random.randint(min_cycles, max_cycles)
        cycles = np.arange(1, failure_time + 1)

        # More complex degradation for N-CMAPSS
        degradation = 1 - (cycles / failure_time) ** np.random.uniform(0.8, 1.5)
        degradation = np.clip(degradation, 0, 1)

        sensor_data = np.zeros((failure_time, num_sensors))

        for sensor in range(num_sensors):
            base_value = np.random.uniform(-1, 1)
            degradation_response = np.random.uniform(0.3, 1.5)
            trend_direction = 1 if np.random.random() > 0.5 else -1

            # Add flight profile variation
            flight_variation = np.sin(np.linspace(0, 10 * np.pi, failure_time)) * 0.1

            sensor_data[:, sensor] = (
                base_value +
                trend_direction * degradation * degradation_response +
                flight_variation +
                np.random.randn(failure_time) * 0.03
            )

        rul = failure_time - cycles

        all_sensor_data.append(sensor_data)
        all_rul.append(rul)
        all_engine_ids.append(np.full(failure_time, engine_id))

    x = np.concatenate(all_sensor_data, axis=0)
    y = np.concatenate(all_rul, axis=0)
    engine_ids = np.concatenate(all_engine_ids, axis=0)

    data = {'x_train': x, 'y_train': y, 'engine_ids_train': engine_ids}

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        np.savez(save_path, **data)

    return data
