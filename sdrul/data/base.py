"""
Base classes for RUL prediction datasets.

Provides abstract interface that all RUL datasets should implement
for consistent integration with the continual learning framework.
"""

import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List


class BaseRULDataset(Dataset, ABC):
    """
    Abstract base class for RUL prediction datasets.

    All RUL datasets should inherit from this class to ensure
    consistent interface with the continual learning framework.

    The key requirement is that __getitem__ returns a 3-tuple:
    (sequence, rul, condition_id) to support multi-condition training.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        pass

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            sequence: Sensor data [seq_len, sensor_dim]
            rul: Remaining useful life (scalar tensor)
            condition_id: Operating condition ID (scalar tensor, dtype=long)
        """
        pass

    @abstractmethod
    def get_sensor_dim(self) -> int:
        """
        Get sensor dimension.

        Returns:
            sensor_dim: Number of sensor channels
        """
        pass

    @abstractmethod
    def get_condition_id(self) -> int:
        """
        Get the condition ID for this dataset.

        For datasets representing a single condition, returns that ID.
        For mixed-condition datasets, returns -1.

        Returns:
            condition_id: Condition ID or -1 if mixed
        """
        pass

    def get_full_trajectory(
        self, engine_id: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get full trajectory for an engine.

        Optional method for datasets that support trajectory-level access.

        Args:
            engine_id: Engine identifier

        Returns:
            data: Full sensor data [trajectory_len, sensor_dim]
            rul: Full RUL trajectory [trajectory_len]
            Or None if not supported/not found
        """
        return None

    def get_all_trajectories(
        self
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get all trajectories in the dataset.

        Optional method for prototype initialization.

        Returns:
            trajectories: List of (sensor_data, rul) tuples
            Or None if not supported
        """
        return None


class RULDatasetWrapper(BaseRULDataset):
    """
    Wrapper to add condition_id to existing datasets.

    Use this to wrap datasets that don't natively return condition_id.

    Args:
        dataset: Underlying dataset (returns (sequence, rul))
        condition_id: Condition ID to assign to all samples
        sensor_dim: Sensor dimension (if not inferrable from dataset)
    """

    def __init__(
        self,
        dataset: Dataset,
        condition_id: int = 0,
        sensor_dim: Optional[int] = None,
    ):
        self.dataset = dataset
        self._condition_id = condition_id
        self._sensor_dim = sensor_dim

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.dataset[idx]

        if len(result) == 2:
            sequence, rul = result
            cond = torch.tensor(self._condition_id, dtype=torch.long)
            return sequence, rul, cond
        elif len(result) == 3:
            return result
        else:
            raise ValueError(
                f"Dataset returned {len(result)} values, expected 2 or 3"
            )

    def get_sensor_dim(self) -> int:
        if self._sensor_dim is not None:
            return self._sensor_dim

        # Try to infer from dataset
        if hasattr(self.dataset, 'get_sensor_dim'):
            return self.dataset.get_sensor_dim()
        if hasattr(self.dataset, 'sensor_dim'):
            return self.dataset.sensor_dim

        # Infer from first sample
        sample = self.dataset[0]
        if len(sample) >= 1:
            return sample[0].shape[-1]

        raise ValueError("Cannot infer sensor_dim from dataset")

    def get_condition_id(self) -> int:
        return self._condition_id

    def get_full_trajectory(
        self, engine_id: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(self.dataset, 'get_full_trajectory'):
            return self.dataset.get_full_trajectory(engine_id)
        return None


class ConcatRULDataset(BaseRULDataset):
    """
    Concatenate multiple RUL datasets.

    Useful for combining datasets from different conditions
    while preserving their individual condition IDs.

    Args:
        datasets: List of BaseRULDataset instances
    """

    def __init__(self, datasets: List[BaseRULDataset]):
        if not datasets:
            raise ValueError("At least one dataset required")

        self.datasets = datasets

        # Compute cumulative lengths for indexing
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)

        # Verify consistent sensor_dim
        sensor_dims = [ds.get_sensor_dim() for ds in datasets]
        if len(set(sensor_dims)) > 1:
            raise ValueError(
                f"Inconsistent sensor dimensions: {sensor_dims}"
            )
        self._sensor_dim = sensor_dims[0]

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx = len(self) + idx

        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                if i == 0:
                    local_idx = idx
                else:
                    local_idx = idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][local_idx]

        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def get_sensor_dim(self) -> int:
        return self._sensor_dim

    def get_condition_id(self) -> int:
        # Mixed conditions
        return -1

    def get_all_trajectories(
        self
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        all_trajs = []
        for ds in self.datasets:
            trajs = ds.get_all_trajectories()
            if trajs is not None:
                all_trajs.extend(trajs)
        return all_trajs if all_trajs else None
