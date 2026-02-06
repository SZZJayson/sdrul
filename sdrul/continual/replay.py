"""
Replay buffer for continual learning.

Manages storage and sampling of replay samples for
experience replay strategies.

Includes SmartReplayBuffer with:
- Prototype-guided sample selection
- Importance-weighted sampling
- Dynamic replay ratio adjustment
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict
import random
import logging

if TYPE_CHECKING:
    from models.tcsd.prototype_manager import TrajectoryPrototypeManager

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Replay buffer for continual learning.

    Supports:
    - Per-task/condition storage
    - Reservoir sampling for bounded memory
    - Priority-based sampling
    - Integration with generative replay

    Args:
        max_size: Maximum total buffer size (default: 10000)
        per_task_limit: Maximum samples per task (default: None, uses max_size/num_tasks)
        prioritized: Use prioritized replay (default: False)
    """

    def __init__(
        self,
        max_size: int = 10000,
        per_task_limit: Optional[int] = None,
        prioritized: bool = False,
    ):
        self.max_size = max_size
        self.per_task_limit = per_task_limit
        self.prioritized = prioritized

        # Storage: task_id -> list of (x, y, metadata)
        self.buffer: Dict[int, List[Tuple]] = defaultdict(list)

        # Priority storage (if prioritized)
        self.priorities: Dict[int, List[float]] = defaultdict(list)

        # Statistics
        self.total_added = 0
        self.total_sampled = 0

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        priority: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        """
        Add a sample to the buffer.

        Args:
            x: Input data [seq_len, sensor_dim] or [sensor_dim]
            y: Target value (scalar or tensor)
            task_id: Task/condition identifier
            priority: Sample priority (for prioritized replay)
            metadata: Optional metadata dictionary
        """
        sample = (x.detach().cpu(), y.detach().cpu() if isinstance(y, torch.Tensor) else y, metadata)

        task_buffer = self.buffer[task_id]
        task_priorities = self.priorities[task_id]

        # Determine limit for this task
        if self.per_task_limit is not None:
            limit = self.per_task_limit
        else:
            num_tasks = max(1, len(self.buffer))
            limit = self.max_size // num_tasks

        if len(task_buffer) < limit:
            # Buffer not full, just add
            task_buffer.append(sample)
            task_priorities.append(priority)
        else:
            # Reservoir sampling
            idx = random.randint(0, self.total_added)
            if idx < limit:
                task_buffer[idx] = sample
                task_priorities[idx] = priority

        self.total_added += 1

    def add_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: Union[int, torch.Tensor],
        priorities: Optional[torch.Tensor] = None,
    ):
        """
        Add a batch of samples.

        Args:
            x: Input batch [batch, seq_len, sensor_dim]
            y: Target batch [batch]
            task_id: Task ID (int) or per-sample IDs [batch]
            priorities: Optional priority values [batch]
        """
        batch_size = x.size(0)

        if isinstance(task_id, int):
            task_ids = [task_id] * batch_size
        else:
            task_ids = task_id.tolist()

        if priorities is None:
            priorities = [1.0] * batch_size
        else:
            priorities = priorities.tolist()

        for i in range(batch_size):
            self.add(x[i], y[i], task_ids[i], priorities[i])

    def sample(
        self,
        batch_size: int,
        task_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of samples
            task_id: Specific task to sample from (None = all tasks)
            device: Device to move tensors to

        Returns:
            x: Input batch [batch, ...]
            y: Target batch [batch]
            task_ids: Task IDs [batch]
        """
        if task_id is not None:
            # Sample from specific task
            samples = self._sample_from_task(batch_size, task_id)
            task_ids = [task_id] * len(samples)
        else:
            # Sample from all tasks
            samples = []
            task_ids = []

            # Distribute samples across tasks
            tasks = list(self.buffer.keys())
            if not tasks:
                raise ValueError("Buffer is empty")

            samples_per_task = batch_size // len(tasks)
            remainder = batch_size % len(tasks)

            for i, tid in enumerate(tasks):
                n = samples_per_task + (1 if i < remainder else 0)
                task_samples = self._sample_from_task(n, tid)
                samples.extend(task_samples)
                task_ids.extend([tid] * len(task_samples))

        if not samples:
            raise ValueError("No samples available")

        # Stack samples
        x_list = [s[0] for s in samples]
        y_list = [s[1] for s in samples]

        x = torch.stack(x_list)
        y = torch.tensor(y_list) if not isinstance(y_list[0], torch.Tensor) else torch.stack(y_list)
        task_ids = torch.tensor(task_ids)

        if device is not None:
            x = x.to(device)
            y = y.to(device)
            task_ids = task_ids.to(device)

        self.total_sampled += len(samples)

        return x, y, task_ids

    def _sample_from_task(
        self,
        n: int,
        task_id: int,
    ) -> List[Tuple]:
        """Sample n items from a specific task."""
        task_buffer = self.buffer.get(task_id, [])

        if not task_buffer:
            return []

        n = min(n, len(task_buffer))

        if self.prioritized:
            # Priority-based sampling
            priorities = np.array(self.priorities[task_id])
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(task_buffer), size=n, replace=False, p=probs)
        else:
            # Uniform sampling
            indices = random.sample(range(len(task_buffer)), n)

        return [task_buffer[i] for i in indices]

    def sample_for_replay(
        self,
        batch_size: int,
        current_task: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample for replay during training on current task.

        Samples from all previous tasks (excluding current).

        Args:
            batch_size: Number of samples
            current_task: Current task ID (excluded from sampling)
            device: Device to move tensors to

        Returns:
            x, y, task_ids tensors
        """
        previous_tasks = [t for t in self.buffer.keys() if t != current_task]

        if not previous_tasks:
            # No previous tasks, return empty
            return (
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        samples = []
        task_ids = []

        samples_per_task = batch_size // len(previous_tasks)
        remainder = batch_size % len(previous_tasks)

        for i, tid in enumerate(previous_tasks):
            n = samples_per_task + (1 if i < remainder else 0)
            task_samples = self._sample_from_task(n, tid)
            samples.extend(task_samples)
            task_ids.extend([tid] * len(task_samples))

        if not samples:
            return (
                torch.empty(0, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        x = torch.stack([s[0] for s in samples])
        y_list = [s[1] for s in samples]
        y = torch.tensor(y_list) if not isinstance(y_list[0], torch.Tensor) else torch.stack(y_list)
        task_ids = torch.tensor(task_ids)

        if device is not None:
            x = x.to(device)
            y = y.to(device)
            task_ids = task_ids.to(device)

        return x, y, task_ids

    def get_task_data(
        self,
        task_id: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all data for a specific task.

        Args:
            task_id: Task identifier
            device: Device to move tensors to

        Returns:
            x: All inputs for task [n, ...]
            y: All targets for task [n]
        """
        task_buffer = self.buffer.get(task_id, [])

        if not task_buffer:
            return torch.empty(0), torch.empty(0)

        x = torch.stack([s[0] for s in task_buffer])
        y_list = [s[1] for s in task_buffer]
        y = torch.tensor(y_list) if not isinstance(y_list[0], torch.Tensor) else torch.stack(y_list)

        if device is not None:
            x = x.to(device)
            y = y.to(device)

        return x, y

    def __len__(self) -> int:
        """Total number of samples in buffer."""
        return sum(len(v) for v in self.buffer.values())

    def task_sizes(self) -> Dict[int, int]:
        """Get number of samples per task."""
        return {k: len(v) for k, v in self.buffer.items()}

    def clear(self, task_id: Optional[int] = None):
        """
        Clear buffer.

        Args:
            task_id: Specific task to clear (None = clear all)
        """
        if task_id is not None:
            self.buffer[task_id] = []
            self.priorities[task_id] = []
        else:
            self.buffer.clear()
            self.priorities.clear()

    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            'buffer': dict(self.buffer),
            'priorities': dict(self.priorities),
            'max_size': self.max_size,
            'per_task_limit': self.per_task_limit,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
        }

    def load_state_dict(self, state: Dict):
        """Load state from checkpoint."""
        self.buffer = defaultdict(list, state['buffer'])
        self.priorities = defaultdict(list, state['priorities'])
        self.max_size = state['max_size']
        self.per_task_limit = state['per_task_limit']
        self.total_added = state['total_added']
        self.total_sampled = state['total_sampled']


class SmartReplayBuffer(ReplayBuffer):
    """
    Smart experience replay buffer for continual learning.

    Integrates prototype-guided sample selection and dynamic replay scheduling.

    Innovations:
    1. Prototype-guided sample selection - stores most representative samples
    2. Importance weighting - based on prediction uncertainty and prototype distance
    3. Dynamic replay ratio - adjusts based on task distribution divergence

    Args:
        max_size: Maximum total buffer size
        prototype_manager: Optional TrajectoryPrototypeManager for guided selection
        diversity_weight: Weight for diversity score in importance calculation
        uncertainty_weight: Weight for uncertainty score
        recency_weight: Weight for recency score
        base_replay_ratio: Base ratio of replay samples in training batch
        min_replay_ratio: Minimum replay ratio
        max_replay_ratio: Maximum replay ratio
    """

    def __init__(
        self,
        max_size: int = 10000,
        prototype_manager: Optional['TrajectoryPrototypeManager'] = None,
        diversity_weight: float = 0.3,
        uncertainty_weight: float = 0.3,
        recency_weight: float = 0.4,
        base_replay_ratio: float = 0.5,
        min_replay_ratio: float = 0.2,
        max_replay_ratio: float = 0.8,
    ):
        super().__init__(max_size=max_size, prioritized=True)
        self.prototype_manager = prototype_manager

        # Importance score weights
        self.diversity_weight = diversity_weight
        self.uncertainty_weight = uncertainty_weight
        self.recency_weight = recency_weight

        # Dynamic replay ratio parameters
        self.base_replay_ratio = base_replay_ratio
        self.min_replay_ratio = min_replay_ratio
        self.max_replay_ratio = max_replay_ratio

        # Task distribution statistics for dynamic replay ratio
        self.task_stats: Dict[int, Dict[str, float]] = {}

    def set_prototype_manager(self, prototype_manager: 'TrajectoryPrototypeManager'):
        """Set the prototype manager (can be set after initialization)."""
        self.prototype_manager = prototype_manager

    # ========== Smart Sample Addition ==========

    def add_with_selection(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        model_output: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Add sample with importance-based selection.

        Importance score = α×diversity + β×uncertainty + γ×recency

        Args:
            x: Input data [seq_len, sensor_dim] or [sensor_dim]
            y: Target RUL value
            task_id: Task/condition identifier
            model_output: Optional (mu, sigma) from model prediction
        """
        score = self._compute_importance_score(x, y, task_id, model_output)
        self.add(x, y, task_id, priority=score)

    def add_batch_with_selection(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: Union[int, torch.Tensor],
        model_outputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Add batch of samples with importance-based selection.

        Args:
            x: Input batch [batch, seq_len, sensor_dim]
            y: Target batch [batch]
            task_id: Task ID (int) or per-sample IDs [batch]
            model_outputs: Optional (mu, sigma) tensors from model
        """
        batch_size = x.size(0)

        if isinstance(task_id, int):
            task_ids = [task_id] * batch_size
        else:
            task_ids = task_id.tolist()

        for i in range(batch_size):
            output = None
            if model_outputs is not None:
                mu, sigma = model_outputs
                output = (mu[i:i+1], sigma[i:i+1])
            self.add_with_selection(x[i], y[i], task_ids[i], output)

    def _compute_importance_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        model_output: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> float:
        """
        Compute sample importance score.

        Score combines:
        1. Diversity: distance from nearest prototype (more diverse = higher)
        2. Uncertainty: model prediction uncertainty (higher uncertainty = higher)
        3. Recency: new samples get bonus (always 1.0)
        """
        score = 0.0

        # 1. Diversity score (distance from prototype)
        if self.prototype_manager is not None:
            y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y)
            nearest = self.prototype_manager.get_nearest_prototype(y_tensor, task_id)
            if nearest is not None:
                _, similarity = nearest
                diversity = 1.0 - similarity  # Further from prototype = more diverse
                score += self.diversity_weight * diversity
            else:
                # No prototype yet, give full diversity score
                score += self.diversity_weight * 1.0
        else:
            score += self.diversity_weight * 0.5  # Default mid-range

        # 2. Uncertainty score
        if model_output is not None:
            mu, sigma = model_output
            sigma_val = sigma.item() if sigma.numel() == 1 else sigma.mean().item()
            # Normalize uncertainty to [0, 1] range
            uncertainty = sigma_val / (sigma_val + 1.0)
            score += self.uncertainty_weight * uncertainty
        else:
            score += self.uncertainty_weight * 0.5  # Default mid-range

        # 3. Recency score (new samples always get full recency)
        score += self.recency_weight * 1.0

        return score

    # ========== Weighted Sampling ==========

    def sample_weighted(
        self,
        batch_size: int,
        current_task: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample with importance weights for weighted loss computation.

        Args:
            batch_size: Number of samples
            current_task: Current task ID (excluded from sampling)
            device: Device to move tensors to

        Returns:
            x: Input batch [batch, ...]
            y: Target batch [batch]
            task_ids: Task IDs [batch]
            weights: Sample weights [batch]
        """
        x, y, task_ids = self.sample_for_replay(batch_size, current_task, device)

        if x.size(0) == 0:
            empty_weights = torch.empty(0, device=device)
            return x, y, task_ids, empty_weights

        weights = self._compute_sample_weights(y)
        if device is not None:
            weights = weights.to(device)

        return x, y, task_ids, weights

    def _compute_sample_weights(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute replay weights based on RUL values.

        Strategy: Middle degradation stages (RUL in [30, 80]) get higher weights
        as they represent critical decision boundaries.

        Args:
            y: RUL values [batch]

        Returns:
            weights: Normalized weights [batch]
        """
        # Gaussian weighting centered at RUL=55 (middle of typical range)
        # Samples near decision boundaries are more important
        rul_weights = 1.0 + 0.5 * torch.exp(-((y - 55.0) / 25.0) ** 2)

        # Normalize to sum to batch size (preserves expected gradient magnitude)
        weights = rul_weights / rul_weights.sum() * len(y)

        return weights

    # ========== Dynamic Replay Ratio ==========

    def update_task_stats(self, task_id: int, rul_values: List[float]):
        """
        Update distribution statistics for a task.

        Args:
            task_id: Task identifier
            rul_values: List of RUL values from the task
        """
        if len(rul_values) == 0:
            return

        self.task_stats[task_id] = {
            'mean': float(np.mean(rul_values)),
            'std': float(np.std(rul_values)),
            'count': len(rul_values),
        }
        logger.debug(
            f"Task {task_id} stats updated: mean={self.task_stats[task_id]['mean']:.2f}, "
            f"std={self.task_stats[task_id]['std']:.2f}, count={len(rul_values)}"
        )

    def compute_replay_ratio(
        self,
        current_task: int,
        previous_tasks: List[int],
    ) -> float:
        """
        Compute dynamic replay ratio based on task distribution divergence.

        Strategy: Larger distribution shift between new and old tasks
        requires more replay to prevent forgetting.

        Args:
            current_task: Current task ID
            previous_tasks: List of completed task IDs

        Returns:
            ratio: Replay ratio in [min_replay_ratio, max_replay_ratio]
        """
        if not previous_tasks or current_task not in self.task_stats:
            return self.base_replay_ratio

        current_stats = self.task_stats[current_task]

        # Compute average distribution divergence with previous tasks
        total_divergence = 0.0
        valid_tasks = 0

        for prev_task in previous_tasks:
            if prev_task in self.task_stats:
                prev_stats = self.task_stats[prev_task]

                # Simple divergence measure based on mean and std differences
                # Normalized by typical RUL range (0-125)
                mean_diff = abs(current_stats['mean'] - prev_stats['mean'])
                std_diff = abs(current_stats['std'] - prev_stats['std'])

                divergence = (mean_diff / 125.0) + (std_diff / 50.0)
                total_divergence += divergence
                valid_tasks += 1

        if valid_tasks == 0:
            return self.base_replay_ratio

        avg_divergence = total_divergence / valid_tasks

        # Scale divergence to replay ratio adjustment
        # Higher divergence -> higher replay ratio
        ratio = self.base_replay_ratio + 0.3 * avg_divergence
        ratio = np.clip(ratio, self.min_replay_ratio, self.max_replay_ratio)

        logger.debug(
            f"Dynamic replay ratio: {ratio:.3f} (divergence={avg_divergence:.3f}, "
            f"current_task={current_task}, prev_tasks={previous_tasks})"
        )

        return float(ratio)

    def get_replay_stats(self) -> Dict:
        """Get statistics about the replay buffer."""
        stats = {
            'total_samples': len(self),
            'task_sizes': self.task_sizes(),
            'task_stats': self.task_stats.copy(),
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
        }
        return stats

    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        state = super().state_dict()
        state['task_stats'] = self.task_stats.copy()
        state['diversity_weight'] = self.diversity_weight
        state['uncertainty_weight'] = self.uncertainty_weight
        state['recency_weight'] = self.recency_weight
        state['base_replay_ratio'] = self.base_replay_ratio
        state['min_replay_ratio'] = self.min_replay_ratio
        state['max_replay_ratio'] = self.max_replay_ratio
        return state

    def load_state_dict(self, state: Dict):
        """Load state from checkpoint."""
        super().load_state_dict(state)
        self.task_stats = state.get('task_stats', {})
        self.diversity_weight = state.get('diversity_weight', 0.3)
        self.uncertainty_weight = state.get('uncertainty_weight', 0.3)
        self.recency_weight = state.get('recency_weight', 0.4)
        self.base_replay_ratio = state.get('base_replay_ratio', 0.5)
        self.min_replay_ratio = state.get('min_replay_ratio', 0.2)
        self.max_replay_ratio = state.get('max_replay_ratio', 0.8)
