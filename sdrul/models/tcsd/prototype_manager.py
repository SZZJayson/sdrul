"""
Trajectory Prototype Manager for TCSD.

Manages trajectory prototypes with K-means initialization
and trajectory shape encoding integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.trajectory import TrajectoryShapeEncoder


class PrototypeBuffer:
    """
    Buffer for storing degradation trajectory prototypes.

    Prototypes are stored per operating condition and used for
    conditioning during replay and distillation.
    """

    def __init__(
        self,
        num_conditions: int,
        max_prototypes_per_condition: int = 10,
        novelty_threshold: float = 0.7,
    ):
        self.num_conditions = num_conditions
        self.max_prototypes = max_prototypes_per_condition
        self.novelty_threshold = novelty_threshold

        # Buffer: dict mapping condition_id -> list of (trajectory, encoding)
        self.buffer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            i: [] for i in range(num_conditions)
        }

    def add(
        self,
        trajectory: torch.Tensor,
        encoding: torch.Tensor,
        condition_id: int,
        encoder_model: Optional[nn.Module] = None,
    ) -> bool:
        """
        Add a trajectory to the buffer.

        Args:
            trajectory: RUL trajectory [seq_len]
            encoding: Trajectory encoding [encoding_dim]
            condition_id: Operating condition
            encoder_model: Optional encoder for computing similarities

        Returns:
            is_novel: Whether the trajectory was novel (added to buffer)
        """
        if condition_id >= self.num_conditions:
            self.num_conditions = condition_id + 1
            self.buffer[condition_id] = []

        condition_buffer = self.buffer[condition_id]

        # Check novelty if buffer not empty
        if len(condition_buffer) > 0:
            similarities = []
            for _, existing_enc in condition_buffer:
                sim = F.cosine_similarity(
                    encoding.unsqueeze(0),
                    existing_enc.unsqueeze(0),
                ).item()
                similarities.append(sim)

            max_sim = max(similarities)
            if max_sim > self.novelty_threshold:
                return False

        # Add to buffer
        condition_buffer.append((trajectory.detach(), encoding.detach()))

        # Prune if exceeding max
        if len(condition_buffer) > self.max_prototypes:
            condition_buffer.pop(0)

        return True

    def get_prototypes(self, condition_id: int) -> List[torch.Tensor]:
        """Get all prototypes for a condition."""
        if condition_id not in self.buffer:
            return []
        return [traj for traj, _ in self.buffer[condition_id]]

    def get_random_prototype(self, condition_id: int) -> Optional[torch.Tensor]:
        """Get a random prototype for a condition."""
        prototypes = self.get_prototypes(condition_id)
        if not prototypes:
            return None
        return prototypes[np.random.randint(len(prototypes))]

    def __len__(self) -> int:
        """Total number of prototypes across all conditions."""
        return sum(len(v) for v in self.buffer.values())

    def __getitem__(self, condition_id: int) -> List:
        """Get prototypes for a condition."""
        return self.buffer.get(condition_id, [])


class TrajectoryPrototypeManager(PrototypeBuffer):
    """
    Manages trajectory prototypes for TCSD.

    Extends the existing PrototypeBuffer with:
    - K-means clustering for prototype initialization
    - EMA updates for online learning
    - Integration with TrajectoryShapeEncoder

    Args:
        trajectory_encoder: TrajectoryShapeEncoder instance
        num_prototypes_per_condition: Number of prototypes per condition (default: 5)
        num_conditions: Number of operating conditions (default: 6)
        ema_decay: EMA decay rate for prototype updates (default: 0.9)
        min_samples_for_init: Minimum samples needed for K-means init (default: 10)
    """

    def __init__(
        self,
        trajectory_encoder: TrajectoryShapeEncoder,
        num_prototypes_per_condition: int = 5,
        num_conditions: int = 6,
        ema_decay: float = 0.9,
        min_samples_for_init: int = 10,
    ):
        # Initialize parent PrototypeBuffer
        super().__init__(
            num_conditions=num_conditions,
            max_prototypes_per_condition=num_prototypes_per_condition,
            novelty_threshold=0.7,
        )

        self.encoder = trajectory_encoder
        self.num_prototypes = num_prototypes_per_condition
        self.ema_decay = ema_decay
        self.min_samples_for_init = min_samples_for_init

        # Track initialization status
        self.initialized: Dict[int, bool] = {i: False for i in range(num_conditions)}

        # Temporary storage for K-means initialization
        self._init_buffer: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(num_conditions)
        }

    def fit_kmeans(
        self,
        trajectories: List[torch.Tensor],
        condition_id: int,
        device: Optional[torch.device] = None,
    ) -> bool:
        """
        Initialize prototypes using K-means clustering.

        Args:
            trajectories: List of RUL trajectories [seq_len] each
            condition_id: Operating condition ID
            device: Device for tensors

        Returns:
            success: Whether initialization was successful
        """
        if len(trajectories) < self.num_prototypes:
            return False

        if device is None:
            device = trajectories[0].device if isinstance(trajectories[0], torch.Tensor) else torch.device('cpu')

        # Encode all trajectories
        with torch.no_grad():
            encodings = []
            for traj in trajectories:
                if not isinstance(traj, torch.Tensor):
                    traj = torch.tensor(traj, dtype=torch.float32, device=device)
                if traj.dim() == 1:
                    traj = traj.unsqueeze(0)
                enc = self.encoder(traj).squeeze(0)
                encodings.append(enc.cpu().numpy())

        encodings_np = np.stack(encodings)

        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.num_prototypes,
            random_state=42,
            n_init=10,
        )
        kmeans.fit(encodings_np)

        # Clear existing prototypes for this condition (using parent's buffer)
        self.buffer[condition_id] = []

        # Find representative trajectory for each cluster
        for cluster_idx in range(self.num_prototypes):
            # Find trajectory closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            distances = np.linalg.norm(encodings_np - cluster_center, axis=1)
            closest_idx = np.argmin(distances)

            traj = trajectories[closest_idx]
            if not isinstance(traj, torch.Tensor):
                traj = torch.tensor(traj, dtype=torch.float32, device=device)

            encoding = torch.tensor(cluster_center, dtype=torch.float32, device=device)

            # Use parent's buffer structure
            self.buffer[condition_id].append((traj.clone(), encoding))

        self.initialized[condition_id] = True
        return True

    def add_trajectory(
        self,
        trajectory: torch.Tensor,
        condition_id: int,
    ) -> bool:
        """
        Add a trajectory for potential prototype initialization or update.

        Args:
            trajectory: RUL trajectory [seq_len] or [1, seq_len]
            condition_id: Operating condition ID

        Returns:
            added: Whether trajectory was added to buffer
        """
        if condition_id >= self.num_conditions:
            # Expand for new condition
            self.num_conditions = condition_id + 1
            self.buffer[condition_id] = []
            self.initialized[condition_id] = False
            self._init_buffer[condition_id] = []

        if trajectory.dim() == 2:
            trajectory = trajectory.squeeze(0)

        if not self.initialized.get(condition_id, False):
            # Add to initialization buffer
            self._init_buffer[condition_id].append(trajectory.detach().clone())

            # Try to initialize if enough samples
            if len(self._init_buffer[condition_id]) >= self.min_samples_for_init:
                success = self.fit_kmeans(
                    self._init_buffer[condition_id],
                    condition_id,
                    trajectory.device,
                )
                if success:
                    self._init_buffer[condition_id] = []  # Clear buffer
                return success
            return True
        else:
            # Update existing prototypes via EMA using parent's update_online
            self._update_prototypes_ema(trajectory, condition_id)
            return True

    def _update_prototypes_ema(
        self,
        trajectory: torch.Tensor,
        condition_id: int,
    ):
        """Update prototypes using exponential moving average."""
        if not self.buffer.get(condition_id):
            return

        with torch.no_grad():
            # Encode new trajectory
            if trajectory.dim() == 1:
                trajectory = trajectory.unsqueeze(0)
            new_encoding = self.encoder(trajectory).squeeze(0)

            # Find most similar prototype
            similarities = []
            for _, proto_enc in self.buffer[condition_id]:
                sim = F.cosine_similarity(
                    new_encoding.unsqueeze(0),
                    proto_enc.unsqueeze(0),
                ).item()
                similarities.append(sim)

            most_similar_idx = int(np.argmax(similarities))

            # EMA update of encoding
            old_traj, old_enc = self.buffer[condition_id][most_similar_idx]
            new_enc = self.ema_decay * old_enc + (1 - self.ema_decay) * new_encoding
            self.buffer[condition_id][most_similar_idx] = (old_traj, new_enc)

    def get_prototype(
        self,
        condition_id: int,
        prototype_idx: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a prototype for a condition.

        Args:
            condition_id: Operating condition ID
            prototype_idx: Specific prototype index (None for random)

        Returns:
            (trajectory, encoding) tuple or None if not available
        """
        if condition_id not in self.buffer or not self.buffer[condition_id]:
            return None

        if prototype_idx is None:
            prototype_idx = np.random.randint(len(self.buffer[condition_id]))

        return self.buffer[condition_id][prototype_idx]

    def get_random_prototype(
        self,
        condition_id: int,
    ) -> Optional[torch.Tensor]:
        """
        Get a random prototype encoding for a condition.

        Args:
            condition_id: Operating condition ID

        Returns:
            Prototype encoding [feature_dim] or None
        """
        result = self.get_prototype(condition_id)
        if result is None:
            return None
        return result[1]  # Return encoding

    def get_all_prototypes(
        self,
        condition_id: int,
    ) -> List[torch.Tensor]:
        """
        Get all prototype encodings for a condition.

        Args:
            condition_id: Operating condition ID

        Returns:
            List of prototype encodings
        """
        if condition_id not in self.buffer:
            return []
        return [enc for _, enc in self.buffer[condition_id]]

    def get_nearest_prototype(
        self,
        trajectory: torch.Tensor,
        condition_id: int,
    ) -> Optional[Tuple[torch.Tensor, float]]:
        """
        Get the nearest prototype to a given trajectory.

        Args:
            trajectory: Query trajectory [seq_len]
            condition_id: Operating condition ID

        Returns:
            (prototype_encoding, similarity) or None
        """
        if not self.buffer.get(condition_id):
            return None

        with torch.no_grad():
            if trajectory.dim() == 1:
                trajectory = trajectory.unsqueeze(0)
            query_enc = self.encoder(trajectory).squeeze(0)

            best_sim = -1.0
            best_proto = None

            for _, proto_enc in self.buffer[condition_id]:
                sim = F.cosine_similarity(
                    query_enc.unsqueeze(0),
                    proto_enc.unsqueeze(0),
                ).item()
                if sim > best_sim:
                    best_sim = sim
                    best_proto = proto_enc

            return (best_proto, best_sim) if best_proto is not None else None

    def get_mean_prototype(self) -> Optional[torch.Tensor]:
        """
        Get mean prototype across all conditions (fallback when specific condition unavailable).

        Returns:
            Mean prototype encoding [feature_dim] or None if no prototypes exist
        """
        all_encodings = []
        for condition_id in self.buffer:
            for _, enc in self.buffer[condition_id]:
                all_encodings.append(enc)

        if not all_encodings:
            return None

        return torch.stack(all_encodings).mean(dim=0)

    def is_initialized(self, condition_id: int) -> bool:
        """Check if prototypes are initialized for a condition."""
        return self.initialized.get(condition_id, False)

    def num_prototypes_for_condition(self, condition_id: int) -> int:
        """Get number of prototypes for a condition."""
        return len(self.buffer.get(condition_id, []))

    def save_state(self) -> Dict:
        """Save manager state for checkpointing."""
        state = {
            'num_prototypes': self.num_prototypes,
            'num_conditions': self.num_conditions,
            'ema_decay': self.ema_decay,
            'initialized': self.initialized.copy(),
            'buffer': {},
        }

        for cond_id, protos in self.buffer.items():
            state['buffer'][cond_id] = [
                (traj.cpu(), enc.cpu()) for traj, enc in protos
            ]

        return state

    def load_state(self, state: Dict, device: torch.device = None):
        """Load manager state from checkpoint."""
        self.num_prototypes = state['num_prototypes']
        self.num_conditions = state['num_conditions']
        self.ema_decay = state['ema_decay']
        self.initialized = state['initialized'].copy()

        self.buffer = {}
        for cond_id, protos in state['buffer'].items():
            cond_id = int(cond_id)
            self.buffer[cond_id] = []
            for traj, enc in protos:
                if device is not None:
                    traj = traj.to(device)
                    enc = enc.to(device)
                self.buffer[cond_id].append((traj, enc))
