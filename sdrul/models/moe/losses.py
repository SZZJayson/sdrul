"""
Loss functions for DSA-MoE training.

Includes load balancing loss to prevent expert collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def load_balance_loss(
    expert_loads: torch.Tensor,
    target_load: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute load balancing loss to prevent expert collapse.

    Encourages uniform load distribution across experts by penalizing
    deviation from target load.

    Args:
        expert_loads: Expert load distribution [batch, num_experts] or [num_experts]
                     Values should sum to 1 (routing probabilities)
        target_load: Target load per expert (default: 1/num_experts)

    Returns:
        loss: Load balancing loss (scalar)
    """
    if expert_loads.dim() == 1:
        expert_loads = expert_loads.unsqueeze(0)

    num_experts = expert_loads.size(-1)

    if target_load is None:
        target_load = 1.0 / num_experts

    # Compute mean load across batch
    mean_loads = expert_loads.mean(dim=0)  # [num_experts]

    # Squared deviation from target
    loss = ((mean_loads - target_load) ** 2).sum()

    return loss


def importance_loss(
    router_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute importance loss based on router logits.

    Encourages all experts to be used by penalizing uneven importance scores.

    Args:
        router_logits: Raw router logits [batch, num_experts]
        temperature: Softmax temperature

    Returns:
        loss: Importance loss (scalar)
    """
    # Compute importance as sum of routing probabilities
    probs = F.softmax(router_logits / temperature, dim=-1)
    importance = probs.sum(dim=0)  # [num_experts]

    # Normalize
    importance = importance / importance.sum()

    # Compute coefficient of variation
    mean_importance = importance.mean()
    std_importance = importance.std()
    cv = std_importance / (mean_importance + 1e-8)

    return cv ** 2


class LoadBalanceLoss(nn.Module):
    """
    Module wrapper for load balancing loss.

    Combines load balancing and importance losses with configurable weights.

    Args:
        load_balance_weight: Weight for load balance loss (default: 0.01)
        importance_weight: Weight for importance loss (default: 0.01)
        num_experts: Number of experts (for target computation)
    """

    def __init__(
        self,
        load_balance_weight: float = 0.01,
        importance_weight: float = 0.01,
        num_experts: Optional[int] = None,
    ):
        super().__init__()
        self.load_balance_weight = load_balance_weight
        self.importance_weight = importance_weight
        self.num_experts = num_experts

    def forward(
        self,
        expert_loads: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined load balancing loss.

        Args:
            expert_loads: Expert load distribution [batch, num_experts]
            router_logits: Optional raw router logits [batch, num_experts]

        Returns:
            loss: Combined loss (scalar)
        """
        loss = torch.tensor(0.0, device=expert_loads.device)

        # Load balance loss
        if self.load_balance_weight > 0:
            lb_loss = load_balance_loss(expert_loads)
            loss = loss + self.load_balance_weight * lb_loss

        # Importance loss
        if self.importance_weight > 0 and router_logits is not None:
            imp_loss = importance_loss(router_logits)
            loss = loss + self.importance_weight * imp_loss

        return loss


def expert_diversity_loss(
    expert_outputs: torch.Tensor,
    min_diversity: float = 0.1,
) -> torch.Tensor:
    """
    Encourage diversity among expert outputs.

    Penalizes experts that produce too similar outputs.

    Args:
        expert_outputs: Outputs from all experts [batch, seq_len, d_model, num_experts]
        min_diversity: Minimum desired diversity (cosine distance)

    Returns:
        loss: Diversity loss (scalar)
    """
    batch_size, seq_len, d_model, num_experts = expert_outputs.shape

    # Flatten spatial dimensions
    outputs_flat = expert_outputs.view(batch_size, -1, num_experts)  # [batch, seq*d, num_experts]

    # Compute pairwise cosine similarity between experts
    outputs_norm = F.normalize(outputs_flat, dim=1)  # [batch, seq*d, num_experts]

    # [batch, num_experts, num_experts]
    similarity = torch.bmm(
        outputs_norm.transpose(1, 2),
        outputs_norm
    )

    # Mask diagonal (self-similarity)
    mask = torch.eye(num_experts, device=similarity.device).bool()
    similarity = similarity.masked_fill(mask.unsqueeze(0), 0)

    # Penalize high similarity (low diversity)
    # We want similarity < (1 - min_diversity)
    threshold = 1 - min_diversity
    excess_similarity = F.relu(similarity - threshold)

    return excess_similarity.mean()
