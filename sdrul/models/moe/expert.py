"""
Adapter-based expert modules for DSA-MoE.

Lightweight experts using bottleneck architecture for efficient
multi-condition, multi-stage knowledge organization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdapterExpert(nn.Module):
    """
    Lightweight Adapter expert with bottleneck architecture.

    Architecture: LayerNorm -> Down(d_model->bottleneck) -> GELU -> Up(bottleneck->d_model) -> scale*out + residual

    This design follows the adapter pattern from parameter-efficient fine-tuning,
    allowing each expert to specialize with minimal parameters.

    Args:
        d_model: Input/output dimension (default: 256)
        bottleneck_dim: Bottleneck dimension (default: 64)
        scale: Output scaling factor (default: 1.0)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        bottleneck_dim: int = 64,
        scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Down projection
        self.down_proj = nn.Linear(d_model, bottleneck_dim)

        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize up_proj to near-zero for stable training start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation.

        Args:
            x: Input features [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            output: Transformed features with residual [batch, seq_len, d_model]
        """
        residual = x

        # Normalize
        x = self.norm(x)

        # Down projection
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Up projection
        x = self.up_proj(x)

        # Scale and add residual
        return self.scale * x + residual


class ExpertGroup(nn.Module):
    """
    Group of experts for a specific condition or stage.

    Manages multiple AdapterExperts and provides weighted combination.

    Args:
        num_experts: Number of experts in the group
        d_model: Feature dimension
        bottleneck_dim: Bottleneck dimension for each expert
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int = 256,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            AdapterExpert(d_model, bottleneck_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply experts with optional weighting.

        Args:
            x: Input features [batch, seq_len, d_model]
            weights: Expert weights [batch, num_experts] or None for uniform

        Returns:
            output: Weighted expert outputs [batch, seq_len, d_model]
        """
        if weights is None:
            # Uniform weighting
            weights = torch.ones(x.size(0), self.num_experts, device=x.device)
            weights = weights / self.num_experts

        # Compute all expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=-1)  # [batch, seq_len, d_model, num_experts]

        # Weighted combination
        weights = weights.view(weights.size(0), 1, 1, -1)  # [batch, 1, 1, num_experts]
        output = (expert_outputs * weights).sum(dim=-1)  # [batch, seq_len, d_model]

        return output

    def get_expert(self, idx: int) -> AdapterExpert:
        """Get a specific expert by index."""
        return self.experts[idx]
