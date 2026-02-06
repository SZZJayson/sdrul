"""
Gaussian RUL prediction head for distributional RUL prediction.

Outputs mean and standard deviation for uncertainty-aware RUL prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GaussianRULHead(nn.Module):
    """
    RUL prediction head that outputs Gaussian distribution parameters.

    Predicts both mean (mu) and standard deviation (sigma) for
    uncertainty-aware RUL prediction and distributional distillation.

    Args:
        d_model: Input feature dimension (default: 256)
        hidden_dim: Hidden layer dimension (default: 128)
        min_sigma: Minimum sigma value for numerical stability (default: 0.1)
        max_sigma: Maximum sigma value (default: 50.0)
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        min_sigma: float = 0.1,
        max_sigma: float = 50.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Mean prediction head
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Log-sigma prediction head (predict log for numerical stability)
        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        return_log_sigma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict RUL distribution parameters.

        Args:
            features: Input features [batch, seq_len, d_model] or [batch, d_model]
            return_log_sigma: If True, return log_sigma instead of sigma

        Returns:
            mu: Mean RUL prediction [batch]
            sigma: Standard deviation [batch] (or log_sigma if return_log_sigma=True)
        """
        # Handle both sequence and pooled features
        if features.dim() == 3:
            # Use last time step
            features = features[:, -1, :]  # [batch, d_model]

        # Shared processing
        shared_features = self.shared(features)  # [batch, hidden_dim]

        # Predict mean
        mu = self.mu_head(shared_features).squeeze(-1)  # [batch]

        # Predict log_sigma and convert to sigma
        log_sigma = self.log_sigma_head(shared_features).squeeze(-1)  # [batch]

        if return_log_sigma:
            return mu, log_sigma

        # Convert to sigma with bounds
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, self.min_sigma, self.max_sigma)

        return mu, sigma

    def nll_loss(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for Gaussian distribution.

        Args:
            mu: Predicted mean [batch]
            sigma: Predicted std [batch]
            target: Target RUL values [batch]

        Returns:
            loss: NLL loss (scalar)
        """
        # Gaussian NLL: 0.5 * (log(2*pi*sigma^2) + (y-mu)^2/sigma^2)
        variance = sigma ** 2
        nll = 0.5 * (torch.log(2 * torch.pi * variance) + (target - mu) ** 2 / variance)
        return nll.mean()

    def sample(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from the predicted distribution.

        Args:
            mu: Mean [batch]
            sigma: Std [batch]
            num_samples: Number of samples per prediction

        Returns:
            samples: Sampled RUL values [batch, num_samples]
        """
        # Reparameterization trick
        eps = torch.randn(mu.size(0), num_samples, device=mu.device)
        samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps
        return samples


class DeterministicRULHead(nn.Module):
    """
    Simple deterministic RUL prediction head.

    For comparison with Gaussian head or when uncertainty is not needed.

    Args:
        d_model: Input feature dimension (default: 256)
        hidden_dim: Hidden layer dimension (default: 128)
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict RUL.

        Args:
            features: Input features [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            rul: Predicted RUL [batch]
        """
        if features.dim() == 3:
            features = features[:, -1, :]

        return self.head(features).squeeze(-1)
