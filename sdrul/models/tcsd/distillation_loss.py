"""
Distributional RUL loss for TCSD.

Combines KL divergence and Wasserstein distance for distilling
Gaussian RUL distributions from teacher to student.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DistributionalRULLoss(nn.Module):
    """
    Distributional loss for RUL prediction distillation.

    Combines:
    - KL divergence between Gaussian distributions
    - Wasserstein-2 distance for distribution matching

    For Gaussian distributions N(mu_t, sigma_t^2) and N(mu_s, sigma_s^2):
    - KL(teacher || student) = log(sigma_s/sigma_t) + (sigma_t^2 + (mu_t-mu_s)^2)/(2*sigma_s^2) - 0.5
    - W2^2 = (mu_t - mu_s)^2 + (sigma_t - sigma_s)^2

    Args:
        alpha: Weight for KL divergence (default: 0.5)
        min_sigma: Minimum sigma for numerical stability (default: 0.1)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_sigma: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.min_sigma = min_sigma
        self.reduction = reduction

    def gaussian_kl_divergence(
        self,
        mu_t: torch.Tensor,
        sigma_t: torch.Tensor,
        mu_s: torch.Tensor,
        sigma_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(teacher || student) for Gaussians.

        Args:
            mu_t: Teacher mean [batch]
            sigma_t: Teacher std [batch]
            mu_s: Student mean [batch]
            sigma_s: Student std [batch]

        Returns:
            kl: KL divergence [batch]
        """
        # Clamp sigmas for stability
        sigma_t = torch.clamp(sigma_t, min=self.min_sigma)
        sigma_s = torch.clamp(sigma_s, min=self.min_sigma)

        # KL divergence for univariate Gaussians
        # KL = log(sigma_s/sigma_t) + (sigma_t^2 + (mu_t - mu_s)^2) / (2 * sigma_s^2) - 0.5
        kl = (
            torch.log(sigma_s / sigma_t)
            + (sigma_t ** 2 + (mu_t - mu_s) ** 2) / (2 * sigma_s ** 2)
            - 0.5
        )

        return kl

    def wasserstein_2_squared(
        self,
        mu_t: torch.Tensor,
        sigma_t: torch.Tensor,
        mu_s: torch.Tensor,
        sigma_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute squared Wasserstein-2 distance for Gaussians.

        For univariate Gaussians:
        W2^2 = (mu_t - mu_s)^2 + (sigma_t - sigma_s)^2

        Args:
            mu_t: Teacher mean [batch]
            sigma_t: Teacher std [batch]
            mu_s: Student mean [batch]
            sigma_s: Student std [batch]

        Returns:
            w2_sq: Squared W2 distance [batch]
        """
        mean_diff_sq = (mu_t - mu_s) ** 2
        std_diff_sq = (sigma_t - sigma_s) ** 2

        return mean_diff_sq + std_diff_sq

    def forward(
        self,
        teacher_output: Tuple[torch.Tensor, torch.Tensor],
        student_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute distributional distillation loss.

        Args:
            teacher_output: (mu_t, sigma_t) from teacher
            student_output: (mu_s, sigma_s) from student

        Returns:
            loss: Combined distillation loss
        """
        mu_t, sigma_t = teacher_output
        mu_s, sigma_s = student_output

        # KL divergence
        kl_loss = self.gaussian_kl_divergence(mu_t, sigma_t, mu_s, sigma_s)

        # Wasserstein distance
        w2_loss = self.wasserstein_2_squared(mu_t, sigma_t, mu_s, sigma_s)

        # Combined loss
        loss = self.alpha * kl_loss + (1 - self.alpha) * w2_loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward_with_components(
        self,
        teacher_output: Tuple[torch.Tensor, torch.Tensor],
        student_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss with individual components for logging.

        Returns:
            total_loss: Combined loss
            kl_loss: KL divergence component
            w2_loss: Wasserstein component
        """
        mu_t, sigma_t = teacher_output
        mu_s, sigma_s = student_output

        kl_loss = self.gaussian_kl_divergence(mu_t, sigma_t, mu_s, sigma_s)
        w2_loss = self.wasserstein_2_squared(mu_t, sigma_t, mu_s, sigma_s)

        total_loss = self.alpha * kl_loss + (1 - self.alpha) * w2_loss

        if self.reduction == 'mean':
            return total_loss.mean(), kl_loss.mean(), w2_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum(), kl_loss.sum(), w2_loss.sum()
        else:
            return total_loss, kl_loss, w2_loss


class RULRegressionLoss(nn.Module):
    """
    Combined regression loss for RUL prediction.

    Supports both deterministic and probabilistic predictions.

    Args:
        use_asymmetric: Use asymmetric loss (penalize late predictions more)
        asymmetric_factor: Factor for asymmetric penalty (default: 2.0)
    """

    def __init__(
        self,
        use_asymmetric: bool = True,
        asymmetric_factor: float = 2.0,
    ):
        super().__init__()
        self.use_asymmetric = use_asymmetric
        self.asymmetric_factor = asymmetric_factor

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute RUL regression loss.

        Args:
            pred: Predicted RUL [batch]
            target: Target RUL [batch]
            sigma: Optional predicted uncertainty [batch]

        Returns:
            loss: Regression loss
        """
        error = pred - target

        if self.use_asymmetric:
            # Asymmetric loss: penalize late predictions (negative error) more
            # Late prediction = pred < target = negative error
            weights = torch.where(
                error < 0,
                torch.tensor(self.asymmetric_factor, device=error.device),
                torch.tensor(1.0, device=error.device),
            )
            loss = weights * (error ** 2)
        else:
            loss = error ** 2

        if sigma is not None:
            # Gaussian NLL with learned uncertainty
            variance = sigma ** 2 + 1e-6
            loss = 0.5 * (torch.log(variance) + loss / variance)

        return loss.mean()


class DistillationWithSupervisionLoss(nn.Module):
    """
    Combined loss for supervised learning with distillation.

    Balances:
    - Supervised loss on ground truth
    - Distillation loss from teacher

    Args:
        distillation_weight: Weight for distillation loss (default: 1.0)
        supervision_weight: Weight for supervised loss (default: 1.0)
        alpha_kl: KL weight in distillation (default: 0.5)
    """

    def __init__(
        self,
        distillation_weight: float = 1.0,
        supervision_weight: float = 1.0,
        alpha_kl: float = 0.5,
    ):
        super().__init__()
        self.distillation_weight = distillation_weight
        self.supervision_weight = supervision_weight

        self.distillation_loss = DistributionalRULLoss(alpha=alpha_kl)
        self.supervision_loss = RULRegressionLoss()

    def forward(
        self,
        student_output: Tuple[torch.Tensor, torch.Tensor],
        teacher_output: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            student_output: (mu_s, sigma_s) from student
            teacher_output: (mu_t, sigma_t) from teacher
            target: Ground truth RUL [batch]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        mu_s, sigma_s = student_output

        # Distillation loss
        dist_loss = self.distillation_loss(teacher_output, student_output)

        # Supervised loss
        sup_loss = self.supervision_loss(mu_s, target, sigma_s)

        # Combined
        total_loss = (
            self.distillation_weight * dist_loss
            + self.supervision_weight * sup_loss
        )

        loss_dict = {
            'distillation_loss': dist_loss.item(),
            'supervision_loss': sup_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict
