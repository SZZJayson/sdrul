"""
Elastic Weight Consolidation (EWC) for continual learning.

Implements EWC regularization to prevent catastrophic forgetting
by penalizing changes to important parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Iterator
from copy import deepcopy


class EWCRegularizer:
    """
    Elastic Weight Consolidation regularizer.

    Computes Fisher information matrix to identify important parameters
    and penalizes changes to them during training on new tasks.

    Args:
        model: The model to regularize
        ewc_lambda: Regularization strength (default: 1000)
        online: Use online EWC (running average of Fisher) (default: True)
        gamma: Decay factor for online EWC (default: 0.9)
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        online: bool = True,
        gamma: float = 0.9,
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        # Storage for Fisher information and optimal parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

        # Task counter
        self.num_tasks = 0

    def compute_fisher(
        self,
        dataloader: Iterator,
        num_samples: Optional[int] = None,
        empirical: bool = True,
    ):
        """
        Compute Fisher information matrix from data.

        Args:
            dataloader: DataLoader providing (x, y) tuples
            num_samples: Number of samples to use (None = all)
            empirical: Use empirical Fisher (gradients of loss) vs true Fisher
        """
        self.model.eval()

        # Initialize Fisher accumulator
        fisher_acc = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_samples is not None and n_samples >= num_samples:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    x = batch[0]
                    y = None
            else:
                x = batch
                y = None

            x = x.to(next(self.model.parameters()).device)
            if y is not None:
                y = y.to(x.device)

            batch_size = x.size(0)
            n_samples += batch_size

            # Compute gradients
            self.model.zero_grad()

            if empirical and y is not None:
                # Empirical Fisher: use actual loss
                output = self.model(x)
                # Handle models that return (output, aux) tuple
                if isinstance(output, tuple) and len(output) == 2:
                    pred, aux = output
                    if isinstance(pred, tuple) and len(pred) == 2:
                        # (mu, sigma), aux format
                        mu, sigma = pred
                        loss = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (y - mu) ** 2 / (sigma ** 2 + 1e-6))
                        loss = loss.mean()
                    elif isinstance(pred, torch.Tensor):
                        loss = F.mse_loss(pred, y)
                    else:
                        loss = F.mse_loss(output[0], y)
                else:
                    loss = F.mse_loss(output, y)
            else:
                # True Fisher: use log-likelihood of predictions
                output = self.model(x)
                # Handle models that return (output, aux) tuple
                if isinstance(output, tuple) and len(output) == 2:
                    pred, aux = output
                    if isinstance(pred, tuple) and len(pred) == 2:
                        mu, sigma = pred
                        samples = mu + sigma * torch.randn_like(mu)
                        loss = 0.5 * (torch.log(sigma ** 2 + 1e-6) + (samples - mu) ** 2 / (sigma ** 2 + 1e-6))
                        loss = loss.mean()
                    elif isinstance(pred, torch.Tensor):
                        loss = (pred ** 2).mean()
                    else:
                        loss = (output[0] ** 2).mean()
                else:
                    # For deterministic output, use squared output as proxy
                    loss = (output ** 2).mean()

            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_acc[name] += param.grad.data ** 2 * batch_size

        # Normalize by number of samples
        for name in fisher_acc:
            fisher_acc[name] /= n_samples

        # Update Fisher information
        if self.online and self.num_tasks > 0:
            # Online EWC: running average
            for name in fisher_acc:
                if name in self.fisher:
                    self.fisher[name] = (
                        self.gamma * self.fisher[name]
                        + (1 - self.gamma) * fisher_acc[name]
                    )
                else:
                    self.fisher[name] = fisher_acc[name]
        else:
            # Standard EWC: replace
            self.fisher = fisher_acc

        # Store optimal parameters
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.num_tasks += 1
        self.model.train()

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Returns:
            loss: EWC penalty term
        """
        if not self.fisher:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.optimal_params:
                # Penalty for deviating from optimal parameters
                # weighted by Fisher information
                diff = param - self.optimal_params[name]
                loss += (self.fisher[name] * diff ** 2).sum()

        return 0.5 * self.ewc_lambda * loss

    def penalty(self) -> torch.Tensor:
        """Alias for ewc_loss()."""
        return self.ewc_loss()

    def consolidate(self, dataloader: Iterator, num_samples: Optional[int] = None):
        """
        Consolidate knowledge after training on a task.

        This computes Fisher information and stores optimal parameters.

        Args:
            dataloader: DataLoader for the completed task
            num_samples: Number of samples to use for Fisher computation
        """
        self.compute_fisher(dataloader, num_samples)

    def get_importance(self, name: str) -> Optional[torch.Tensor]:
        """Get Fisher information (importance) for a parameter."""
        return self.fisher.get(name)

    def get_optimal_param(self, name: str) -> Optional[torch.Tensor]:
        """Get stored optimal parameter value."""
        return self.optimal_params.get(name)

    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            'fisher': {k: v.cpu() for k, v in self.fisher.items()},
            'optimal_params': {k: v.cpu() for k, v in self.optimal_params.items()},
            'num_tasks': self.num_tasks,
            'ewc_lambda': self.ewc_lambda,
            'online': self.online,
            'gamma': self.gamma,
        }

    def load_state_dict(self, state: Dict, device: Optional[torch.device] = None):
        """Load state from checkpoint."""
        if device is None:
            device = next(self.model.parameters()).device

        self.fisher = {k: v.to(device) for k, v in state['fisher'].items()}
        self.optimal_params = {k: v.to(device) for k, v in state['optimal_params'].items()}
        self.num_tasks = state['num_tasks']
        self.ewc_lambda = state.get('ewc_lambda', self.ewc_lambda)
        self.online = state.get('online', self.online)
        self.gamma = state.get('gamma', self.gamma)


class EWCLoss(nn.Module):
    """
    EWC loss as a PyTorch module.

    Wraps EWCRegularizer for easier integration with training loops.

    Args:
        model: Model to regularize
        ewc_lambda: Regularization strength
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        super().__init__()
        self.regularizer = EWCRegularizer(model, ewc_lambda)

    def forward(self) -> torch.Tensor:
        """Compute EWC loss."""
        return self.regularizer.ewc_loss()

    def consolidate(self, dataloader: Iterator, num_samples: Optional[int] = None):
        """Consolidate after task completion."""
        self.regularizer.consolidate(dataloader, num_samples)
