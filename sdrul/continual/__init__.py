"""
Continual learning modules for RUL prediction.
"""

from .ewc import EWCRegularizer
from .replay import ReplayBuffer, SmartReplayBuffer
from .trainer import ContinualTrainer, ContinualTrainerConfig

__all__ = [
    'EWCRegularizer',
    'ReplayBuffer',
    'SmartReplayBuffer',
    'ContinualTrainer',
    'ContinualTrainerConfig',
]
