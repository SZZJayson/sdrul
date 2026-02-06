"""
TCSD (Trajectory-Conditioned Self-Distillation) modules.
"""

from .prototype_manager import TrajectoryPrototypeManager
from .distillation_loss import DistributionalRULLoss
from .teacher_student import FiLMLayer, TeacherBranch, StudentBranch
from .tcsd_module import TCSD, TCSDConfig

__all__ = [
    'TrajectoryPrototypeManager',
    'DistributionalRULLoss',
    'FiLMLayer',
    'TeacherBranch',
    'StudentBranch',
    'TCSD',
    'TCSDConfig',
]
