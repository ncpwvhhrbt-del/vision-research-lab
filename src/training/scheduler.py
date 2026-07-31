"""Learning-rate scheduler construction for controlled experiments."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from .config import TrainingConfig


def build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    total_epochs: int,
) -> LRScheduler | None:
    """Build one epoch-level scheduler selected by configuration.

    ``None`` means fixed learning rate. Otherwise the caller invokes
    ``scheduler.step()`` exactly once after validation at each epoch end.
    """
    config.validate()
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if config.scheduler_name == "none":
        return None
    if config.scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if config.scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=config.scheduler_min_lr,
        )
    raise AssertionError("TrainingConfig already validates scheduler_name.")
