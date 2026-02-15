"""
Dynamic Learning Rate Scheduler.

Moved from sac_v430_training_optimizations.py in 063# SAC cleanup.
Plateau detection + recovery scheduling for PyTorch optimizers.
"""

from typing import Any, Dict, Optional

import torch


class DynamicLRScheduler:
    """Dynamic learning rate scheduler with plateau detection and recovery."""

    def __init__(
        self,
        optimizer: Optional["torch.optim.Optimizer"],
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.counter = 0
        self.last_lr = self._get_lr()

    def _get_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def _set_lr(self, lr: float) -> None:
        """Set learning rate for all parameter groups."""
        if self.optimizer is None:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, current_loss: float) -> Dict[str, Any]:
        """Update learning rate based on current loss."""
        info: Dict[str, Any] = {"lr_changed": False, "lr": self._get_lr(), "action": "none"}

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            info["action"] = "improvement"
        else:
            self.counter += 1
            info["action"] = f"plateau_{self.counter}"

        if self.counter >= self.patience:
            new_lr = max(self._get_lr() * self.factor, self.min_lr)
            if new_lr < self._get_lr():
                self._set_lr(new_lr)
                self.counter = 0
                info["lr_changed"] = True
                info["lr"] = new_lr
                info["action"] = "lr_decay"
            else:
                info["action"] = "lr_floor_reached"

        return info
