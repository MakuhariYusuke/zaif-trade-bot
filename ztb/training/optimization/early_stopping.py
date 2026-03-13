#!/usr/bin/env python3
"""
Early stopping implementation for training optimization.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting during training.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights
            monitor: Metric to monitor
            mode: 'min' or 'max' - whether to minimize or maximize the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode

        self.best_score: float | None = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

        if mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'")

        self.monitor_op = np.less if mode == "min" else np.greater
        self.min_delta *= 1 if mode == "min" else -1

    def __call__(self, score: float, model: Any = None) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value
            model: Model to save weights from (optional)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = (
                    model.get_weights() if hasattr(model, "get_weights") else None
                )
            return False

        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = (
                    model.get_weights() if hasattr(model, "get_weights") else None
                )
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = self.counter
            return True

        return False

    def restore_weights(self, model: Any) -> None:
        """Restore best weights to model."""
        if (
            self.restore_best_weights
            and self.best_weights is not None
            and model is not None
        ):
            if hasattr(model, "set_weights"):
                model.set_weights(self.best_weights)
            logger.info(
                f"Restored model weights from epoch with best {self.monitor}: {self.best_score}"
            )

    @property
    def should_stop(self) -> bool:
        """Check if early stopping was triggered."""
        return self.counter >= self.patience
