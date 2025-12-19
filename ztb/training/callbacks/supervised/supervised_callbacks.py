#!/usr/bin/env python3
"""
Supervised Learning Callbacks.

This module provides callbacks optimized for supervised learning
tasks including classification and regression.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    MemoryOptimizedCallback,
)


class EarlyStoppingCallback(MemoryOptimizedCallback):
    """
    Early stopping callback for supervised learning.

    Monitors validation metrics and stops training when performance
    stops improving to prevent overfitting.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "auto",
        restore_best_weights: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_weights = restore_best_weights

        if mode == "auto":
            mode = "min" if "loss" in monitor else "max"
        self.mode = mode

        # State tracking
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize early stopping state."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        self.logger.info(
            f"Early stopping initialized: monitor={self.monitor}, patience={self.patience}"
        )

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Check if training should stop."""
        if logs is None:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        # Check if current value is better
        if self.mode == "min":
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)

        if is_better:
            self.best_value = current_value
            self.wait_count = 0
            self.best_epoch = context.epoch

            # Save best weights if requested
            if self.restore_best_weights and "weights" in logs:
                self.best_weights = logs["weights"].copy()

            self.logger.debug(
                f"New best {self.monitor}: {self.best_value:.4f} at epoch {context.epoch}"
            )
        else:
            self.wait_count += 1

        # Check if patience exceeded
        if self.wait_count >= self.patience:
            self.stopped_epoch = context.epoch
            self.logger.info(
                f"Early stopping triggered at epoch {context.epoch}. "
                f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}"
            )

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Restore best weights if requested."""
        if self.restore_best_weights and self.best_weights is not None:
            # In a real implementation, this would restore model weights
            self.logger.info(f"Restoring best weights from epoch {self.best_epoch}")


    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass
        pass

    def should_stop_training(self) -> bool:
        """Check if training should be stopped."""
        return self.stopped_epoch > 0

            "best_epoch": self.best_epoch,
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
            "should_stop": self.should_stop_training(),
        }


class LearningRateSchedulerCallback(MemoryOptimizedCallback):
    """
    Learning rate scheduler for supervised learning.

    Provides various learning rate scheduling strategies including
    step decay, exponential decay, cosine annealing, and plateau detection.
    """

    def __init__(self, schedule_type: str = "step", **schedule_params):
        super().__init__()
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params

        # Initialize scheduler based on type
        if schedule_type == "step":
            self.scheduler = self._step_decay
        elif schedule_type == "exponential":
            self.scheduler = self._exponential_decay
        elif schedule_type == "cosine":
            self.scheduler = self._cosine_annealing
        elif schedule_type == "plateau":
            self.scheduler = self._plateau_decay
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # State tracking
        self.initial_lr = schedule_params.get("initial_lr", 0.001)
        self.current_lr = self.initial_lr
        self.lr_history: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize learning rate scheduling."""
        self.current_lr = self.initial_lr
        self.lr_history = [self.current_lr]
        self.logger.info(
            f"Learning rate scheduler initialized: {self.schedule_type}, initial_lr={self.initial_lr}"
        )

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update learning rate according to schedule."""
        new_lr = self.scheduler(context.epoch, logs or {})
        if new_lr != self.current_lr:
            self.current_lr = new_lr
            self.lr_history.append(self.current_lr)
            self.logger.debug(f"Learning rate updated to: {self.current_lr:.6f}")

        # Always set current learning rate in logs
        if logs is not None:
            logs["learning_rate"] = self.current_lr

    def _step_decay(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Step decay schedule."""
        step_size = self.schedule_params.get("step_size", 10)
        gamma = self.schedule_params.get("gamma", 0.1)
        return self.initial_lr * (gamma ** (epoch // step_size))

    def _exponential_decay(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Exponential decay schedule."""
        decay_rate = self.schedule_params.get("decay_rate", 0.95)
        decay_steps = self.schedule_params.get("decay_steps", 1)
        return self.initial_lr * (decay_rate ** (epoch // decay_steps))

    def _cosine_annealing(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Cosine annealing schedule."""
        max_epochs = self.schedule_params.get("max_epochs", 100)
        min_lr = self.schedule_params.get("min_lr", 1e-6)

        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
        return min_lr + (self.initial_lr - min_lr) * cosine_decay

    def _plateau_decay(self, epoch: int, logs: Dict[str, Any]) -> float:
        """Plateau-based decay."""
        patience = self.schedule_params.get("patience", 5)
        factor = self.schedule_params.get("factor", 0.5)
        min_lr = self.schedule_params.get("min_lr", 1e-6)

        # Simple plateau detection based on recent validation loss
        if "val_loss" in logs and len(self.lr_history) > patience:
            recent_losses = [logs.get("val_loss", float("inf"))]  # Would need history
            if len(recent_losses) >= patience:
                # Check if loss has plateaued
                loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                if abs(loss_trend) < 0.001:  # Very small trend
                    new_lr = max(self.current_lr * factor, min_lr)
                    return new_lr

        return self.current_lr

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def get_lr_schedule_info(self) -> Dict[str, Any]:
        """Get learning rate schedule information."""
        return {
            "schedule_type": self.schedule_type,
            "initial_lr": self.initial_lr,
            "current_lr": self.current_lr,
            "schedule_params": self.schedule_params,
            "history_length": len(self.lr_history),
        }


    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        """Called at the end of each batch."""
        pass


class ModelCheckpointCallback(MemoryOptimizedCallback):
    """
    Model checkpoint callback for supervised learning.

    Saves model checkpoints based on performance metrics,
    with options for best model saving and periodic checkpoints.
    """
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = "auto",
        period: int = 1,
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        # State tracking
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_filepath = None

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint if conditions are met."""
        if logs is None:
            return

        # Check if this epoch should be saved
        if context.epoch % self.period != 0:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        should_save = False
        if self.save_best_only:
            # Check if current value is better
            if self.mode == "min":
                is_better = current_value < self.best_value
            else:
                is_better = current_value > self.best_value

            if is_better:
                self.best_value = current_value
                should_save = True
        else:
            should_save = True

        if should_save:
            self._save_checkpoint(context, logs)

    def _save_checkpoint(self, context: LearningContext, logs: Dict[str, Any]) -> None:
        """Save model checkpoint."""
        try:
            # Create filename with epoch and metric
            metric_value = logs.get(self.monitor, 0)
            filename = self.filepath.format(
                epoch=context.epoch, **{self.monitor: metric_value}
            )

            # In a real implementation, this would save the actual model
            {
                "epoch": context.epoch,
                "model_config": context.model_config,
                "metrics": logs,
                "timestamp": datetime.now().isoformat(),
            }

            # Simulate saving
            self.logger.info(f"Saving checkpoint to: {filename}")
            self.best_filepath = filename

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get checkpoint information."""
        return {
            "filepath": self.filepath,
            "monitor": self.monitor,
            "best_value": self.best_value,
            "best_filepath": self.best_filepath,
            "save_best_only": self.save_best_only,
            "period": self.period,
        }


    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of each batch."""
        pass


class ClassificationMetricsCallback(MemoryOptimizedCallback):
    """
    Classification metrics callback.

    Computes and tracks classification metrics including accuracy,
    precision, recall, and F1-score during training.
        self.compute_frequency = compute_frequency

        # Metrics history
        self.accuracy_history: List[float] = []
        self.precision_history: List[float] = []
        self.recall_history: List[float] = []
        self.f1_history: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Compute classification metrics."""
        if context.epoch % self.compute_frequency != 0:
            return
        targets = logs["targets"]

        # Convert predictions to class labels if needed
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        try:
            # Compute metrics
            accuracy = accuracy_score(targets, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions, average="weighted", zero_division=0
            )

            # Store in history
            self.accuracy_history.append(accuracy)
            self.precision_history.append(precision)
            self.cache_metrics(
                metrics_key,
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "epoch": context.epoch,
                },
            )

            self.logger.debug(
                f"Classification metrics - Acc: {accuracy:.4f}, "
                f"F1: {f1:.4f} at epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to compute classification metrics: {e}")

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification metrics statistics."""
        stats = {"epochs_computed": len(self.accuracy_history)}

        if self.accuracy_history:
            stats.update(
                {
                    "accuracy_mean": float(np.mean(self.accuracy_history)),
                    "accuracy_std": float(np.std(self.accuracy_history)),
                    "accuracy_latest": self.accuracy_history[-1],
                    "precision_mean": float(np.mean(self.precision_history)),
                    "recall_mean": float(np.mean(self.recall_history)),
                    "f1_mean": float(np.mean(self.f1_history)),
                }
            )

        return stats

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of training."""
        pass
        pass

    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
class RegressionMetricsCallback(MemoryOptimizedCallback):
    """
    Regression metrics callback.

    Computes and tracks regression metrics including MSE, RMSE,
    MAE, and R² score during training.
    """

    def __init__(self, compute_frequency: int = 1):
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency

        # Metrics history
        self.mse_history: List[float] = []
        self.rmse_history: List[float] = []
        self.mae_history: List[float] = []
        self.r2_history: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(

        if logs is None or "predictions" not in logs or "targets" not in logs:
            return

        predictions = logs["predictions"]
        targets = logs["targets"]

        try:
            # Compute metrics
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            r2 = r2_score(targets, predictions)

            # Store in history
            self.mse_history.append(mse)
            self.rmse_history.append(rmse)
            self.mae_history.append(mae)
            self.r2_history.append(r2)

            # Cache metrics
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "epoch": context.epoch,
                },
            )

            self.logger.debug(
                f"Regression metrics - MSE: {mse:.4f}, "
                f"R²: {r2:.4f} at epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to compute regression metrics: {e}")

    def get_regression_stats(self) -> Dict[str, Any]:
        """Get regression metrics statistics."""
        stats = {"epochs_computed": len(self.mse_history)}

        if self.mse_history:
            stats.update(
                    "mae_mean": float(np.mean(self.mae_history)),
                    "r2_mean": float(np.mean(self.r2_history)),
                    "r2_latest": self.r2_history[-1],
                }
            )

        return stats

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of training."""
        pass

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of training."""
        """Called at the start of each epoch."""
        pass

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of each batch."""
        pass


# Factory functions for easy instantiation
def create_early_stopping(**kwargs) -> EarlyStoppingCallback:
    """Create early stopping callback with default settings."""
    defaults = {"monitor": "val_loss", "patience": 10, "restore_best_weights": True}
    defaults.update(kwargs)
) -> LearningRateSchedulerCallback:
    """Create learning rate scheduler with default settings."""
    defaults = {"initial_lr": 0.001}
    if schedule_type == "step":
        defaults.update({"step_size": 10, "gamma": 0.1})
    elif schedule_type == "cosine":
        defaults.update({"max_epochs": 100, "min_lr": 1e-6})

    defaults.update(kwargs)
    return LearningRateSchedulerCallback(schedule_type, **defaults)


def create_model_checkpoint(**kwargs) -> ModelCheckpointCallback:
    """Create model checkpoint callback with default settings."""
    defaults = {
        "filepath": "checkpoint_epoch_{epoch:02d}_{val_loss:.2f}.h5",
        "monitor": "val_loss",
        "save_best_only": True,
    }
    defaults.update(kwargs)
    return ModelCheckpointCallback(**defaults)


def create_classification_metrics(**kwargs) -> ClassificationMetricsCallback:
    """Create classification metrics callback with default settings."""
    defaults = {"compute_frequency": 1}
    """Create regression metrics callback with default settings."""
    defaults = {"compute_frequency": 1}
    defaults.update(kwargs)
    return RegressionMetricsCallback(**defaults)
