#!/usr/bin/env python3
"""
Supervised Learning Callbacks.

Callbacks optimized for supervised learning tasks (classification/regression)
with shared monitored-callback and metrics-callback abstractions.
"""

from __future__ import annotations

import abc
import copy
import logging
from datetime import datetime
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    NoOpMemoryOptimizedCallback,
)
from ztb.training.callbacks.shared.utils.value_utils import (
    append_bounded as _append_bounded_value,
    as_optional_float as _as_float,
)
from ztb.types.common import ObjectMap

_HISTORY_LIMIT = 10_000

def _append_bounded(
    history: list[float], value: float, max_len: int = _HISTORY_LIMIT
) -> None:
    _append_bounded_value(history, value, max_len)

class _MonitoredCallback(NoOpMemoryOptimizedCallback):
    """Shared monitor/mode comparison logic for supervised callbacks."""

    def __init__(self, monitor: str = "val_loss", mode: str = "auto") -> None:
        super().__init__()
        self.monitor = monitor

        normalized_mode = mode.lower()
        if normalized_mode == "auto":
            normalized_mode = "min" if "loss" in monitor else "max"
        if normalized_mode not in {"min", "max"}:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = normalized_mode
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def _reset_best_value(self) -> None:
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def _is_improved(self, current_value: float, min_delta: float = 0.0) -> bool:
        if self.mode == "min":
            return current_value < (self.best_value - min_delta)
        return current_value > (self.best_value + min_delta)

class EarlyStoppingCallback(_MonitoredCallback):
    """
    Early stopping callback for supervised learning.

    Monitors validation metrics and marks training to stop when
    performance stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "auto",
        restore_best_weights: bool = False,
    ):
        super().__init__(monitor=monitor, mode=mode)
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_weights = restore_best_weights

        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights: object | None = None
        self.best_epoch = 0
        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self._reset_best_value()
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        self.logger.info(
            "Early stopping initialized: monitor=%s, patience=%s",
            self.monitor,
            self.patience,
        )

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if logs is None:
            return

        metric_value = _as_float(logs.get(self.monitor))
        if metric_value is None:
            return

        if self._is_improved(metric_value, self.min_delta):
            self.best_value = metric_value
            self.wait_count = 0
            self.best_epoch = context.epoch

            if self.restore_best_weights and "weights" in logs:
                try:
                    self.best_weights = copy.deepcopy(logs["weights"])
                except Exception:
                    self.best_weights = logs["weights"]

            self.logger.debug(
                "New best %s=%.6f at epoch %s",
                self.monitor,
                self.best_value,
                context.epoch,
            )
            return

        self.wait_count += 1
        if self.wait_count >= self.patience and self.stopped_epoch == 0:
            self.stopped_epoch = context.epoch
            self.logger.info(
                "Early stopping triggered at epoch %s (best %s=%.6f at epoch %s)",
                context.epoch,
                self.monitor,
                self.best_value,
                self.best_epoch,
            )

    def on_training_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if self.restore_best_weights and self.best_weights is not None:
            self.logger.info("Restoring best weights from epoch %s", self.best_epoch)

    def should_stop_training(self) -> bool:
        return self.stopped_epoch > 0

    def get_early_stopping_stats(self) -> ObjectMap:
        return {
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
            "should_stop": self.should_stop_training(),
        }

class LearningRateSchedulerCallback(NoOpMemoryOptimizedCallback):
    """Learning rate scheduler callback with multiple scheduling strategies."""

    def __init__(self, schedule_type: str = "step", **schedule_params):
        super().__init__()
        self.schedule_type = schedule_type
        self.schedule_params: ObjectMap = dict(schedule_params)

        self._schedulers: dict[str, Callable[[int, ObjectMap], float]] = {
            "step": self._step_decay,
            "exponential": self._exponential_decay,
            "cosine": self._cosine_annealing,
            "plateau": self._plateau_decay,
        }
        if self.schedule_type not in self._schedulers:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        self.scheduler = self._schedulers[self.schedule_type]
        self.initial_lr = self._get_float_param("initial_lr", 0.001)
        self.current_lr = self.initial_lr
        self.lr_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.logger = logging.getLogger(__name__)

    def _get_float_param(self, key: str, default: float) -> float:
        value = _as_float(self.schedule_params.get(key, default))
        return default if value is None else value

    def _get_int_param(self, key: str, default: int) -> int:
        raw = self.schedule_params.get(key, default)
        if isinstance(raw, bool):
            return default
        if isinstance(raw, int):
            return raw
        value = _as_float(raw)
        return default if value is None else int(value)

    def on_training_start(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.current_lr = self.initial_lr
        self.lr_history = [self.current_lr]
        self.val_loss_history = []
        self.logger.info(
            "Learning rate scheduler initialized: type=%s, initial_lr=%.6f",
            self.schedule_type,
            self.initial_lr,
        )

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        log_data = logs or {}
        new_lr = self.scheduler(context.epoch, log_data)
        if not np.isfinite(new_lr) or new_lr <= 0.0:
            self.logger.warning(
                "Ignoring invalid learning rate %.6f at epoch %s", new_lr, context.epoch
            )
            return

        if not np.isclose(new_lr, self.current_lr):
            self.current_lr = float(new_lr)
            _append_bounded(self.lr_history, self.current_lr)
            self.logger.debug("Learning rate updated to %.6f", self.current_lr)

        if logs is not None:
            logs["learning_rate"] = self.current_lr

    def _step_decay(self, epoch: int, logs: ObjectMap) -> float:
        step_size = max(1, self._get_int_param("step_size", 10))
        gamma = self._get_float_param("gamma", 0.1)
        return float(self.initial_lr * (gamma ** (epoch // step_size)))

    def _exponential_decay(self, epoch: int, logs: ObjectMap) -> float:
        decay_rate = self._get_float_param("decay_rate", 0.95)
        decay_steps = max(1, self._get_int_param("decay_steps", 1))
        return float(self.initial_lr * (decay_rate ** (epoch // decay_steps)))

    def _cosine_annealing(self, epoch: int, logs: ObjectMap) -> float:
        max_epochs = max(1, self._get_int_param("max_epochs", 100))
        min_lr = self._get_float_param("min_lr", 1e-6)

        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * epoch / max_epochs))
        return float(min_lr + (self.initial_lr - min_lr) * cosine_decay)

    def _plateau_decay(self, epoch: int, logs: ObjectMap) -> float:
        patience = max(1, self._get_int_param("patience", 5))
        factor = self._get_float_param("factor", 0.5)
        min_lr = self._get_float_param("min_lr", 1e-6)
        plateau_epsilon = self._get_float_param("plateau_epsilon", 1e-4)

        current_val_loss = _as_float(logs.get("val_loss"))
        if current_val_loss is not None:
            _append_bounded(self.val_loss_history, current_val_loss)

        if len(self.val_loss_history) < patience:
            return self.current_lr

        recent_losses = self.val_loss_history[-patience:]
        if max(recent_losses) - min(recent_losses) <= plateau_epsilon:
            return float(max(self.current_lr * factor, min_lr))

        return self.current_lr

    def get_current_lr(self) -> float:
        return self.current_lr

    def get_lr_schedule_info(self) -> ObjectMap:
        return {
            "schedule_type": self.schedule_type,
            "initial_lr": self.initial_lr,
            "current_lr": self.current_lr,
            "schedule_params": self.schedule_params,
            "history_length": len(self.lr_history),
        }

class ModelCheckpointCallback(_MonitoredCallback):
    """Checkpoint callback with monitor-driven best-model saving."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = "auto",
        period: int = 1,
    ):
        super().__init__(monitor=monitor, mode=mode)
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = max(1, period)
        self.best_filepath: str | None = None
        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if logs is None or context.epoch % self.period != 0:
            return

        metric_value = _as_float(logs.get(self.monitor))

        if self.save_best_only:
            if metric_value is None:
                return
            if not self._is_improved(metric_value):
                return
            self.best_value = metric_value

        self._save_checkpoint(context, logs, metric_value)

    def _save_checkpoint(
        self,
        context: LearningContext,
        logs: ObjectMap,
        metric_value: float | None,
    ) -> None:
        try:
            format_vars: ObjectMap = dict(logs)
            format_vars.setdefault(self.monitor, metric_value if metric_value is not None else 0.0)
            try:
                filename = self.filepath.format(epoch=context.epoch, **format_vars)
            except Exception:
                filename = f"{self.filepath}_epoch_{context.epoch}"

            # Keep cached metadata compact to avoid storing large tensors/arrays.
            filtered_metrics: ObjectMap = {
                k: v
                for k, v in logs.items()
                if k not in {"predictions", "targets", "weights", "model"}
            }

            checkpoint_metadata: ObjectMap = {
                "epoch": context.epoch,
                "model_config": context.model_config,
                "metrics": filtered_metrics,
                "timestamp": datetime.now().isoformat(),
                "save_weights_only": self.save_weights_only,
            }
            self.cache_metrics(f"checkpoint_epoch_{context.epoch}", checkpoint_metadata)

            self.best_filepath = filename
            self.logger.info("Saving checkpoint to: %s", filename)
        except Exception as exc:
            self.logger.error("Failed to save checkpoint: %s", exc)

    def get_checkpoint_info(self) -> ObjectMap:
        return {
            "filepath": self.filepath,
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_filepath": self.best_filepath,
            "save_best_only": self.save_best_only,
            "save_weights_only": self.save_weights_only,
            "period": self.period,
        }

class _BaseSupervisedMetricsCallback(NoOpMemoryOptimizedCallback, abc.ABC):
    """Shared extraction and scheduling logic for supervised metrics callbacks."""

    def __init__(self, compute_frequency: int = 1):
        super().__init__(cache_size=1000)
        self.compute_frequency = max(1, compute_frequency)

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        payload = self._extract_predictions_targets(logs)
        if payload is None:
            return

        predictions, targets = payload
        try:
            self._compute_metrics(context, predictions, targets)
        except Exception as exc:
            self.logger.error("Failed to compute metrics: %s", exc)

    def _extract_predictions_targets(
        self, logs: ObjectMap
    ) -> tuple[np.ndarray, np.ndarray] | None:
        predictions_obj = logs.get("predictions")
        targets_obj = logs.get("targets")
        if predictions_obj is None or targets_obj is None:
            return None

        predictions = np.asarray(predictions_obj)
        targets = np.asarray(targets_obj)
        if predictions.size == 0 or targets.size == 0:
            return None
        return predictions, targets

    @abc.abstractmethod
    def _compute_metrics(
        self,
        context: LearningContext,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Compute and store callback-specific metrics."""

class ClassificationMetricsCallback(_BaseSupervisedMetricsCallback):
    """Compute and track classification metrics during training."""

    def __init__(self, compute_frequency: int = 1):
        super().__init__(compute_frequency=compute_frequency)
        self.accuracy_history: list[float] = []
        self.precision_history: list[float] = []
        self.recall_history: list[float] = []
        self.f1_history: list[float] = []
        self.logger = logging.getLogger(__name__)

    def _compute_metrics(
        self,
        context: LearningContext,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        accuracy = float(accuracy_score(targets, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        )

        precision_f = float(precision)
        recall_f = float(recall)
        f1_f = float(f1)

        _append_bounded(self.accuracy_history, accuracy)
        _append_bounded(self.precision_history, precision_f)
        _append_bounded(self.recall_history, recall_f)
        _append_bounded(self.f1_history, f1_f)

        self.cache_metrics(
            f"classification_epoch_{context.epoch}",
            {
                "accuracy": accuracy,
                "precision": precision_f,
                "recall": recall_f,
                "f1": f1_f,
                "epoch": context.epoch,
            },
        )

        self.logger.debug(
            "Classification metrics at epoch %s: acc=%.4f, f1=%.4f",
            context.epoch,
            accuracy,
            f1_f,
        )

    def get_classification_stats(self) -> ObjectMap:
        stats: ObjectMap = {"epochs_computed": len(self.accuracy_history)}
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

class RegressionMetricsCallback(_BaseSupervisedMetricsCallback):
    """Compute and track regression metrics during training."""

    def __init__(self, compute_frequency: int = 1):
        super().__init__(compute_frequency=compute_frequency)
        self.mse_history: list[float] = []
        self.rmse_history: list[float] = []
        self.mae_history: list[float] = []
        self.r2_history: list[float] = []
        self.logger = logging.getLogger(__name__)

    def _compute_metrics(
        self,
        context: LearningContext,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        pred_flat = predictions.reshape(-1)
        targ_flat = targets.reshape(-1)

        mse = float(mean_squared_error(targ_flat, pred_flat))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(pred_flat - targ_flat)))
        r2 = float(r2_score(targ_flat, pred_flat))

        _append_bounded(self.mse_history, mse)
        _append_bounded(self.rmse_history, rmse)
        _append_bounded(self.mae_history, mae)
        _append_bounded(self.r2_history, r2)

        self.cache_metrics(
            f"regression_epoch_{context.epoch}",
            {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "epoch": context.epoch,
            },
        )

        self.logger.debug(
            "Regression metrics at epoch %s: mse=%.4f, r2=%.4f",
            context.epoch,
            mse,
            r2,
        )

    def get_regression_stats(self) -> ObjectMap:
        stats: ObjectMap = {"epochs_computed": len(self.mse_history)}
        if self.mse_history:
            stats.update(
                {
                    "mse_mean": float(np.mean(self.mse_history)),
                    "rmse_mean": float(np.mean(self.rmse_history)),
                    "mae_mean": float(np.mean(self.mae_history)),
                    "r2_mean": float(np.mean(self.r2_history)),
                    "r2_latest": self.r2_history[-1],
                }
            )
        return stats

# Factory functions for easy instantiation

def create_early_stopping(**kwargs) -> EarlyStoppingCallback:
    """Create early stopping callback with default settings."""
    defaults: ObjectMap = {
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True,
    }
    defaults.update(kwargs)
    return EarlyStoppingCallback(**defaults)

def create_learning_rate_scheduler(
    schedule_type: str = "step", **kwargs
) -> LearningRateSchedulerCallback:
    """Create learning rate scheduler with default settings."""
    defaults: ObjectMap = {"initial_lr": 0.001}
    if schedule_type == "step":
        defaults.update({"step_size": 10, "gamma": 0.1})
    elif schedule_type == "cosine":
        defaults.update({"max_epochs": 100, "min_lr": 1e-6})

    defaults.update(kwargs)
    return LearningRateSchedulerCallback(schedule_type, **defaults)

def create_model_checkpoint(**kwargs) -> ModelCheckpointCallback:
    """Create model checkpoint callback with default settings."""
    defaults: ObjectMap = {
        "filepath": "checkpoint_epoch_{epoch:02d}_{val_loss:.2f}.h5",
        "monitor": "val_loss",
        "save_best_only": True,
    }
    defaults.update(kwargs)
    return ModelCheckpointCallback(**defaults)

def create_classification_metrics(**kwargs) -> ClassificationMetricsCallback:
    """Create classification metrics callback with default settings."""
    defaults: ObjectMap = {"compute_frequency": 1}
    defaults.update(kwargs)
    return ClassificationMetricsCallback(**defaults)

def create_regression_metrics(**kwargs) -> RegressionMetricsCallback:
    """Create regression metrics callback with default settings."""
    defaults: ObjectMap = {"compute_frequency": 1}
    defaults.update(kwargs)
    return RegressionMetricsCallback(**defaults)

__all__ = [
    "EarlyStoppingCallback",
    "LearningRateSchedulerCallback",
    "ModelCheckpointCallback",
    "ClassificationMetricsCallback",
    "RegressionMetricsCallback",
    "create_early_stopping",
    "create_learning_rate_scheduler",
    "create_model_checkpoint",
    "create_classification_metrics",
    "create_regression_metrics",
]
