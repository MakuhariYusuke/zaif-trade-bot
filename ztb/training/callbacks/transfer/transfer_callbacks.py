#!/usr/bin/env python3
"""
Transfer Learning Callbacks.

Callbacks for transfer-learning workflows including domain adaptation,
fine-tuning stability, and source/target performance diagnostics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    NoOpMemoryOptimizedCallback,
)
from ztb.types.common import ObjectMap


_HISTORY_LIMIT = 1_000


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _append_bounded(history: list[float], value: float, max_len: int = _HISTORY_LIMIT) -> None:
    history.append(value)
    overflow = len(history) - max_len
    if overflow > 0:
        del history[:overflow]


def _as_array(value: object) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _update_float_history_map(
    history_map: dict[str, list[float]],
    payload: object,
    max_len: int = _HISTORY_LIMIT,
) -> None:
    if not isinstance(payload, dict):
        return

    for key, raw in payload.items():
        if not isinstance(key, str):
            continue
        value = _as_float(raw)
        if value is None:
            continue
        history = history_map.setdefault(key, [])
        _append_bounded(history, value, max_len=max_len)


class DomainAdaptationCallback(NoOpMemoryOptimizedCallback):
    """Monitor domain-adaptation progress and source-target feature shift."""

    def __init__(
        self,
        compute_frequency: int = 1,
        adaptation_method: str = "auto",
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = max(1, compute_frequency)
        self.adaptation_method = adaptation_method

        self.source_domain_stats: dict[str, list[float]] = {}
        self.target_domain_stats: dict[str, list[float]] = {}
        self.domain_shift_scores: dict[str, list[float]] = {}

        self.feature_alignment_scores: list[float] = []
        self.classifier_alignment_scores: list[float] = []
        self.adaptation_losses: list[float] = []
        self.discriminator_accuracies: list[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[ObjectMap] = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        try:
            source_features = _as_array(logs.get("source_features"))
            target_features = _as_array(logs.get("target_features"))
            if (
                source_features is not None
                and target_features is not None
                and source_features.ndim >= 2
                and target_features.ndim >= 2
                and source_features.shape[1] == target_features.shape[1]
            ):
                self._track_domain_statistics(source_features, target_features)

            feature_alignment_loss = _as_float(logs.get("feature_alignment_loss"))
            if feature_alignment_loss is not None:
                _append_bounded(
                    self.feature_alignment_scores,
                    1.0 / (1.0 + feature_alignment_loss),
                )

            classifier_alignment_loss = _as_float(logs.get("classifier_alignment_loss"))
            if classifier_alignment_loss is not None:
                _append_bounded(
                    self.classifier_alignment_scores,
                    1.0 / (1.0 + classifier_alignment_loss),
                )

            adaptation_loss = _as_float(logs.get("domain_adaptation_loss"))
            if adaptation_loss is not None:
                _append_bounded(self.adaptation_losses, adaptation_loss)

            discriminator_acc = _as_float(logs.get("discriminator_accuracy"))
            if discriminator_acc is not None:
                _append_bounded(self.discriminator_accuracies, discriminator_acc)

            metrics: ObjectMap = {
                "epoch": context.epoch,
                "adaptation_method": self.adaptation_method,
            }

            if self.feature_alignment_scores:
                metrics["feature_alignment_score"] = self.feature_alignment_scores[-1]
            if self.classifier_alignment_scores:
                metrics["classifier_alignment_score"] = self.classifier_alignment_scores[-1]
            if self.adaptation_losses:
                metrics["adaptation_loss"] = self.adaptation_losses[-1]
            if self.discriminator_accuracies:
                metrics["discriminator_accuracy"] = self.discriminator_accuracies[-1]

            metrics.update(self._get_domain_shift_summary())
            self.cache_metrics(f"domain_adaptation_epoch_{context.epoch}", metrics)

        except Exception as exc:
            self.logger.error("Failed to monitor domain adaptation: %s", exc)

    def _track_domain_statistics(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> None:
        try:
            source_mean = np.mean(source_features, axis=0)
            source_std = np.std(source_features, axis=0)
            target_mean = np.mean(target_features, axis=0)
            target_std = np.std(target_features, axis=0)

            for i in range(len(source_mean)):
                mean_key = f"feature_{i}_mean"
                std_key = f"feature_{i}_std"

                _append_bounded(
                    self.source_domain_stats.setdefault(mean_key, []),
                    float(source_mean[i]),
                )
                _append_bounded(
                    self.target_domain_stats.setdefault(mean_key, []),
                    float(target_mean[i]),
                )
                _append_bounded(
                    self.source_domain_stats.setdefault(std_key, []),
                    float(source_std[i]),
                )
                _append_bounded(
                    self.target_domain_stats.setdefault(std_key, []),
                    float(target_std[i]),
                )

                _append_bounded(
                    self.domain_shift_scores.setdefault(mean_key, []),
                    float(abs(source_mean[i] - target_mean[i])),
                )
                _append_bounded(
                    self.domain_shift_scores.setdefault(std_key, []),
                    float(abs(source_std[i] - target_std[i])),
                )
        except Exception as exc:
            self.logger.warning("Failed to track domain statistics: %s", exc)

    def _get_domain_shift_summary(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        all_shifts: list[float] = []

        for shifts in self.domain_shift_scores.values():
            all_shifts.extend(shifts)

        if all_shifts:
            summary["avg_domain_shift"] = float(np.mean(all_shifts))
            summary["max_domain_shift"] = float(np.max(all_shifts))
            summary["domain_shift_std"] = float(np.std(all_shifts))

        return summary

    def get_domain_adaptation_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "adaptation_method": self.adaptation_method,
            "epochs_monitored": len(self.feature_alignment_scores),
            "domain_features_tracked": len(self.source_domain_stats),
        }

        if self.feature_alignment_scores:
            stats.update(
                {
                    "feature_alignment_mean": float(
                        np.mean(self.feature_alignment_scores)
                    ),
                    "feature_alignment_latest": self.feature_alignment_scores[-1],
                }
            )

        if self.classifier_alignment_scores:
            stats.update(
                {
                    "classifier_alignment_mean": float(
                        np.mean(self.classifier_alignment_scores)
                    ),
                    "classifier_alignment_latest": self.classifier_alignment_scores[-1],
                }
            )

        if self.adaptation_losses:
            stats.update(
                {
                    "adaptation_loss_mean": float(np.mean(self.adaptation_losses)),
                    "adaptation_loss_latest": self.adaptation_losses[-1],
                }
            )

        if self.discriminator_accuracies:
            stats.update(
                {
                    "discriminator_accuracy_mean": float(
                        np.mean(self.discriminator_accuracies)
                    ),
                    "discriminator_accuracy_latest": self.discriminator_accuracies[-1],
                }
            )

        stats.update(self._get_domain_shift_summary())
        return stats


class FineTuningCallback(NoOpMemoryOptimizedCallback):
    """Monitor fine-tuning dynamics and catastrophic forgetting."""

    def __init__(
        self,
        compute_frequency: int = 1,
        freeze_layers: Optional[list[str]] = None,
        monitor_catastrophic_forgetting: bool = True,
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = max(1, compute_frequency)
        self.freeze_layers = freeze_layers or []
        self.monitor_catastrophic_forgetting = monitor_catastrophic_forgetting

        self.layer_learning_rates: dict[str, list[float]] = {}
        self.layer_gradients: dict[str, list[float]] = {}
        self.task_performance: dict[str, list[float]] = {}

        self.source_task_performance: dict[str, list[float]] = {}
        self.forgetting_scores: dict[str, list[float]] = {}
        self.layer_weights_change: dict[str, list[float]] = {}

        self._prev_weights: dict[str, np.ndarray] = {}
        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[ObjectMap] = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        try:
            _update_float_history_map(self.layer_learning_rates, logs.get("layer_learning_rates"))
            _update_float_history_map(self.layer_gradients, logs.get("layer_gradients"))
            _update_float_history_map(self.task_performance, logs.get("task_performance"))

            if self.monitor_catastrophic_forgetting:
                self._monitor_catastrophic_forgetting(logs)

            self._track_weight_changes(logs.get("layer_weights"))

            metrics: ObjectMap = {
                "epoch": context.epoch,
                "frozen_layers": self.freeze_layers,
                "monitor_catastrophic_forgetting": self.monitor_catastrophic_forgetting,
            }

            for task, perf in self.task_performance.items():
                if perf:
                    metrics[f"{task}_performance"] = perf[-1]
            for task, scores in self.forgetting_scores.items():
                if scores:
                    metrics[f"{task}_forgetting"] = scores[-1]

            self.cache_metrics(f"fine_tuning_epoch_{context.epoch}", metrics)
        except Exception as exc:
            self.logger.error("Failed to monitor fine-tuning: %s", exc)

    def _monitor_catastrophic_forgetting(self, logs: ObjectMap) -> None:
        payload = logs.get("source_task_performance")
        if not isinstance(payload, dict):
            return

        for task_name, performance in payload.items():
            if not isinstance(task_name, str):
                continue
            current_perf = _as_float(performance)
            if current_perf is None:
                continue

            history = self.source_task_performance.setdefault(task_name, [])
            _append_bounded(history, current_perf)

            if len(history) >= 2:
                forgetting = history[0] - current_perf
                _append_bounded(self.forgetting_scores.setdefault(task_name, []), forgetting)

    def _track_weight_changes(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        for layer_name, weights_obj in payload.items():
            if not isinstance(layer_name, str):
                continue

            weights = _as_array(weights_obj)
            if weights is None:
                continue

            history = self.layer_weights_change.setdefault(layer_name, [])
            prev = self._prev_weights.get(layer_name)
            if prev is None or prev.shape != weights.shape:
                _append_bounded(history, 0.0)
            else:
                _append_bounded(history, float(np.mean(np.abs(weights - prev))))

            self._prev_weights[layer_name] = weights.copy()

    def get_fine_tuning_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "frozen_layers": self.freeze_layers,
            "monitored_layers": len(self.layer_learning_rates),
            "monitored_tasks": len(self.task_performance),
            "catastrophic_forgetting_monitored": self.monitor_catastrophic_forgetting,
        }

        for layer, lrs in self.layer_learning_rates.items():
            if lrs:
                stats[f"{layer}_lr_mean"] = float(np.mean(lrs))
                stats[f"{layer}_lr_latest"] = lrs[-1]

        for task, perf in self.task_performance.items():
            if perf:
                stats[f"{task}_perf_mean"] = float(np.mean(perf))
                stats[f"{task}_perf_latest"] = perf[-1]
                stats[f"{task}_perf_improvement"] = perf[-1] - perf[0] if len(perf) > 1 else 0.0

        for task, scores in self.forgetting_scores.items():
            if scores:
                stats[f"{task}_forgetting_mean"] = float(np.mean(scores))
                stats[f"{task}_forgetting_latest"] = scores[-1]
                stats[f"{task}_max_forgetting"] = float(np.max(scores))

        return stats


class TransferPerformanceCallback(NoOpMemoryOptimizedCallback):
    """Monitor source/target transfer quality and overfitting trend."""

    def __init__(
        self,
        compute_frequency: int = 1,
        evaluation_metrics: Optional[list[str]] = None,
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = max(1, compute_frequency)
        self.evaluation_metrics = evaluation_metrics or ["accuracy", "f1"]

        self.source_performance: dict[str, list[float]] = {}
        self.target_performance: dict[str, list[float]] = {}
        self.transfer_gap: dict[str, list[float]] = {}

        self.generalization_scores: list[float] = []
        self.overfitting_indicators: list[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[ObjectMap] = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        try:
            source_predictions = _as_array(logs.get("source_predictions"))
            source_labels = _as_array(logs.get("source_labels"))
            if source_predictions is not None and source_labels is not None:
                source_perf = self._compute_performance_metrics(source_predictions, source_labels)
                self._update_performance_history(self.source_performance, source_perf)

            target_predictions = _as_array(logs.get("target_predictions"))
            target_labels = _as_array(logs.get("target_labels"))
            if target_predictions is not None and target_labels is not None:
                target_perf = self._compute_performance_metrics(target_predictions, target_labels)
                self._update_performance_history(self.target_performance, target_perf)

            self._compute_transfer_gap()

            val_predictions = _as_array(logs.get("validation_predictions"))
            val_labels = _as_array(logs.get("validation_labels"))
            if val_predictions is not None and val_labels is not None:
                val_perf = self._compute_performance_metrics(val_predictions, val_labels)
                _append_bounded(
                    self.generalization_scores,
                    self._compute_generalization_score(val_perf),
                )

            if self.source_performance and self.target_performance:
                _append_bounded(
                    self.overfitting_indicators,
                    self._compute_overfitting_indicator(),
                )

            metrics: ObjectMap = {
                "epoch": context.epoch,
                "evaluation_metrics": self.evaluation_metrics,
            }
            for metric in self.evaluation_metrics:
                source_hist = self.source_performance.get(metric, [])
                target_hist = self.target_performance.get(metric, [])
                gap_hist = self.transfer_gap.get(metric, [])
                if source_hist:
                    metrics[f"source_{metric}"] = source_hist[-1]
                if target_hist:
                    metrics[f"target_{metric}"] = target_hist[-1]
                if gap_hist:
                    metrics[f"transfer_gap_{metric}"] = gap_hist[-1]

            if self.generalization_scores:
                metrics["generalization_score"] = self.generalization_scores[-1]
            if self.overfitting_indicators:
                metrics["overfitting_indicator"] = self.overfitting_indicators[-1]

            self.cache_metrics(f"transfer_performance_epoch_{context.epoch}", metrics)
        except Exception as exc:
            self.logger.error("Failed to monitor transfer performance: %s", exc)

    def _compute_performance_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        try:
            labels_flat = labels.reshape(-1)

            if predictions.ndim > 1 and predictions.shape[1] > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions.reshape(-1) > 0.5).astype(int)

            if pred_classes.shape[0] != labels_flat.shape[0]:
                return metrics

            if "accuracy" in self.evaluation_metrics:
                metrics["accuracy"] = float(accuracy_score(labels_flat, pred_classes))

            if "f1" in self.evaluation_metrics:
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    metrics["f1"] = float(
                        f1_score(labels_flat, pred_classes, average="weighted")
                    )
                else:
                    metrics["f1"] = float(f1_score(labels_flat, pred_classes))
        except Exception as exc:
            self.logger.warning("Failed to compute performance metrics: %s", exc)

        return metrics

    def _update_performance_history(
        self,
        history: dict[str, list[float]],
        current_perf: dict[str, float],
    ) -> None:
        for metric, value in current_perf.items():
            _append_bounded(history.setdefault(metric, []), value)

    def _compute_transfer_gap(self) -> None:
        for metric in self.evaluation_metrics:
            source_hist = self.source_performance.get(metric, [])
            target_hist = self.target_performance.get(metric, [])
            if source_hist and target_hist:
                _append_bounded(
                    self.transfer_gap.setdefault(metric, []),
                    source_hist[-1] - target_hist[-1],
                )

    def _compute_generalization_score(self, validation_perf: dict[str, float]) -> float:
        scores = [validation_perf[m] for m in self.evaluation_metrics if m in validation_perf]
        return float(np.mean(scores)) if scores else 0.0

    def _compute_overfitting_indicator(self) -> float:
        indicators: list[float] = []
        for metric in self.evaluation_metrics:
            source_hist = self.source_performance.get(metric, [])
            target_hist = self.target_performance.get(metric, [])
            if len(source_hist) > 1 and len(target_hist) > 1 and source_hist[0] != 0.0:
                indicators.append((source_hist[-1] - target_hist[-1]) / source_hist[0])
        return float(np.mean(indicators)) if indicators else 0.0

    def get_transfer_performance_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "evaluation_metrics": self.evaluation_metrics,
            "epochs_monitored": len(self.generalization_scores),
        }

        for metric in self.evaluation_metrics:
            source_hist = self.source_performance.get(metric, [])
            target_hist = self.target_performance.get(metric, [])
            gap_hist = self.transfer_gap.get(metric, [])

            if source_hist:
                stats[f"source_{metric}_mean"] = float(np.mean(source_hist))
                stats[f"source_{metric}_latest"] = source_hist[-1]
            if target_hist:
                stats[f"target_{metric}_mean"] = float(np.mean(target_hist))
                stats[f"target_{metric}_latest"] = target_hist[-1]
            if gap_hist:
                stats[f"transfer_gap_{metric}_mean"] = float(np.mean(gap_hist))
                stats[f"transfer_gap_{metric}_latest"] = gap_hist[-1]

        if self.generalization_scores:
            stats["generalization_score_mean"] = float(np.mean(self.generalization_scores))
            stats["generalization_score_latest"] = self.generalization_scores[-1]

        if self.overfitting_indicators:
            stats["overfitting_indicator_mean"] = float(np.mean(self.overfitting_indicators))
            stats["overfitting_indicator_latest"] = self.overfitting_indicators[-1]

        return stats


# Factory functions for easy instantiation

def create_domain_adaptation(**kwargs) -> DomainAdaptationCallback:
    """Create domain adaptation callback with default settings."""
    defaults: ObjectMap = {"compute_frequency": 1, "adaptation_method": "auto"}
    defaults.update(kwargs)
    return DomainAdaptationCallback(**defaults)


def create_fine_tuning(**kwargs) -> FineTuningCallback:
    """Create fine-tuning callback with default settings."""
    defaults: ObjectMap = {
        "compute_frequency": 1,
        "monitor_catastrophic_forgetting": True,
    }
    defaults.update(kwargs)
    return FineTuningCallback(**defaults)


def create_transfer_performance(**kwargs) -> TransferPerformanceCallback:
    """Create transfer performance callback with default settings."""
    defaults: ObjectMap = {"compute_frequency": 1, "evaluation_metrics": ["accuracy", "f1"]}
    defaults.update(kwargs)
    return TransferPerformanceCallback(**defaults)


__all__ = [
    "DomainAdaptationCallback",
    "FineTuningCallback",
    "TransferPerformanceCallback",
    "create_domain_adaptation",
    "create_fine_tuning",
    "create_transfer_performance",
]
