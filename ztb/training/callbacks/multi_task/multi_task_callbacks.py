#!/usr/bin/env python3
"""
Multi-Task Learning Callbacks.

This module provides callbacks optimized for multi-task learning
including task balancing, shared representation monitoring, and
task interference assessment.
"""

from __future__ import annotations

import logging

import numpy as np

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    NoOpMemoryOptimizedCallback,
)
from ztb.training.callbacks.shared.utils.value_utils import (
    append_bounded as _append_bounded_value,
    as_optional_array as _as_array,
    as_optional_float as _as_float,
)
from ztb.types.common import ObjectMap

_HISTORY_LIMIT = 1_000
_EPSILON = 1e-8

def _append_bounded(history: list[float], value: float, max_len: int = _HISTORY_LIMIT) -> None:
    _append_bounded_value(history, value, max_len)

class _BaseFrequencyCallback(NoOpMemoryOptimizedCallback):
    """Shared base class for frequency-gated callback processing."""

    def __init__(self, compute_frequency: int = 1):
        super().__init__(cache_size=1000)
        self.compute_frequency = max(1, int(compute_frequency))
        self.logger = logging.getLogger(__name__)

    def _should_process(
        self, context: LearningContext, logs: ObjectMap | None
    ) -> bool:
        if logs is None:
            return False
        return context.epoch % self.compute_frequency == 0

class TaskBalancingCallback(_BaseFrequencyCallback):
    """
    Task balancing monitoring callback.

    Monitors the balance between different tasks in multi-task learning,
    including loss weighting, gradient magnitudes, and task performance.
    """

    def __init__(
        self,
        task_names: list[str],
        compute_frequency: int = 1,
        balance_threshold: float = 0.1,
    ):
        super().__init__(compute_frequency=compute_frequency)
        self.task_names = [task for task in task_names if isinstance(task, str)]
        self.balance_threshold = float(balance_threshold)

        self.task_losses: dict[str, list[float]] = {
            task: [] for task in self.task_names
        }
        self.task_weights: dict[str, list[float]] = {
            task: [] for task in self.task_names
        }
        self.task_gradients: dict[str, list[float]] = {
            task: [] for task in self.task_names
        }

        self.task_loss_ratios: dict[str, list[float]] = {}
        self.balance_scores: list[float] = []
        self.imbalance_warnings: list[str] = []

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        """Monitor task balancing."""
        if not self._should_process(context, logs):
            return
        assert logs is not None

        try:
            current_losses: dict[str, float] = {}
            total_loss = 0.0

            for task in self.task_names:
                loss = _as_float(logs.get(f"{task}_loss"))
                if loss is None:
                    continue
                _append_bounded(self.task_losses[task], loss)
                current_losses[task] = loss
                total_loss += loss

            for task in self.task_names:
                weight = _as_float(logs.get(f"{task}_weight"))
                if weight is not None:
                    _append_bounded(self.task_weights[task], weight)

                grad_norm = _as_float(logs.get(f"{task}_grad_norm"))
                if grad_norm is not None:
                    _append_bounded(self.task_gradients[task], grad_norm)

            if total_loss > 0 and current_losses:
                self._compute_balance_metrics(current_losses, total_loss)
            self._check_task_imbalance()

            metrics_data: ObjectMap = {
                "epoch": context.epoch,
                "task_names": self.task_names,
                "balance_threshold": self.balance_threshold,
            }
            for task in self.task_names:
                if self.task_losses[task]:
                    metrics_data[f"{task}_loss"] = self.task_losses[task][-1]
                if self.task_weights[task]:
                    metrics_data[f"{task}_weight"] = self.task_weights[task][-1]
                if self.task_gradients[task]:
                    metrics_data[f"{task}_grad_norm"] = self.task_gradients[task][-1]
            if self.balance_scores:
                metrics_data["balance_score"] = self.balance_scores[-1]

            self.cache_metrics(f"task_balancing_epoch_{context.epoch}", metrics_data)
            self.logger.debug("Task balancing metrics updated for epoch %s", context.epoch)

        except Exception as exc:
            self.logger.error("Failed to monitor task balancing: %s", exc)

    def _compute_balance_metrics(
        self, current_losses: dict[str, float], total_loss: float
    ) -> None:
        """Compute task balance metrics."""
        for task, loss in current_losses.items():
            ratio = loss / total_loss
            history = self.task_loss_ratios.setdefault(task, [])
            _append_bounded(history, ratio)

        if len(current_losses) <= 1:
            return
        ratios = np.asarray(list(current_losses.values()), dtype=float)
        mean_ratio = float(np.mean(ratios))
        balance_score = float(np.std(ratios) / (mean_ratio + _EPSILON))
        _append_bounded(self.balance_scores, balance_score)

    def _check_task_imbalance(self) -> None:
        """Check for task imbalance and generate warnings."""
        if not self.balance_scores:
            return

        current_balance = self.balance_scores[-1]
        if current_balance <= self.balance_threshold:
            return

        warning = (
            f"Task imbalance detected at epoch {len(self.balance_scores)}: "
            f"balance score {current_balance:.4f} > threshold {self.balance_threshold}"
        )

        latest_ratios = {
            task: ratios[-1]
            for task, ratios in self.task_loss_ratios.items()
            if ratios
        }
        if latest_ratios:
            max_task = max(latest_ratios, key=latest_ratios.get)
            min_task = min(latest_ratios, key=latest_ratios.get)
            warning += (
                f" - {max_task} dominates ({latest_ratios[max_task]:.3f}) vs "
                f"{min_task} ({latest_ratios[min_task]:.3f})"
            )

        self.imbalance_warnings.append(warning)
        self.logger.warning(warning)

    def get_task_balancing_stats(self) -> ObjectMap:
        """Get task balancing statistics."""
        stats: ObjectMap = {
            "task_names": self.task_names,
            "balance_threshold": self.balance_threshold,
            "imbalance_warnings_count": len(self.imbalance_warnings),
            "epochs_monitored": len(self.balance_scores),
        }

        for task in self.task_names:
            losses = self.task_losses[task]
            if losses:
                stats.update(
                    {
                        f"{task}_loss_mean": float(np.mean(losses)),
                        f"{task}_loss_latest": losses[-1],
                        f"{task}_loss_std": float(np.std(losses)),
                    }
                )

            weights = self.task_weights[task]
            if weights:
                stats.update(
                    {
                        f"{task}_weight_mean": float(np.mean(weights)),
                        f"{task}_weight_latest": weights[-1],
                    }
                )

        if self.balance_scores:
            stats.update(
                {
                    "balance_score_mean": float(np.mean(self.balance_scores)),
                    "balance_score_latest": self.balance_scores[-1],
                    "balance_score_std": float(np.std(self.balance_scores)),
                    "imbalance_epochs": sum(
                        1 for score in self.balance_scores if score > self.balance_threshold
                    ),
                }
            )

        return stats

class SharedRepresentationCallback(_BaseFrequencyCallback):
    """Shared representation monitoring callback."""

    def __init__(
        self,
        compute_frequency: int = 1,
        representation_layers: list[str] | None = None,
    ):
        super().__init__(compute_frequency=compute_frequency)
        if representation_layers:
            self.representation_layers = [
                layer for layer in representation_layers if isinstance(layer, str)
            ]
        else:
            self.representation_layers = ["shared_encoder"]

        self.representation_diversity: dict[str, list[float]] = {}
        self.representation_stability: dict[str, list[float]] = {}
        self.task_alignment_scores: dict[str, list[float]] = {}

        self.layer_activations: dict[str, list[np.ndarray]] = {}
        self.layer_gradients: dict[str, list[float]] = {}

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        """Monitor shared representations."""
        if not self._should_process(context, logs):
            return
        assert logs is not None

        try:
            for layer in self.representation_layers:
                activations = _as_array(logs.get(f"{layer}_activations"))
                if activations is not None:
                    self.layer_activations.setdefault(layer, []).append(activations.copy())

                grad_norm = _as_float(logs.get(f"{layer}_grad_norm"))
                if grad_norm is not None:
                    _append_bounded(self.layer_gradients.setdefault(layer, []), grad_norm)

            self._compute_representation_metrics()

            metrics_data: ObjectMap = {
                "epoch": context.epoch,
                "representation_layers": self.representation_layers,
            }
            for layer in self.representation_layers:
                diversity = self.representation_diversity.get(layer, [])
                if diversity:
                    metrics_data[f"{layer}_diversity"] = diversity[-1]

                stability = self.representation_stability.get(layer, [])
                if stability:
                    metrics_data[f"{layer}_stability"] = stability[-1]

                gradients = self.layer_gradients.get(layer, [])
                if gradients:
                    metrics_data[f"{layer}_grad_norm"] = gradients[-1]

            self.cache_metrics(
                f"shared_representation_epoch_{context.epoch}",
                metrics_data,
            )
            self.logger.debug(
                "Shared representation metrics updated for epoch %s", context.epoch
            )

        except Exception as exc:
            self.logger.error("Failed to monitor shared representations: %s", exc)

    def _compute_representation_metrics(self) -> None:
        """Compute representation quality metrics."""
        for layer in self.representation_layers:
            activations_history = self.layer_activations.get(layer, [])
            if not activations_history:
                continue

            current_activations = activations_history[-1]
            if current_activations.ndim >= 2:
                feature_variance = np.var(current_activations, axis=0)
                diversity = float(np.mean(feature_variance))
                _append_bounded(
                    self.representation_diversity.setdefault(layer, []),
                    diversity,
                )

            if len(activations_history) >= 2:
                previous_activations = activations_history[-2]
                if current_activations.shape == previous_activations.shape:
                    activation_change = float(
                        np.mean(np.abs(current_activations - previous_activations))
                    )
                    stability = 1.0 / (1.0 + activation_change)
                    _append_bounded(
                        self.representation_stability.setdefault(layer, []),
                        stability,
                    )

    def get_shared_representation_stats(self) -> ObjectMap:
        """Get shared representation statistics."""
        stats: ObjectMap = {
            "representation_layers": self.representation_layers,
            "epochs_monitored": max(
                (len(acts) for acts in self.layer_activations.values()),
                default=0,
            ),
        }

        for layer in self.representation_layers:
            diversity = self.representation_diversity.get(layer, [])
            if diversity:
                stats.update(
                    {
                        f"{layer}_diversity_mean": float(np.mean(diversity)),
                        f"{layer}_diversity_latest": diversity[-1],
                    }
                )

            stability = self.representation_stability.get(layer, [])
            if stability:
                stats.update(
                    {
                        f"{layer}_stability_mean": float(np.mean(stability)),
                        f"{layer}_stability_latest": stability[-1],
                    }
                )

            gradients = self.layer_gradients.get(layer, [])
            if gradients:
                stats.update(
                    {
                        f"{layer}_grad_norm_mean": float(np.mean(gradients)),
                        f"{layer}_grad_norm_latest": gradients[-1],
                    }
                )

        return stats

class TaskInterferenceCallback(_BaseFrequencyCallback):
    """Task interference monitoring callback."""

    def __init__(
        self,
        task_names: list[str],
        compute_frequency: int = 1,
        interference_threshold: float = -0.05,
    ):
        super().__init__(compute_frequency=compute_frequency)
        self.task_names = [task for task in task_names if isinstance(task, str)]
        self.interference_threshold = float(interference_threshold)

        self.task_performance_history: dict[str, list[float]] = {
            task: [] for task in self.task_names
        }
        self.task_interference_scores: dict[str, list[float]] = {
            task: [] for task in self.task_names
        }
        self.task_performance_correlations: dict[tuple[str, str], list[float]] = {}
        self.interference_events: list[ObjectMap] = []
        self.negative_transfer_detected = False

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if not self._should_process(context, logs):
            return
        assert logs is not None

        try:
            current_performances: dict[str, float] = {}
            for task in self.task_names:
                performance = _as_float(logs.get(f"{task}_performance"))
                if performance is None:
                    continue
                _append_bounded(self.task_performance_history[task], performance)
                current_performances[task] = performance

            if len(current_performances) == len(self.task_names):
                self._compute_interference_scores(current_performances)
            self._detect_negative_transfer()
            self._compute_task_correlations()

            metrics_data: ObjectMap = {
                "epoch": context.epoch,
                "task_names": self.task_names,
                "interference_threshold": self.interference_threshold,
                "negative_transfer_detected": self.negative_transfer_detected,
            }
            for task in self.task_names:
                if task in current_performances:
                    metrics_data[f"{task}_performance"] = current_performances[task]
                scores = self.task_interference_scores.get(task, [])
                if scores:
                    metrics_data[f"{task}_interference"] = scores[-1]

            self.cache_metrics(
                f"task_interference_epoch_{context.epoch}",
                metrics_data,
            )
            self.logger.debug(
                "Task interference metrics updated for epoch %s", context.epoch
            )

        except Exception as exc:
            self.logger.error("Failed to monitor task interference: %s", exc)

    def _compute_interference_scores(self, current_performances: dict[str, float]) -> None:
        for task in self.task_names:
            history = self.task_performance_history.get(task, [])
            if not history:
                continue

            initial_perf = history[0]
            current_perf = current_performances[task]
            if abs(initial_perf) <= _EPSILON:
                continue

            interference_score = (current_perf - initial_perf) / initial_perf
            _append_bounded(
                self.task_interference_scores.setdefault(task, []),
                float(interference_score),
            )

            if interference_score < self.interference_threshold:
                event: ObjectMap = {
                    "epoch": len(history),
                    "task": task,
                    "interference_score": float(interference_score),
                    "initial_performance": float(initial_perf),
                    "current_performance": float(current_perf),
                }
                self.interference_events.append(event)
                self.logger.warning(
                    "Negative interference detected for %s: %.4f",
                    task,
                    interference_score,
                )

    def _detect_negative_transfer(self) -> None:
        current_interferences = [
            scores[-1]
            for scores in self.task_interference_scores.values()
            if scores
        ]
        if not current_interferences:
            return

        negative_tasks = sum(1 for score in current_interferences if score < 0)
        negative_ratio = negative_tasks / len(current_interferences)
        if negative_ratio > 0.5 and not self.negative_transfer_detected:
            self.negative_transfer_detected = True
            self.logger.warning(
                "Negative transfer detected: %.2f of tasks showing performance degradation",
                negative_ratio,
            )

    def _compute_task_correlations(self) -> None:
        if not all(
            len(history) >= 3 for history in self.task_performance_history.values()
        ):
            return

        for idx, task1 in enumerate(self.task_names):
            for task2 in self.task_names[idx + 1 :]:
                perf1 = self.task_performance_history[task1]
                perf2 = self.task_performance_history[task2]
                if len(perf1) != len(perf2):
                    continue
                correlation = float(np.corrcoef(perf1, perf2)[0, 1])
                if not np.isfinite(correlation):
                    continue
                _append_bounded(
                    self.task_performance_correlations.setdefault((task1, task2), []),
                    correlation,
                )

    def get_task_interference_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "task_names": self.task_names,
            "interference_threshold": self.interference_threshold,
            "negative_transfer_detected": self.negative_transfer_detected,
            "interference_events_count": len(self.interference_events),
            "epochs_monitored": max(
                (len(history) for history in self.task_performance_history.values()),
                default=0,
            ),
        }

        for task in self.task_names:
            scores = self.task_interference_scores.get(task, [])
            if scores:
                stats.update(
                    {
                        f"{task}_interference_mean": float(np.mean(scores)),
                        f"{task}_interference_latest": scores[-1],
                        f"{task}_negative_interference_count": sum(
                            1 for score in scores if score < 0
                        ),
                    }
                )

        all_correlations: list[float] = []
        for pair_corrs in self.task_performance_correlations.values():
            all_correlations.extend(pair_corrs)
        if all_correlations:
            stats.update(
                {
                    "avg_task_correlation": float(np.mean(all_correlations)),
                    "task_correlation_std": float(np.std(all_correlations)),
                }
            )

        return stats

def create_shared_representation(**kwargs: object) -> SharedRepresentationCallback:
    """Create a shared representation callback with default settings."""
    compute_frequency = kwargs.get("compute_frequency", 1)
    layers = kwargs.get("representation_layers")
    parsed_frequency = int(compute_frequency) if isinstance(compute_frequency, int) else 1
    parsed_layers = (
        [layer for layer in layers if isinstance(layer, str)]
        if isinstance(layers, list)
        else None
    )
    return SharedRepresentationCallback(
        compute_frequency=parsed_frequency,
        representation_layers=parsed_layers,
    )

def create_task_interference(
    task_names: list[str], **kwargs: object
) -> TaskInterferenceCallback:
    """Create a task interference callback with default settings."""
    compute_frequency = kwargs.get("compute_frequency", 1)
    interference_threshold = kwargs.get("interference_threshold", -0.05)
    parsed_frequency = int(compute_frequency) if isinstance(compute_frequency, int) else 1
    parsed_threshold = (
        float(interference_threshold)
        if isinstance(interference_threshold, (int, float))
        else -0.05
    )
    return TaskInterferenceCallback(
        task_names=task_names,
        compute_frequency=parsed_frequency,
        interference_threshold=parsed_threshold,
    )
