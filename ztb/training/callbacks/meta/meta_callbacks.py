#!/usr/bin/env python3
"""
Meta Learning Callbacks.

This module provides callbacks optimized for meta learning
tasks including MAML, few-shot learning, and adaptation monitoring.
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
    as_optional_float as _as_float,
)
from ztb.types.common import ObjectMap

_HISTORY_LIMIT = 1_000
_EPSILON = 1e-8

def _as_float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    parsed: list[float] = []
    for item in value:
        parsed_item = _as_float(item)
        if parsed_item is not None:
            parsed.append(parsed_item)
    return parsed

def _as_float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        val = _as_float(raw)
        if val is None:
            continue
        parsed[key] = val
    return parsed

def _append_bounded(history: list[float], value: float, max_len: int = _HISTORY_LIMIT) -> None:
    _append_bounded_value(history, value, max_len)

class _BaseMetaCallback(NoOpMemoryOptimizedCallback):
    """Shared base for meta callbacks with frequency gating."""

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

class MAMLCallback(_BaseMetaCallback):
    """
    MAML (Model-Agnostic Meta-Learning) monitoring callback.

    Monitors MAML training progress including inner-loop adaptation,
    meta-loss convergence, and task generalization.
    """

    def __init__(
        self,
        compute_frequency: int = 1,
        num_inner_steps: int = 5,
        adaptation_lr: float = 0.01,
    ):
        super().__init__(compute_frequency=compute_frequency)
        self.num_inner_steps = max(1, int(num_inner_steps))
        self.adaptation_lr = float(adaptation_lr)

        self.inner_losses: list[list[float]] = []
        self.meta_losses: list[float] = []
        self.adaptation_accuracies: list[list[float]] = []

        self.task_generalization_scores: list[float] = []
        self.meta_gradient_norms: list[float] = []

        self.adaptation_speeds: list[float] = []
        self.overfitting_indicators: list[float] = []

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        """Monitor MAML training progress."""
        if not self._should_process(context, logs):
            return
        assert logs is not None

        try:
            inner_loss_history = _as_float_list(logs.get("inner_losses"))
            if inner_loss_history:
                self.inner_losses.append(inner_loss_history)

            meta_loss = _as_float(logs.get("meta_loss"))
            if meta_loss is not None:
                _append_bounded(self.meta_losses, meta_loss)

            adaptation_acc_history = _as_float_list(logs.get("adaptation_accuracies"))
            if adaptation_acc_history:
                self.adaptation_accuracies.append(adaptation_acc_history)

            grad_norm = _as_float(logs.get("meta_grad_norm"))
            if grad_norm is not None:
                _append_bounded(self.meta_gradient_norms, grad_norm)

            self._compute_adaptation_metrics()
            self._compute_task_generalization()

            metrics_data: ObjectMap = {
                "epoch": context.epoch,
                "num_inner_steps": self.num_inner_steps,
                "adaptation_lr": self.adaptation_lr,
            }
            if self.meta_losses:
                metrics_data["meta_loss"] = self.meta_losses[-1]
            if self.meta_gradient_norms:
                metrics_data["meta_grad_norm"] = self.meta_gradient_norms[-1]
            if self.task_generalization_scores:
                metrics_data["task_generalization"] = self.task_generalization_scores[-1]
            if self.adaptation_speeds:
                metrics_data["adaptation_speed"] = self.adaptation_speeds[-1]

            self.cache_metrics(f"maml_epoch_{context.epoch}", metrics_data)
            self.logger.debug("MAML metrics updated for epoch %s", context.epoch)

        except Exception as exc:
            self.logger.error("Failed to monitor MAML training: %s", exc)

    def _compute_adaptation_metrics(self) -> None:
        """Compute adaptation speed and related metrics."""
        if not self.inner_losses or not self.adaptation_accuracies:
            return

        last_losses = self.inner_losses[-1]
        last_accs = self.adaptation_accuracies[-1]
        if len(last_losses) >= 2 and len(last_accs) >= 2:
            initial_loss = last_losses[0]
            final_loss = last_losses[-1]

            if initial_loss > _EPSILON:
                loss_improvement = (initial_loss - final_loss) / initial_loss
                adaptation_speed = loss_improvement / len(last_losses)
                _append_bounded(self.adaptation_speeds, float(adaptation_speed))

            initial_acc = last_accs[0]
            final_acc = last_accs[-1]
            _append_bounded(self.overfitting_indicators, float(final_acc - initial_acc))

    def _compute_task_generalization(self) -> None:
        """Compute task generalization score."""
        if len(self.meta_losses) < 2:
            return

        recent_losses = self.meta_losses[-5:]
        loss_stability = 1.0 / (1.0 + float(np.std(recent_losses)))
        _append_bounded(self.task_generalization_scores, loss_stability)

    def get_maml_stats(self) -> ObjectMap:
        """Get MAML training statistics."""
        stats: ObjectMap = {
            "num_inner_steps": self.num_inner_steps,
            "adaptation_lr": self.adaptation_lr,
            "epochs_monitored": len(self.meta_losses),
            "adaptation_sessions": len(self.inner_losses),
        }

        if self.meta_losses:
            stats.update(
                {
                    "meta_loss_mean": float(np.mean(self.meta_losses)),
                    "meta_loss_std": float(np.std(self.meta_losses)),
                    "meta_loss_latest": self.meta_losses[-1],
                    "meta_loss_trend": "improving"
                    if len(self.meta_losses) >= 2
                    and self.meta_losses[-1] < self.meta_losses[0]
                    else "stable",
                }
            )

        if self.inner_losses:
            avg_inner_losses = np.mean(self.inner_losses, axis=0)
            if len(avg_inner_losses) > 0:
                stats["avg_initial_inner_loss"] = float(avg_inner_losses[0])
                stats["avg_final_inner_loss"] = float(avg_inner_losses[-1])
            if len(avg_inner_losses) > 1:
                stats["inner_loss_improvement"] = float(
                    avg_inner_losses[0] - avg_inner_losses[-1]
                )
            else:
                stats["inner_loss_improvement"] = 0.0

        if self.adaptation_speeds:
            stats.update(
                {
                    "adaptation_speed_mean": float(np.mean(self.adaptation_speeds)),
                    "adaptation_speed_latest": self.adaptation_speeds[-1],
                }
            )

        if self.task_generalization_scores:
            stats.update(
                {
                    "task_generalization_mean": float(
                        np.mean(self.task_generalization_scores)
                    ),
                    "task_generalization_latest": self.task_generalization_scores[-1],
                }
            )

        return stats

class FewShotCallback(_BaseMetaCallback):
    """
    Few-shot learning monitoring callback.

    Monitors few-shot learning performance including N-way K-shot
    accuracy, prototype quality, and episode-based training progress.
    """

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 1,
        compute_frequency: int = 1,
        num_episodes: int | None = None,
    ):
        super().__init__(compute_frequency=compute_frequency)
        self.n_way = max(1, int(n_way))
        self.k_shot = max(1, int(k_shot))
        self.num_episodes = num_episodes

        self.episode_accuracies: list[float] = []
        self.episode_losses: list[float] = []
        self.prototype_distances: list[float] = []
        self.query_accuracies: list[float] = []
        self.query_confidences: list[float] = []
        self.episode_stats: list[ObjectMap] = []

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        """Monitor few-shot learning progress."""
        if not self._should_process(context, logs):
            return
        assert logs is not None

        try:
            episode_accuracy = _as_float(logs.get("episode_accuracy"))
            if episode_accuracy is not None:
                _append_bounded(self.episode_accuracies, episode_accuracy)

            episode_loss = _as_float(logs.get("episode_loss"))
            if episode_loss is not None:
                _append_bounded(self.episode_losses, episode_loss)

            distances = logs.get("prototype_distances")
            if isinstance(distances, (list, np.ndarray)):
                dist_array = np.asarray(distances, dtype=float)
                if dist_array.size > 0:
                    _append_bounded(self.prototype_distances, float(np.mean(dist_array)))

            query_acc = _as_float(logs.get("query_accuracy"))
            if query_acc is not None:
                _append_bounded(self.query_accuracies, query_acc)

            query_confidence = _as_float(logs.get("query_confidence"))
            if query_confidence is not None:
                _append_bounded(self.query_confidences, query_confidence)

            episode_stat: ObjectMap = {
                "epoch": context.epoch,
                "n_way": self.n_way,
                "k_shot": self.k_shot,
            }
            for key in [
                "episode_accuracy",
                "episode_loss",
                "query_accuracy",
                "query_confidence",
            ]:
                val = _as_float(logs.get(key))
                if val is not None:
                    episode_stat[key] = val
            if isinstance(distances, (list, np.ndarray)):
                episode_stat["prototype_distances"] = np.asarray(distances).tolist()
            self.episode_stats.append(episode_stat)
            if len(self.episode_stats) > _HISTORY_LIMIT:
                del self.episode_stats[: len(self.episode_stats) - _HISTORY_LIMIT]

            metrics_data: ObjectMap = {
                "epoch": context.epoch,
                "n_way": self.n_way,
                "k_shot": self.k_shot,
                "num_episodes": self.num_episodes,
            }
            if self.episode_accuracies:
                metrics_data["episode_accuracy"] = self.episode_accuracies[-1]
            if self.episode_losses:
                metrics_data["episode_loss"] = self.episode_losses[-1]
            if self.query_accuracies:
                metrics_data["query_accuracy"] = self.query_accuracies[-1]
            if self.prototype_distances:
                metrics_data["prototype_distance"] = self.prototype_distances[-1]

            self.cache_metrics(f"few_shot_epoch_{context.epoch}", metrics_data)
            self.logger.debug("Few-shot metrics updated for epoch %s", context.epoch)

        except Exception as exc:
            self.logger.error("Failed to monitor few-shot learning: %s", exc)

    def get_few_shot_stats(self) -> ObjectMap:
        """Get few-shot learning statistics."""
        stats: ObjectMap = {
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "num_episodes": self.num_episodes,
            "epochs_monitored": len(self.episode_accuracies),
            "total_episodes": len(self.episode_stats),
        }

        if self.episode_accuracies:
            stats.update(
                {
                    "episode_accuracy_mean": float(np.mean(self.episode_accuracies)),
                    "episode_accuracy_std": float(np.std(self.episode_accuracies)),
                    "episode_accuracy_latest": self.episode_accuracies[-1],
                    "episode_accuracy_best": float(np.max(self.episode_accuracies)),
                }
            )

        if self.episode_losses:
            stats.update(
                {
                    "episode_loss_mean": float(np.mean(self.episode_losses)),
                    "episode_loss_std": float(np.std(self.episode_losses)),
                    "episode_loss_latest": self.episode_losses[-1],
                    "episode_loss_best": float(np.min(self.episode_losses)),
                }
            )

        if self.query_accuracies:
            stats.update(
                {
                    "query_accuracy_mean": float(np.mean(self.query_accuracies)),
                    "query_accuracy_latest": self.query_accuracies[-1],
                }
            )

        if self.prototype_distances:
            stats.update(
                {
                    "prototype_distance_mean": float(np.mean(self.prototype_distances)),
                    "prototype_distance_std": float(np.std(self.prototype_distances)),
                    "prototype_distance_latest": self.prototype_distances[-1],
                }
            )

        return stats

class MetaAdaptationCallback(_BaseMetaCallback):
    """
    Meta adaptation monitoring callback.

    Monitors meta-learning adaptation processes including adaptation curves,
    convergence analysis, and cross-task generalization.
    """

    def __init__(
        self,
        compute_frequency: int = 1,
        adaptation_steps: int = 10,
        stability_threshold: float = 0.01,
    ):
        super().__init__(compute_frequency=compute_frequency)
        self.adaptation_steps = max(1, int(adaptation_steps))
        self.stability_threshold = float(stability_threshold)

        self.adaptation_curves: list[list[float]] = []
        self.adaptation_speeds: list[float] = []
        self.convergence_steps: list[int] = []
        self.adaptation_stability: list[float] = []
        self.task_similarity_scores: list[float] = []
        self.cross_task_performance: dict[str, list[float]] = {}
        self.adaptation_generalization: list[float] = []

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        """Monitor meta adaptation progress."""
        if not self._should_process(context, logs):
            return
        assert logs is not None

        try:
            curve = _as_float_list(logs.get("adaptation_curve"))
            if curve:
                self.adaptation_curves.append(curve)

            convergence_step = logs.get("convergence_step")
            if isinstance(convergence_step, int):
                self.convergence_steps.append(convergence_step)

            cross_perf = _as_float_map(logs.get("cross_task_performance"))
            for task, perf in cross_perf.items():
                _append_bounded(
                    self.cross_task_performance.setdefault(task, []),
                    perf,
                )

            self._compute_adaptation_metrics()

            metrics_data: ObjectMap = {
                "epoch": context.epoch,
                "adaptation_steps": self.adaptation_steps,
                "stability_threshold": self.stability_threshold,
            }
            if self.adaptation_speeds:
                metrics_data["adaptation_speed"] = self.adaptation_speeds[-1]
            if self.convergence_steps:
                metrics_data["convergence_step"] = self.convergence_steps[-1]
            if self.adaptation_stability:
                metrics_data["adaptation_stability"] = self.adaptation_stability[-1]
            if self.adaptation_generalization:
                metrics_data["adaptation_generalization"] = self.adaptation_generalization[-1]

            self.cache_metrics(f"meta_adaptation_epoch_{context.epoch}", metrics_data)
            self.logger.debug(
                "Meta adaptation metrics updated for epoch %s", context.epoch
            )

        except Exception as exc:
            self.logger.error("Failed to monitor meta adaptation: %s", exc)

    def _compute_adaptation_metrics(self) -> None:
        """Compute adaptation speed, stability, and generalization."""
        if not self.adaptation_curves:
            return

        current_curve = self.adaptation_curves[-1]
        if len(current_curve) >= 2:
            initial_perf = current_curve[0]
            early_perf = current_curve[min(2, len(current_curve) - 1)]
            if abs(initial_perf) > _EPSILON:
                adaptation_speed = (early_perf - initial_perf) / abs(initial_perf)
                _append_bounded(self.adaptation_speeds, float(adaptation_speed))

            if len(current_curve) > 3:
                recent_perfs = current_curve[-3:]
                stability = 1.0 / (1.0 + float(np.std(recent_perfs)))
                _append_bounded(self.adaptation_stability, float(stability))

        if not self.convergence_steps and len(current_curve) >= 3:
            for i in range(2, len(current_curve)):
                recent_changes = [
                    abs(current_curve[j] - current_curve[j - 1])
                    for j in range(i - 2, i)
                ]
                if all(change < self.stability_threshold for change in recent_changes):
                    self.convergence_steps.append(i)
                    break

        if self.cross_task_performance:
            current_perfs = [
                task_perf[-1]
                for task_perf in self.cross_task_performance.values()
                if task_perf
            ]
            if current_perfs:
                _append_bounded(
                    self.adaptation_generalization,
                    float(np.mean(current_perfs)),
                )

    def get_meta_adaptation_stats(self) -> ObjectMap:
        """Get meta adaptation statistics."""
        stats: ObjectMap = {
            "adaptation_steps": self.adaptation_steps,
            "stability_threshold": self.stability_threshold,
            "epochs_monitored": len(self.adaptation_curves),
            "cross_tasks_tracked": len(self.cross_task_performance),
        }

        if self.adaptation_speeds:
            stats.update(
                {
                    "adaptation_speed_mean": float(np.mean(self.adaptation_speeds)),
                    "adaptation_speed_std": float(np.std(self.adaptation_speeds)),
                    "adaptation_speed_latest": self.adaptation_speeds[-1],
                }
            )

        if self.convergence_steps:
            stats.update(
                {
                    "avg_convergence_steps": float(np.mean(self.convergence_steps)),
                    "min_convergence_steps": int(np.min(self.convergence_steps)),
                    "max_convergence_steps": int(np.max(self.convergence_steps)),
                }
            )

        if self.adaptation_stability:
            stats.update(
                {
                    "adaptation_stability_mean": float(
                        np.mean(self.adaptation_stability)
                    ),
                    "adaptation_stability_latest": self.adaptation_stability[-1],
                }
            )

        if self.adaptation_generalization:
            stats.update(
                {
                    "adaptation_generalization_mean": float(
                        np.mean(self.adaptation_generalization)
                    ),
                    "adaptation_generalization_latest": self.adaptation_generalization[-1],
                }
            )

        return stats

# Factory functions for easy instantiation
def create_maml(**kwargs: object) -> MAMLCallback:
    """Create MAML callback with default settings."""
    compute_frequency = kwargs.get("compute_frequency", 1)
    num_inner_steps = kwargs.get("num_inner_steps", 5)
    adaptation_lr = kwargs.get("adaptation_lr", 0.01)
    return MAMLCallback(
        compute_frequency=int(compute_frequency) if isinstance(compute_frequency, int) else 1,
        num_inner_steps=int(num_inner_steps) if isinstance(num_inner_steps, int) else 5,
        adaptation_lr=float(adaptation_lr) if isinstance(adaptation_lr, (int, float)) else 0.01,
    )

def create_meta_adaptation(**kwargs: object) -> MetaAdaptationCallback:
    """Create meta adaptation callback with default settings."""
    compute_frequency = kwargs.get("compute_frequency", 1)
    adaptation_steps = kwargs.get("adaptation_steps", 10)
    stability_threshold = kwargs.get("stability_threshold", 0.01)
    return MetaAdaptationCallback(
        compute_frequency=int(compute_frequency) if isinstance(compute_frequency, int) else 1,
        adaptation_steps=int(adaptation_steps) if isinstance(adaptation_steps, int) else 10,
        stability_threshold=float(stability_threshold)
        if isinstance(stability_threshold, (int, float))
        else 0.01,
    )
