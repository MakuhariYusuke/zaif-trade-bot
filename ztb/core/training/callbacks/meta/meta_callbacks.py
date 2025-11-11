#!/usr/bin/env python3
"""
Meta Learning Callbacks.

This module provides callbacks optimized for meta learning
tasks including MAML, few-shot learning, and adaptation monitoring.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    MemoryOptimizedCallback,
)


class MAMLCallback(MemoryOptimizedCallback):
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
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency
        self.num_inner_steps = num_inner_steps
        self.adaptation_lr = adaptation_lr

        # MAML-specific metrics
        self.inner_losses: List[List[float]] = []  # Losses for each inner step
        self.meta_losses: List[float] = []
        self.adaptation_accuracies: List[
            List[float]
        ] = []  # Accuracies after each inner step

        # Task generalization
        self.task_generalization_scores: List[float] = []
        self.meta_gradient_norms: List[float] = []

        # Adaptation monitoring
        self.adaptation_speeds: List[float] = []
        self.overfitting_indicators: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor MAML training progress."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track inner-loop losses
            if "inner_losses" in logs:
                inner_loss_history = logs["inner_losses"]
                if isinstance(inner_loss_history, list):
                    self.inner_losses.append(
                        [float(loss) for loss in inner_loss_history]
                    )

            # Track meta loss
            if "meta_loss" in logs:
                meta_loss = float(logs["meta_loss"])
                self.meta_losses.append(meta_loss)

            # Track adaptation accuracies
            if "adaptation_accuracies" in logs:
                adaptation_acc_history = logs["adaptation_accuracies"]
                if isinstance(adaptation_acc_history, list):
                    self.adaptation_accuracies.append(
                        [float(acc) for acc in adaptation_acc_history]
                    )

            # Track meta gradient norm
            if "meta_grad_norm" in logs:
                grad_norm = float(logs["meta_grad_norm"])
                self.meta_gradient_norms.append(grad_norm)

            # Compute adaptation metrics
            self._compute_adaptation_metrics()

            # Compute task generalization
            self._compute_task_generalization()

            # Cache MAML metrics
            metrics_key = f"maml_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "num_inner_steps": self.num_inner_steps,
                "adaptation_lr": self.adaptation_lr,
            }

            # Add current metrics
            if self.meta_losses:
                metrics_data["meta_loss"] = self.meta_losses[-1]
            if self.meta_gradient_norms:
                metrics_data["meta_grad_norm"] = self.meta_gradient_norms[-1]
            if self.task_generalization_scores:
                metrics_data["task_generalization"] = self.task_generalization_scores[
                    -1
                ]
            if self.adaptation_speeds:
                metrics_data["adaptation_speed"] = self.adaptation_speeds[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(f"MAML metrics updated for epoch {context.epoch}")

        except Exception as e:
            self.logger.error(f"Failed to monitor MAML training: {e}")

    def _compute_adaptation_metrics(self) -> None:
        """Compute adaptation speed and related metrics."""
        if not self.inner_losses or not self.adaptation_accuracies:
            return

        # Compute adaptation speed (improvement per inner step)
        if len(self.inner_losses[-1]) >= 2 and len(self.adaptation_accuracies[-1]) >= 2:
            initial_loss = self.inner_losses[-1][0]
            final_loss = self.inner_losses[-1][-1]

            if initial_loss > 0:
                loss_improvement = (initial_loss - final_loss) / initial_loss
                adaptation_speed = loss_improvement / len(self.inner_losses[-1])
                self.adaptation_speeds.append(float(adaptation_speed))

            # Compute overfitting indicator (train vs validation performance gap)
            if len(self.adaptation_accuracies[-1]) >= 2:
                initial_acc = self.adaptation_accuracies[-1][0]
                final_acc = self.adaptation_accuracies[-1][-1]
                overfitting = final_acc - initial_acc  # Negative indicates overfitting
                self.overfitting_indicators.append(float(overfitting))

    def _compute_task_generalization(self) -> None:
        """Compute task generalization score."""
        if len(self.meta_losses) < 2:
            return

        # Simple generalization metric: stability of meta loss
        recent_losses = (
            self.meta_losses[-5:] if len(self.meta_losses) >= 5 else self.meta_losses
        )
        loss_stability = 1.0 / (1.0 + np.std(recent_losses))

        self.task_generalization_scores.append(float(loss_stability))

    def get_maml_stats(self) -> Dict[str, Any]:
        """Get MAML training statistics."""
        stats = {
            "num_inner_steps": self.num_inner_steps,
            "adaptation_lr": self.adaptation_lr,
            "epochs_monitored": len(self.meta_losses),
            "adaptation_sessions": len(self.inner_losses),
        }

        # Meta loss stats
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

        # Inner loss stats
        if self.inner_losses:
            avg_inner_losses = np.mean(
                [losses for losses in self.inner_losses if losses], axis=0
            )
            stats.update(
                {
                    "avg_initial_inner_loss": float(avg_inner_losses[0])
                    if len(avg_inner_losses) > 0
                    else 0,
                    "avg_final_inner_loss": float(avg_inner_losses[-1])
                    if len(avg_inner_losses) > 0
                    else 0,
                    "inner_loss_improvement": float(
                        avg_inner_losses[0] - avg_inner_losses[-1]
                    )
                    if len(avg_inner_losses) > 1
                    else 0,
                }
            )

        # Adaptation stats
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

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of training."""
        pass

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of training."""
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
    ) -> None:
        """Called at the end of each batch."""
        pass


class FewShotCallback(MemoryOptimizedCallback):
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
        num_episodes: Optional[int] = None,
    ):
        super().__init__(cache_size=1000)
        self.n_way = n_way
        self.k_shot = k_shot
        self.compute_frequency = compute_frequency
        self.num_episodes = num_episodes

        # Few-shot performance metrics
        self.episode_accuracies: List[float] = []
        self.episode_losses: List[float] = []
        self.prototype_distances: List[float] = []

        # Query set performance
        self.query_accuracies: List[float] = []
        self.query_confidences: List[float] = []

        # Episode statistics
        self.episode_stats: List[Dict[str, Any]] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor few-shot learning progress."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track episode accuracy
            if "episode_accuracy" in logs:
                accuracy = float(logs["episode_accuracy"])
                self.episode_accuracies.append(accuracy)

            # Track episode loss
            if "episode_loss" in logs:
                loss = float(logs["episode_loss"])
                self.episode_losses.append(loss)

            # Track prototype quality
            if "prototype_distances" in logs:
                distances = logs["prototype_distances"]
                if isinstance(distances, (list, np.ndarray)):
                    avg_distance = float(np.mean(distances))
                    self.prototype_distances.append(avg_distance)

            # Track query performance
            if "query_accuracy" in logs:
                query_acc = float(logs["query_accuracy"])
                self.query_accuracies.append(query_acc)

            if "query_confidence" in logs:
                confidence = float(logs["query_confidence"])
                self.query_confidences.append(confidence)

            # Store episode statistics
            episode_stat = {
                "epoch": context.epoch,
                "n_way": self.n_way,
                "k_shot": self.k_shot,
            }

            for key in [
                "episode_accuracy",
                "episode_loss",
                "query_accuracy",
                "prototype_distances",
                "query_confidence",
            ]:
                if key in logs:
                    episode_stat[key] = (
                        float(logs[key])
                        if not isinstance(logs[key], (list, np.ndarray))
                        else logs[key]
                    )

            self.episode_stats.append(episode_stat)

            # Cache few-shot metrics
            metrics_key = f"few_shot_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "n_way": self.n_way,
                "k_shot": self.k_shot,
                "num_episodes": self.num_episodes,
            }

            # Add current metrics
            if self.episode_accuracies:
                metrics_data["episode_accuracy"] = self.episode_accuracies[-1]
            if self.episode_losses:
                metrics_data["episode_loss"] = self.episode_losses[-1]
            if self.query_accuracies:
                metrics_data["query_accuracy"] = self.query_accuracies[-1]
            if self.prototype_distances:
                metrics_data["prototype_distance"] = self.prototype_distances[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(f"Few-shot metrics updated for epoch {context.epoch}")

        except Exception as e:
            self.logger.error(f"Failed to monitor few-shot learning: {e}")

    def get_few_shot_stats(self) -> Dict[str, Any]:
        """Get few-shot learning statistics."""
        stats = {
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "num_episodes": self.num_episodes,
            "epochs_monitored": len(self.episode_accuracies),
            "total_episodes": len(self.episode_stats),
        }

        # Episode accuracy stats
        if self.episode_accuracies:
            stats.update(
                {
                    "episode_accuracy_mean": float(np.mean(self.episode_accuracies)),
                    "episode_accuracy_std": float(np.std(self.episode_accuracies)),
                    "episode_accuracy_latest": self.episode_accuracies[-1],
                    "episode_accuracy_best": float(np.max(self.episode_accuracies)),
                }
            )

        # Episode loss stats
        if self.episode_losses:
            stats.update(
                {
                    "episode_loss_mean": float(np.mean(self.episode_losses)),
                    "episode_loss_std": float(np.std(self.episode_losses)),
                    "episode_loss_latest": self.episode_losses[-1],
                    "episode_loss_best": float(np.min(self.episode_losses)),
                }
            )

        # Query performance stats
        if self.query_accuracies:
            stats.update(
                {
                    "query_accuracy_mean": float(np.mean(self.query_accuracies)),
                    "query_accuracy_latest": self.query_accuracies[-1],
                }
            )

        # Prototype quality stats
        if self.prototype_distances:
            stats.update(
                {
                    "prototype_distance_mean": float(np.mean(self.prototype_distances)),
                    "prototype_distance_std": float(np.std(self.prototype_distances)),
                    "prototype_distance_latest": self.prototype_distances[-1],
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
    ) -> None:
        """Called at the end of each batch."""
        pass


class MetaAdaptationCallback(MemoryOptimizedCallback):
    """
    Meta adaptation monitoring callback.

    Monitors adaptation to new tasks in meta learning scenarios,
    including adaptation speed, stability, and generalization.
    """

    def __init__(
        self,
        compute_frequency: int = 1,
        adaptation_steps: int = 10,
        stability_threshold: float = 0.01,
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency
        self.adaptation_steps = adaptation_steps
        self.stability_threshold = stability_threshold

        # Adaptation tracking
        self.adaptation_curves: List[
            List[float]
        ] = []  # Performance over adaptation steps
        self.adaptation_speeds: List[float] = []
        self.convergence_steps: List[int] = []

        # Stability metrics
        self.adaptation_stability: List[float] = []
        self.task_similarity_scores: List[float] = []

        # Generalization metrics
        self.cross_task_performance: Dict[str, List[float]] = {}
        self.adaptation_generalization: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor meta adaptation progress."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track adaptation curves
            if "adaptation_curve" in logs:
                curve = logs["adaptation_curve"]
                if isinstance(curve, list):
                    self.adaptation_curves.append([float(val) for val in curve])

            # Track convergence information
            if "convergence_step" in logs:
                conv_step = int(logs["convergence_step"])
                self.convergence_steps.append(conv_step)

            # Track cross-task performance
            if "cross_task_performance" in logs:
                cross_perf = logs["cross_task_performance"]
                if isinstance(cross_perf, dict):
                    for task, perf in cross_perf.items():
                        if task not in self.cross_task_performance:
                            self.cross_task_performance[task] = []
                        self.cross_task_performance[task].append(float(perf))

            # Compute adaptation metrics
            self._compute_adaptation_metrics()

            # Cache adaptation metrics
            metrics_key = f"meta_adaptation_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "adaptation_steps": self.adaptation_steps,
                "stability_threshold": self.stability_threshold,
            }

            # Add current metrics
            if self.adaptation_speeds:
                metrics_data["adaptation_speed"] = self.adaptation_speeds[-1]
            if self.convergence_steps:
                metrics_data["convergence_step"] = self.convergence_steps[-1]
            if self.adaptation_stability:
                metrics_data["adaptation_stability"] = self.adaptation_stability[-1]
            if self.adaptation_generalization:
                metrics_data[
                    "adaptation_generalization"
                ] = self.adaptation_generalization[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Meta adaptation metrics updated for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to monitor meta adaptation: {e}")

    def _compute_adaptation_metrics(self) -> None:
        """Compute adaptation speed, stability, and generalization."""
        if not self.adaptation_curves:
            return

        current_curve = self.adaptation_curves[-1]

        if len(current_curve) >= 2:
            # Compute adaptation speed (initial improvement rate)
            initial_perf = current_curve[0]
            early_perf = current_curve[min(2, len(current_curve) - 1)]

            if initial_perf != 0:
                adaptation_speed = (early_perf - initial_perf) / abs(initial_perf)
                self.adaptation_speeds.append(float(adaptation_speed))

            # Compute stability (performance variance during adaptation)
            if len(current_curve) > 3:
                recent_perfs = current_curve[-3:]
                stability = 1.0 / (1.0 + np.std(recent_perfs))
                self.adaptation_stability.append(float(stability))

        # Compute convergence step if not provided
        if not self.convergence_steps and len(current_curve) >= 3:
            # Find when performance stabilizes
            for i in range(2, len(current_curve)):
                recent_changes = [
                    abs(current_curve[j] - current_curve[j - 1])
                    for j in range(i - 2, i)
                ]
                if all(change < self.stability_threshold for change in recent_changes):
                    self.convergence_steps.append(i)
                    break

        # Compute adaptation generalization
        if self.cross_task_performance:
            # Average performance across different tasks
            current_perfs = []
            for task_perfs in self.cross_task_performance.values():
                if task_perfs:
                    current_perfs.append(task_perfs[-1])

            if current_perfs:
                generalization = np.mean(current_perfs)
                self.adaptation_generalization.append(float(generalization))

    def get_meta_adaptation_stats(self) -> Dict[str, Any]:
        """Get meta adaptation statistics."""
        stats = {
            "adaptation_steps": self.adaptation_steps,
            "stability_threshold": self.stability_threshold,
            "epochs_monitored": len(self.adaptation_curves),
            "cross_tasks_tracked": len(self.cross_task_performance),
        }

        # Adaptation speed stats
        if self.adaptation_speeds:
            stats.update(
                {
                    "adaptation_speed_mean": float(np.mean(self.adaptation_speeds)),
                    "adaptation_speed_std": float(np.std(self.adaptation_speeds)),
                    "adaptation_speed_latest": self.adaptation_speeds[-1],
                }
            )

        # Convergence stats
        if self.convergence_steps:
            stats.update(
                {
                    "avg_convergence_steps": float(np.mean(self.convergence_steps)),
                    "min_convergence_steps": int(np.min(self.convergence_steps)),
                    "max_convergence_steps": int(np.max(self.convergence_steps)),
                }
            )

        # Stability stats
        if self.adaptation_stability:
            stats.update(
                {
                    "adaptation_stability_mean": float(
                        np.mean(self.adaptation_stability)
                    ),
                    "adaptation_stability_latest": self.adaptation_stability[-1],
                }
            )

        # Generalization stats
        if self.adaptation_generalization:
            stats.update(
                {
                    "adaptation_generalization_mean": float(
                        np.mean(self.adaptation_generalization)
                    ),
                    "adaptation_generalization_latest": self.adaptation_generalization[
                        -1
                    ],
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
    ) -> None:
        """Called at the end of each batch."""
        pass


# Factory functions for easy instantiation
def create_maml(**kwargs) -> MAMLCallback:
    """Create MAML callback with default settings."""
    defaults = {"compute_frequency": 1, "num_inner_steps": 5, "adaptation_lr": 0.01}
    defaults.update(kwargs)
    return MAMLCallback(**defaults)


def create_few_shot(n_way: int = 5, k_shot: int = 1, **kwargs) -> FewShotCallback:
    """Create few-shot learning callback with default settings."""
    defaults = {"compute_frequency": 1}
    defaults.update(kwargs)
    return FewShotCallback(n_way, k_shot, **defaults)


def create_meta_adaptation(**kwargs) -> MetaAdaptationCallback:
    """Create meta adaptation callback with default settings."""
    defaults = {
        "compute_frequency": 1,
        "adaptation_steps": 10,
        "stability_threshold": 0.01,
    }
    defaults.update(kwargs)
    return MetaAdaptationCallback(**defaults)
