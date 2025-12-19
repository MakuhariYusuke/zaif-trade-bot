#!/usr/bin/env python3
"""
Multi-Task Learning Callbacks.

This module provides callbacks optimized for multi-task learning
including task balancing, shared representation monitoring, and
task interference assessment.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    MemoryOptimizedCallback,
)


class TaskBalancingCallback(MemoryOptimizedCallback):
    """
    Task balancing monitoring callback.

    Monitors the balance between different tasks in multi-task learning,
    including loss weighting, gradient magnitudes, and task performance.
    """

    def __init__(
        self,
        task_names: List[str],
        compute_frequency: int = 1,
        balance_threshold: float = 0.1,
    ):
        super().__init__(cache_size=1000)
        self.task_names = task_names
        self.compute_frequency = compute_frequency
        self.balance_threshold = balance_threshold

        # Task loss tracking
        self.task_losses: Dict[str, List[float]] = {task: [] for task in task_names}
        self.task_weights: Dict[str, List[float]] = {task: [] for task in task_names}

        # Balance metrics
        self.task_loss_ratios: Dict[str, List[float]] = {}
        self.balance_scores: List[float] = []
        self.imbalance_warnings: List[str] = []

        # Gradient tracking
        self.task_gradients: Dict[str, List[float]] = {task: [] for task in task_names}

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor task balancing."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track individual task losses
            total_loss = 0.0
            for task in self.task_names:
                loss_key = f"{task}_loss"
                if loss_key in logs:
                    loss = float(logs[loss_key])
                    self.task_losses[task].append(loss)
                    total_loss += loss

            # Track task weights
            for task in self.task_names:
                weight_key = f"{task}_weight"
                if weight_key in logs:
                    weight = float(logs[weight_key])
                    self.task_weights[task].append(weight)

            # Track task gradients
            for task in self.task_names:
                grad_key = f"{task}_grad_norm"
                if grad_key in logs:
                    grad_norm = float(logs[grad_key])
                    self.task_gradients[task].append(grad_norm)

            # Compute balance metrics
            if total_loss > 0:
                self._compute_balance_metrics(total_loss)

            # Check for imbalance
            self._check_task_imbalance()

            # Cache balancing metrics
            metrics_key = f"task_balancing_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "task_names": self.task_names,
                "balance_threshold": self.balance_threshold,
            }

            # Add current task metrics
            for task in self.task_names:
                if self.task_losses[task]:
                    metrics_data[f"{task}_loss"] = self.task_losses[task][-1]
                if self.task_weights[task]:
                    metrics_data[f"{task}_weight"] = self.task_weights[task][-1]
                if self.task_gradients[task]:
                    metrics_data[f"{task}_grad_norm"] = self.task_gradients[task][-1]

            if self.balance_scores:
                metrics_data["balance_score"] = self.balance_scores[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Task balancing metrics updated for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to monitor task balancing: {e}")

    def _compute_balance_metrics(self, total_loss: float) -> None:
        """Compute task balance metrics."""
        if not all(self.task_losses[task] for task in self.task_names):
            return

        # Compute loss ratios relative to total
        current_losses = {task: self.task_losses[task][-1] for task in self.task_names}

        for task in self.task_names:
            ratio = current_losses[task] / total_loss
            if task not in self.task_loss_ratios:
                self.task_loss_ratios[task] = []
            self.task_loss_ratios[task].append(ratio)

        # Compute balance score (lower is more balanced)
        ratios = list(current_losses.values())
        if len(ratios) > 1:
            mean_ratio = np.mean(ratios)
            balance_score = np.std(ratios) / (
                mean_ratio + 1e-8
            )  # Coefficient of variation
            self.balance_scores.append(float(balance_score))

    def _check_task_imbalance(self) -> None:
        """Check for task imbalance and generate warnings."""
        if not self.balance_scores:
            return

        current_balance = self.balance_scores[-1]

        if current_balance > self.balance_threshold:
            warning = (
                f"Task imbalance detected at epoch {len(self.balance_scores)}: "
                f"balance score {current_balance:.4f} > threshold {self.balance_threshold}"
            )

            # Identify problematic tasks
            if self.task_loss_ratios:
                ratios = {
                    task: self.task_loss_ratios[task][-1]
                    for task in self.task_names
                    if self.task_loss_ratios[task]
                }
                max_task = max(ratios, key=ratios.get)
                min_task = min(ratios, key=ratios.get)

                warning += (
                    f" - {max_task} dominates ({ratios[max_task]:.3f}) vs "
                    f"{min_task} ({ratios[min_task]:.3f})"
                )

            self.imbalance_warnings.append(warning)
            self.logger.warning(warning)

    def get_task_balancing_stats(self) -> Dict[str, Any]:
        """Get task balancing statistics."""
        stats = {
            "task_names": self.task_names,
            "balance_threshold": self.balance_threshold,
            "imbalance_warnings_count": len(self.imbalance_warnings),
            "epochs_monitored": len(self.balance_scores),
        }

        # Task loss stats
        for task in self.task_names:
            if self.task_losses[task]:
                stats.update(
                    {
                        f"{task}_loss_mean": float(np.mean(self.task_losses[task])),
                        f"{task}_loss_latest": self.task_losses[task][-1],
                        f"{task}_loss_std": float(np.std(self.task_losses[task])),
                    }
                )

        # Task weight stats
        for task in self.task_names:
            if self.task_weights[task]:
                stats.update(
                    {
                        f"{task}_weight_mean": float(np.mean(self.task_weights[task])),
                        f"{task}_weight_latest": self.task_weights[task][-1],
                    }
                )

        # Balance stats
        if self.balance_scores:
            stats.update(
                {
                    "balance_score_mean": float(np.mean(self.balance_scores)),
                    "balance_score_latest": self.balance_scores[-1],
                    "balance_score_std": float(np.std(self.balance_scores)),
                    "imbalance_epochs": sum(
                        1 for s in self.balance_scores if s > self.balance_threshold
                    ),
                }
            )

        return stats


    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of training."""
        pass
        pass

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        """Called at the end of each batch."""
        pass


class SharedRepresentationCallback(MemoryOptimizedCallback):
    """

    def __init__(
        self,
        compute_frequency: int = 1,
        representation_layers: Optional[List[str]] = None,
    ):
        self.representation_diversity: Dict[str, List[float]] = {}
        self.representation_stability: Dict[str, List[float]] = {}
        self.task_alignment_scores: Dict[str, List[float]] = {}

        # Layer-wise monitoring
        self.layer_activations: Dict[str, List[np.ndarray]] = {}
        self.layer_gradients: Dict[str, List[float]] = {}

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor shared representations."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track layer activations
            for layer in self.representation_layers:
                activation_key = f"{layer}_activations"
                if activation_key in logs:
                    activations = logs[activation_key]
                    if layer not in self.layer_activations:
                        self.layer_activations[layer] = []
                    self.layer_activations[layer].append(activations.copy())

            # Track layer gradients
            for layer in self.representation_layers:
                grad_key = f"{layer}_grad_norm"
                if grad_key in logs:
                    grad_norm = float(logs[grad_key])
                    if layer not in self.layer_gradients:
                        self.layer_gradients[layer] = []
                    self.layer_gradients[layer].append(grad_norm)

            # Compute representation quality metrics
            self._compute_representation_metrics()

            # Cache representation metrics
            metrics_key = f"shared_representation_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "representation_layers": self.representation_layers,
            }

            # Add current metrics
            for layer in self.representation_layers:
                if (
                    layer in self.representation_diversity
                    and self.representation_diversity[layer]
                ):
                    metrics_data[f"{layer}_diversity"] = self.representation_diversity[
                        layer
                    ][-1]
                if (
                    layer in self.representation_stability
                    and self.representation_stability[layer]
                ):
                    metrics_data[f"{layer}_stability"] = self.representation_stability[
                        layer
                    ][-1]
                if layer in self.layer_gradients and self.layer_gradients[layer]:
                    metrics_data[f"{layer}_grad_norm"] = self.layer_gradients[layer][-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Shared representation metrics updated for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to monitor shared representations: {e}")

    def _compute_representation_metrics(self) -> None:
        """Compute representation quality metrics."""
        for layer in self.representation_layers:
            if layer not in self.layer_activations or len(self.layer_activations[layer]) < 1:
                continue

            current_activations = self.layer_activations[layer][-1]

            # Compute diversity (variance across features) even with a single epoch
            try:
                if current_activations.ndim >= 2:
                    feature_variance = np.var(current_activations, axis=0)
                    diversity = float(np.mean(feature_variance))

                    if layer not in self.representation_diversity:
                        self.representation_diversity[layer] = []
                    self.representation_diversity[layer].append(diversity)
            except Exception:
                # Keep computation best-effort in tests
                pass

            # Compute stability (change from previous epoch) only if previous exists
            if len(self.layer_activations[layer]) >= 2:
                previous_activations = self.layer_activations[layer][-2]
                if current_activations.shape == previous_activations.shape:
                    activation_change = np.mean(
                        np.abs(current_activations - previous_activations)
                    )
                    stability = 1.0 / (1.0 + activation_change)  # Convert to stability score

                    if layer not in self.representation_stability:
                        self.representation_stability[layer] = []
                    self.representation_stability[layer].append(stability)

    def get_shared_representation_stats(self) -> Dict[str, Any]:
        """Get shared representation statistics."""
        stats = {
            "representation_layers": self.representation_layers,
            "epochs_monitored": max(
                len(acts) for acts in self.layer_activations.values()
            )
            if self.layer_activations
            else 0,
        }

        # Layer diversity stats
        for layer in self.representation_layers:
            if (
                layer in self.representation_diversity
                and self.representation_diversity[layer]
            ):
                stats.update(
                    {
                        f"{layer}_diversity_mean": float(
                            np.mean(self.representation_diversity[layer])
                        ),
                        f"{layer}_diversity_latest": self.representation_diversity[
                            layer
                        ][-1],
                    }
                )

            if (
                layer in self.representation_stability
                and self.representation_stability[layer]
            ):
                stats.update(
                    {
                        f"{layer}_stability_mean": float(
                            np.mean(self.representation_stability[layer])
                        ),
                        f"{layer}_stability_latest": self.representation_stability[
                            layer
                        ][-1],
                    }
                )

            if layer in self.layer_gradients and self.layer_gradients[layer]:
                stats.update(
                    {
                        f"{layer}_grad_norm_mean": float(
                            np.mean(self.layer_gradients[layer])
                        ),
                        f"{layer}_grad_norm_latest": self.layer_gradients[layer][-1],
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
    ) -> None:
        """Called at the end of each batch."""
        pass


class TaskInterferenceCallback(MemoryOptimizedCallback):
    """
    Task interference monitoring callback.

    Monitors interference between tasks in multi-task learning,
    including negative transfer and task conflict detection.
        compute_frequency: int = 1,
        interference_threshold: float = -0.05,
    ):
        super().__init__(cache_size=1000)
        self.task_names = task_names
        self.compute_frequency = compute_frequency
        self.interference_threshold = interference_threshold

        # Task performance tracking
        self.task_performance_history: Dict[str, List[float]] = {
            task: [] for task in task_names
        self.negative_transfer_detected: bool = False

        # Task correlation tracking
        self.task_performance_correlations: Dict[Tuple[str, str], List[float]] = {}

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor task interference."""

        try:
            # Track task performances
            current_performances = {}
            for task in self.task_names:
                perf_key = f"{task}_performance"
                if perf_key in logs:
                    performance = float(logs[perf_key])
                    self.task_performance_history[task].append(performance)
                    current_performances[task] = performance

            # Compute interference scores
            if len(current_performances) == len(self.task_names):
                self._compute_interference_scores(current_performances)

            # Detect negative transfer
            self._detect_negative_transfer()

            # Compute task correlations
            self._compute_task_correlations()

            # Cache interference metrics
            metrics_key = f"task_interference_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "task_names": self.task_names,
                "interference_threshold": self.interference_threshold,
                "negative_transfer_detected": self.negative_transfer_detected,
            }

            # Add current performance and interference metrics
            for task in self.task_names:
                if task in current_performances:
                    metrics_data[f"{task}_performance"] = current_performances[task]
                if (
                    task in self.task_interference_scores
                    and self.task_interference_scores[task]
                ):
                    metrics_data[
                        f"{task}_interference"
                    ] = self.task_interference_scores[task][-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Task interference metrics updated for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to monitor task interference: {e}")

    def _compute_interference_scores(
        self, current_performances: Dict[str, float]
    ) -> None:
        """Compute task interference scores."""
        for task in self.task_names:
            # Compute performance change from single-task baseline (if available)
            # For now, use change from initial performance as proxy
            if not self.task_performance_history.get(task):
                continue

            initial_perf = self.task_performance_history[task][0]
            current_perf = current_performances[task]

            if initial_perf > 0:
                interference_score = (current_perf - initial_perf) / initial_perf

                if task not in self.task_interference_scores:
                    self.task_interference_scores[task] = []
                self.task_interference_scores[task].append(interference_score)

                # Check for significant negative interference
                if interference_score < self.interference_threshold:
                    interference_event = {
                        "epoch": len(self.task_performance_history[task]),
                        "task": task,
                        "interference_score": interference_score,
                        "initial_performance": initial_perf,
                        "current_performance": current_perf,
                    }
                    self.interference_events.append(interference_event)
                    self.logger.warning(
                        f"Negative interference detected for {task}: "
                        f"{interference_score:.4f}"
                    )

    def _detect_negative_transfer(self) -> None:
        """Detect overall negative transfer across tasks."""
        if not self.task_interference_scores:
            return

        # Check if majority of tasks show negative interference
        current_interferences = []
        for task in self.task_names:
            if (
                task in self.task_interference_scores
                and self.task_interference_scores[task]
            ):
                current_interferences.append(self.task_interference_scores[task][-1])

        if current_interferences:
            negative_tasks = sum(1 for score in current_interferences if score < 0)
            negative_ratio = negative_tasks / len(current_interferences)

            if negative_ratio > 0.5 and not self.negative_transfer_detected:
                self.negative_transfer_detected = True
                self.logger.warning(
                    f"Negative transfer detected: {negative_ratio:.2f} of tasks "
                    f"showing performance degradation"
                )

    def _compute_task_correlations(self) -> None:
        """Compute correlations between task performances."""
        if not all(len(hist) >= 3 for hist in self.task_performance_history.values()):
            return

        # Compute pairwise correlations
        for i, task1 in enumerate(self.task_names):
            for j, task2 in enumerate(self.task_names):
                if i >= j:  # Avoid duplicate pairs
                    continue

                pair = (task1, task2)
                perf1 = self.task_performance_history[task1]
                perf2 = self.task_performance_history[task2]

                if len(perf1) == len(perf2):
                    try:
                        correlation = np.corrcoef(perf1, perf2)[0, 1]
                        if pair not in self.task_performance_correlations:
                            self.task_performance_correlations[pair] = []
                        self.task_performance_correlations[pair].append(correlation)
                    except Exception:
                        pass

    def get_task_interference_stats(self) -> Dict[str, Any]:
        """Get task interference statistics."""
        stats = {
            "task_names": self.task_names,
            "interference_threshold": self.interference_threshold,
            "negative_transfer_detected": self.negative_transfer_detected,
            "interference_events_count": len(self.interference_events),
            "epochs_monitored": max(
                len(hist) for hist in self.task_performance_history.values()
            )
            if self.task_performance_history
            else 0,
        }

        # Task interference stats
        for task in self.task_names:
            if (
                task in self.task_interference_scores
                and self.task_interference_scores[task]
            ):
                scores = self.task_interference_scores[task]
                stats.update(
                    {
                        f"{task}_interference_mean": float(np.mean(scores)),
                        f"{task}_interference_latest": scores[-1],
                        f"{task}_negative_interference_count": sum(
                            1 for s in scores if s < 0
                        ),
                    }
                )

        # Task correlation stats
        if self.task_performance_correlations:
            correlations = []
            for pair_corrs in self.task_performance_correlations.values():
                if pair_corrs:
                    correlations.extend(pair_corrs)

            if correlations:
                stats.update(
                    {
                        "avg_task_correlation": float(np.mean(correlations)),
                        "task_correlation_std": float(np.std(correlations)),
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


    return TaskBalancingCallback(task_names, **defaults)


def create_shared_representation(**kwargs) -> SharedRepresentationCallback:
    """Create shared representation callback with default settings."""
    defaults = {"compute_frequency": 1, "representation_layers": ["shared_encoder"]}
    defaults.update(kwargs)
    return SharedRepresentationCallback(**defaults)


def create_task_interference(
    task_names: List[str], **kwargs
) -> TaskInterferenceCallback:
    """Create task interference callback with default settings."""
    defaults = {"compute_frequency": 1, "interference_threshold": -0.05}
    defaults.update(kwargs)
    return TaskInterferenceCallback(task_names, **defaults)
