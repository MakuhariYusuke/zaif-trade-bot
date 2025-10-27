#!/usr/bin/env python3
"""
Transfer Learning Callbacks.

This module provides callbacks optimized for transfer learning
tasks including domain adaptation and fine-tuning.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    MemoryOptimizedCallback,
)


class DomainAdaptationCallback(MemoryOptimizedCallback):
    """
    Domain adaptation monitoring callback.

    Monitors the effectiveness of domain adaptation techniques
    by tracking domain shift, feature alignment, and adaptation progress.
    """

    def __init__(self, compute_frequency: int = 1, adaptation_method: str = "auto"):
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency
        self.adaptation_method = adaptation_method

        # Domain shift tracking
        self.source_domain_stats: Dict[str, List[float]] = {}
        self.target_domain_stats: Dict[str, List[float]] = {}
        self.domain_shift_scores: Dict[str, List[float]] = {}

        # Feature alignment tracking
        self.feature_alignment_scores: List[float] = []
        self.classifier_alignment_scores: List[float] = []

        # Adaptation progress
        self.adaptation_losses: List[float] = []
        self.discriminator_accuracies: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor domain adaptation progress."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track domain statistics
            if "source_features" in logs and "target_features" in logs:
                source_features = logs["source_features"]
                target_features = logs["target_features"]

                self._track_domain_statistics(source_features, target_features)

            # Track feature alignment
            if "feature_alignment_loss" in logs:
                alignment_score = 1.0 / (
                    1.0 + logs["feature_alignment_loss"]
                )  # Convert loss to score
                self.feature_alignment_scores.append(alignment_score)

            # Track classifier alignment (for methods like DANN)
            if "classifier_alignment_loss" in logs:
                classifier_score = 1.0 / (1.0 + logs["classifier_alignment_loss"])
                self.classifier_alignment_scores.append(classifier_score)

            # Track adaptation-specific losses
            if "domain_adaptation_loss" in logs:
                self.adaptation_losses.append(logs["domain_adaptation_loss"])

            # Track discriminator accuracy (for adversarial methods)
            if "discriminator_accuracy" in logs:
                self.discriminator_accuracies.append(logs["discriminator_accuracy"])

            # Cache adaptation metrics
            metrics_key = f"domain_adaptation_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "adaptation_method": self.adaptation_method,
            }

            if self.feature_alignment_scores:
                metrics_data["feature_alignment_score"] = self.feature_alignment_scores[
                    -1
                ]
            if self.classifier_alignment_scores:
                metrics_data[
                    "classifier_alignment_score"
                ] = self.classifier_alignment_scores[-1]
            if self.adaptation_losses:
                metrics_data["adaptation_loss"] = self.adaptation_losses[-1]
            if self.discriminator_accuracies:
                metrics_data["discriminator_accuracy"] = self.discriminator_accuracies[
                    -1
                ]

            # Add domain shift metrics
            domain_shift_summary = self._get_domain_shift_summary()
            metrics_data.update(domain_shift_summary)

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Domain adaptation metrics updated for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to monitor domain adaptation: {e}")

    def _track_domain_statistics(
        self, source_features: np.ndarray, target_features: np.ndarray
    ) -> None:
        """Track statistical properties of source and target domains."""
        try:
            # Compute basic statistics for each feature dimension
            source_mean = np.mean(source_features, axis=0)
            source_std = np.std(source_features, axis=0)
            target_mean = np.mean(target_features, axis=0)
            target_std = np.std(target_features, axis=0)

            # Track means and stds
            for i in range(len(source_mean)):
                mean_key = f"feature_{i}_mean"
                std_key = f"feature_{i}_std"

                # Initialize if needed
                if mean_key not in self.source_domain_stats:
                    self.source_domain_stats[mean_key] = []
                    self.target_domain_stats[mean_key] = []
                    self.domain_shift_scores[mean_key] = []

                if std_key not in self.source_domain_stats:
                    self.source_domain_stats[std_key] = []
                    self.target_domain_stats[std_key] = []
                    self.domain_shift_scores[std_key] = []

                # Store current values
                self.source_domain_stats[mean_key].append(float(source_mean[i]))
                self.target_domain_stats[mean_key].append(float(target_mean[i]))
                self.source_domain_stats[std_key].append(float(source_std[i]))
                self.target_domain_stats[std_key].append(float(target_std[i]))

                # Compute domain shift (difference in means/stds)
                mean_shift = abs(source_mean[i] - target_mean[i])
                std_shift = abs(source_std[i] - target_std[i])

                self.domain_shift_scores[mean_key].append(float(mean_shift))
                self.domain_shift_scores[std_key].append(float(std_shift))

        except Exception as e:
            self.logger.warning(f"Failed to track domain statistics: {e}")

    def _get_domain_shift_summary(self) -> Dict[str, float]:
        """Get summary of domain shift metrics."""
        summary = {}

        if self.domain_shift_scores:
            # Average domain shift across all features
            all_shifts = []
            for shifts in self.domain_shift_scores.values():
                if shifts:
                    all_shifts.extend(shifts)

            if all_shifts:
                summary["avg_domain_shift"] = float(np.mean(all_shifts))
                summary["max_domain_shift"] = float(np.max(all_shifts))
                summary["domain_shift_std"] = float(np.std(all_shifts))

        return summary

    def get_domain_adaptation_stats(self) -> Dict[str, Any]:
        """Get domain adaptation statistics."""
        stats = {
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

        # Add domain shift summary
        stats.update(self._get_domain_shift_summary())

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


class FineTuningCallback(MemoryOptimizedCallback):
    """
    Fine-tuning monitoring callback.

    Monitors fine-tuning progress including catastrophic forgetting,
    layer-wise learning rates, and task-specific performance.
    """

    def __init__(
        self,
        compute_frequency: int = 1,
        freeze_layers: Optional[List[str]] = None,
        monitor_catastrophic_forgetting: bool = True,
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency
        self.freeze_layers = freeze_layers or []
        self.monitor_catastrophic_forgetting = monitor_catastrophic_forgetting

        # Fine-tuning metrics
        self.layer_learning_rates: Dict[str, List[float]] = {}
        self.layer_gradients: Dict[str, List[float]] = {}
        self.task_performance: Dict[str, List[float]] = {}

        # Catastrophic forgetting tracking
        self.source_task_performance: Dict[str, List[float]] = {}
        self.forgetting_scores: Dict[str, List[float]] = {}

        # Layer-wise monitoring
        self.layer_weights_change: Dict[str, List[float]] = {}

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor fine-tuning progress."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track layer-wise learning rates
            if "layer_learning_rates" in logs:
                lr_dict = logs["layer_learning_rates"]
                for layer_name, lr in lr_dict.items():
                    if layer_name not in self.layer_learning_rates:
                        self.layer_learning_rates[layer_name] = []
                    self.layer_learning_rates[layer_name].append(float(lr))

            # Track layer gradients
            if "layer_gradients" in logs:
                grad_dict = logs["layer_gradients"]
                for layer_name, grad_norm in grad_dict.items():
                    if layer_name not in self.layer_gradients:
                        self.layer_gradients[layer_name] = []
                    self.layer_gradients[layer_name].append(float(grad_norm))

            # Track task-specific performance
            if "task_performance" in logs:
                perf_dict = logs["task_performance"]
                for task_name, performance in perf_dict.items():
                    if task_name not in self.task_performance:
                        self.task_performance[task_name] = []
                    self.task_performance[task_name].append(float(performance))

            # Monitor catastrophic forgetting
            if self.monitor_catastrophic_forgetting:
                self._monitor_catastrophic_forgetting(logs)

            # Track layer weight changes
            if "layer_weights" in logs:
                weights_dict = logs["layer_weights"]
                self._track_weight_changes(weights_dict)

            # Cache fine-tuning metrics
            metrics_key = f"fine_tuning_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "frozen_layers": self.freeze_layers,
                "monitor_catastrophic_forgetting": self.monitor_catastrophic_forgetting,
            }

            # Add current metrics
            if self.task_performance:
                for task, perf in self.task_performance.items():
                    if perf:
                        metrics_data[f"{task}_performance"] = perf[-1]

            if self.forgetting_scores:
                for task, scores in self.forgetting_scores.items():
                    if scores:
                        metrics_data[f"{task}_forgetting"] = scores[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(f"Fine-tuning metrics updated for epoch {context.epoch}")

        except Exception as e:
            self.logger.error(f"Failed to monitor fine-tuning: {e}")

    def _monitor_catastrophic_forgetting(self, logs: Dict[str, Any]) -> None:
        """Monitor catastrophic forgetting by tracking source task performance."""
        if "source_task_performance" in logs:
            source_perf = logs["source_task_performance"]

            for task_name, performance in source_perf.items():
                if task_name not in self.source_task_performance:
                    self.source_task_performance[task_name] = []

                current_perf = float(performance)
                self.source_task_performance[task_name].append(current_perf)

                # Compute forgetting score (drop from initial performance)
                if len(self.source_task_performance[task_name]) >= 2:
                    initial_perf = self.source_task_performance[task_name][0]
                    forgetting = initial_perf - current_perf

                    if task_name not in self.forgetting_scores:
                        self.forgetting_scores[task_name] = []
                    self.forgetting_scores[task_name].append(forgetting)

    def _track_weight_changes(self, current_weights: Dict[str, np.ndarray]) -> None:
        """Track changes in layer weights."""
        for layer_name, weights in current_weights.items():
            if layer_name not in self.layer_weights_change:
                self.layer_weights_change[layer_name] = []

            # For first epoch, just store the weights for comparison
            if not hasattr(self, f"_prev_weights_{layer_name}"):
                setattr(self, f"_prev_weights_{layer_name}", weights.copy())
                self.layer_weights_change[layer_name].append(0.0)
            else:
                prev_weights = getattr(self, f"_prev_weights_{layer_name}")
                if weights.shape == prev_weights.shape:
                    weight_change = np.mean(np.abs(weights - prev_weights))
                    self.layer_weights_change[layer_name].append(float(weight_change))

                # Update previous weights
                setattr(self, f"_prev_weights_{layer_name}", weights.copy())

    def get_fine_tuning_stats(self) -> Dict[str, Any]:
        """Get fine-tuning statistics."""
        stats = {
            "frozen_layers": self.freeze_layers,
            "monitored_layers": len(self.layer_learning_rates),
            "monitored_tasks": len(self.task_performance),
            "catastrophic_forgetting_monitored": self.monitor_catastrophic_forgetting,
        }

        # Layer learning rate stats
        if self.layer_learning_rates:
            for layer, lrs in self.layer_learning_rates.items():
                if lrs:
                    stats.update(
                        {
                            f"{layer}_lr_mean": float(np.mean(lrs)),
                            f"{layer}_lr_latest": lrs[-1],
                        }
                    )

        # Task performance stats
        if self.task_performance:
            for task, perf in self.task_performance.items():
                if perf:
                    stats.update(
                        {
                            f"{task}_perf_mean": float(np.mean(perf)),
                            f"{task}_perf_latest": perf[-1],
                            f"{task}_perf_improvement": perf[-1] - perf[0]
                            if len(perf) > 1
                            else 0,
                        }
                    )

        # Catastrophic forgetting stats
        if self.forgetting_scores:
            for task, scores in self.forgetting_scores.items():
                if scores:
                    stats.update(
                        {
                            f"{task}_forgetting_mean": float(np.mean(scores)),
                            f"{task}_forgetting_latest": scores[-1],
                            f"{task}_max_forgetting": float(np.max(scores)),
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


class TransferPerformanceCallback(MemoryOptimizedCallback):
    """
    Transfer performance monitoring callback.

    Monitors the effectiveness of transfer learning by comparing
    performance on source vs target domains and tracking generalization.
    """

    def __init__(
        self, compute_frequency: int = 1, evaluation_metrics: Optional[List[str]] = None
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency
        self.evaluation_metrics = evaluation_metrics or ["accuracy", "f1"]

        # Performance tracking
        self.source_performance: Dict[str, List[float]] = {}
        self.target_performance: Dict[str, List[float]] = {}
        self.transfer_gap: Dict[str, List[float]] = {}

        # Generalization metrics
        self.generalization_scores: List[float] = []
        self.overfitting_indicators: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor transfer performance."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None:
            return

        try:
            # Track source domain performance
            if "source_predictions" in logs and "source_labels" in logs:
                source_perf = self._compute_performance_metrics(
                    logs["source_predictions"], logs["source_labels"]
                )
                self._update_performance_history(self.source_performance, source_perf)

            # Track target domain performance
            if "target_predictions" in logs and "target_labels" in logs:
                target_perf = self._compute_performance_metrics(
                    logs["target_predictions"], logs["target_labels"]
                )
                self._update_performance_history(self.target_performance, target_perf)

            # Compute transfer gap
            self._compute_transfer_gap()

            # Compute generalization score
            if "validation_predictions" in logs and "validation_labels" in logs:
                val_perf = self._compute_performance_metrics(
                    logs["validation_predictions"], logs["validation_labels"]
                )
                gen_score = self._compute_generalization_score(val_perf)
                self.generalization_scores.append(gen_score)

            # Compute overfitting indicators
            if self.source_performance and self.target_performance:
                overfitting = self._compute_overfitting_indicator()
                self.overfitting_indicators.append(overfitting)

            # Cache transfer performance metrics
            metrics_key = f"transfer_performance_epoch_{context.epoch}"
            metrics_data = {
                "epoch": context.epoch,
                "evaluation_metrics": self.evaluation_metrics,
            }

            # Add current performance metrics
            for metric in self.evaluation_metrics:
                if (
                    metric in self.source_performance
                    and self.source_performance[metric]
                ):
                    metrics_data[f"source_{metric}"] = self.source_performance[metric][
                        -1
                    ]
                if (
                    metric in self.target_performance
                    and self.target_performance[metric]
                ):
                    metrics_data[f"target_{metric}"] = self.target_performance[metric][
                        -1
                    ]
                if metric in self.transfer_gap and self.transfer_gap[metric]:
                    metrics_data[f"transfer_gap_{metric}"] = self.transfer_gap[metric][
                        -1
                    ]

            if self.generalization_scores:
                metrics_data["generalization_score"] = self.generalization_scores[-1]
            if self.overfitting_indicators:
                metrics_data["overfitting_indicator"] = self.overfitting_indicators[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Transfer performance metrics updated for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(f"Failed to monitor transfer performance: {e}")

    def _compute_performance_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        metrics = {}

        try:
            if "accuracy" in self.evaluation_metrics:
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    # Multi-class classification
                    pred_classes = np.argmax(predictions, axis=1)
                else:
                    # Binary classification
                    pred_classes = (predictions > 0.5).astype(int).flatten()

                metrics["accuracy"] = accuracy_score(labels, pred_classes)

            if "f1" in self.evaluation_metrics:
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    pred_classes = np.argmax(predictions, axis=1)
                    metrics["f1"] = f1_score(labels, pred_classes, average="weighted")
                else:
                    pred_classes = (predictions > 0.5).astype(int).flatten()
                    metrics["f1"] = f1_score(labels, pred_classes)

        except Exception as e:
            self.logger.warning(f"Failed to compute performance metrics: {e}")

        return metrics

    def _update_performance_history(
        self, history: Dict[str, List[float]], current_perf: Dict[str, float]
    ) -> None:
        """Update performance history."""
        for metric, value in current_perf.items():
            if metric not in history:
                history[metric] = []
            history[metric].append(value)

    def _compute_transfer_gap(self) -> None:
        """Compute the gap between source and target performance."""
        for metric in self.evaluation_metrics:
            if (
                metric in self.source_performance
                and metric in self.target_performance
                and self.source_performance[metric]
                and self.target_performance[metric]
            ):
                if metric not in self.transfer_gap:
                    self.transfer_gap[metric] = []

                source_perf = self.source_performance[metric][-1]
                target_perf = self.target_performance[metric][-1]
                gap = source_perf - target_perf  # Positive = source better than target

                self.transfer_gap[metric].append(gap)

    def _compute_generalization_score(self, validation_perf: Dict[str, float]) -> float:
        """Compute generalization score based on validation performance."""
        # Simple generalization score: average of available metrics
        scores = []
        for metric in self.evaluation_metrics:
            if metric in validation_perf:
                scores.append(validation_perf[metric])

        return float(np.mean(scores)) if scores else 0.0

    def _compute_overfitting_indicator(self) -> float:
        """Compute overfitting indicator based on source vs target performance gap."""
        gaps = []
        for metric in self.evaluation_metrics:
            if (
                metric in self.source_performance
                and metric in self.target_performance
                and len(self.source_performance[metric]) > 1
                and len(self.target_performance[metric]) > 1
            ):
                # Recent vs initial performance gap
                source_recent = self.source_performance[metric][-1]
                source_initial = self.source_performance[metric][0]
                target_recent = self.target_performance[metric][-1]

                if source_initial > 0:
                    overfitting = (source_recent - target_recent) / source_initial
                    gaps.append(overfitting)

        return float(np.mean(gaps)) if gaps else 0.0

    def get_transfer_performance_stats(self) -> Dict[str, Any]:
        """Get transfer performance statistics."""
        stats = {
            "evaluation_metrics": self.evaluation_metrics,
            "epochs_monitored": len(self.generalization_scores),
        }

        # Performance stats
        for metric in self.evaluation_metrics:
            if metric in self.source_performance and self.source_performance[metric]:
                stats.update(
                    {
                        f"source_{metric}_mean": float(
                            np.mean(self.source_performance[metric])
                        ),
                        f"source_{metric}_latest": self.source_performance[metric][-1],
                    }
                )

            if metric in self.target_performance and self.target_performance[metric]:
                stats.update(
                    {
                        f"target_{metric}_mean": float(
                            np.mean(self.target_performance[metric])
                        ),
                        f"target_{metric}_latest": self.target_performance[metric][-1],
                    }
                )

            if metric in self.transfer_gap and self.transfer_gap[metric]:
                stats.update(
                    {
                        f"transfer_gap_{metric}_mean": float(
                            np.mean(self.transfer_gap[metric])
                        ),
                        f"transfer_gap_{metric}_latest": self.transfer_gap[metric][-1],
                    }
                )

        # Generalization stats
        if self.generalization_scores:
            stats.update(
                {
                    "generalization_score_mean": float(
                        np.mean(self.generalization_scores)
                    ),
                    "generalization_score_latest": self.generalization_scores[-1],
                }
            )

        if self.overfitting_indicators:
            stats.update(
                {
                    "overfitting_indicator_mean": float(
                        np.mean(self.overfitting_indicators)
                    ),
                    "overfitting_indicator_latest": self.overfitting_indicators[-1],
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
def create_domain_adaptation(**kwargs) -> DomainAdaptationCallback:
    """Create domain adaptation callback with default settings."""
    defaults = {"compute_frequency": 1, "adaptation_method": "auto"}
    defaults.update(kwargs)
    return DomainAdaptationCallback(**defaults)


def create_fine_tuning(**kwargs) -> FineTuningCallback:
    """Create fine-tuning callback with default settings."""
    defaults = {"compute_frequency": 1, "monitor_catastrophic_forgetting": True}
    defaults.update(kwargs)
    return FineTuningCallback(**defaults)


def create_transfer_performance(**kwargs) -> TransferPerformanceCallback:
    """Create transfer performance callback with default settings."""
    defaults = {"compute_frequency": 1, "evaluation_metrics": ["accuracy", "f1"]}
    defaults.update(kwargs)
    return TransferPerformanceCallback(**defaults)
