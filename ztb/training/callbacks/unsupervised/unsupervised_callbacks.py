#!/usr/bin/env python3
"""
Unsupervised Learning Callbacks.

Callbacks for unsupervised workflows including clustering quality,
dimensionality-reduction diagnostics, embedding quality, and convergence.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    NoOpMemoryOptimizedCallback,
)
from ztb.training.callbacks.shared.utils.value_utils import (
    append_bounded as _append_bounded_value,
    as_optional_array as _to_array,
    as_optional_float as _as_float,
)
from ztb.types.common import ObjectMap

_HISTORY_LIMIT = 1_000

def _append_bounded(history: list[float], value: float, max_len: int = _HISTORY_LIMIT) -> None:
    _append_bounded_value(history, value, max_len)

class ClusteringMetricsCallback(NoOpMemoryOptimizedCallback):
    """Compute and track clustering quality metrics."""

    def __init__(self, compute_frequency: int = 1, max_samples: int = 5000):
        super().__init__(cache_size=500)
        self.compute_frequency = max(1, compute_frequency)
        self.max_samples = max(10, max_samples)

        self.silhouette_history: list[float] = []
        self.calinski_harabasz_history: list[float] = []
        self.davies_bouldin_history: list[float] = []

        self.cluster_centers_history: list[np.ndarray] = []
        self.cluster_stability_scores: list[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        embeddings = _to_array(logs.get("embeddings"))
        cluster_labels = _to_array(logs.get("cluster_labels"))
        if embeddings is None or cluster_labels is None:
            return

        if embeddings.shape[0] != cluster_labels.shape[0]:
            self.logger.warning("Embedding/label length mismatch; skipping clustering metrics")
            return

        if embeddings.shape[0] > self.max_samples:
            indices = np.random.choice(embeddings.shape[0], self.max_samples, replace=False)
            embeddings = embeddings[indices]
            cluster_labels = cluster_labels[indices]

        unique_labels = np.unique(cluster_labels)
        if unique_labels.size <= 1:
            self.logger.warning("Only one cluster found; skipping clustering metrics")
            return

        try:
            silhouette = float(silhouette_score(embeddings, cluster_labels))
            calinski_harabasz = float(
                calinski_harabasz_score(embeddings, cluster_labels)
            )
            davies_bouldin = float(davies_bouldin_score(embeddings, cluster_labels))

            _append_bounded(self.silhouette_history, silhouette)
            _append_bounded(self.calinski_harabasz_history, calinski_harabasz)
            _append_bounded(self.davies_bouldin_history, davies_bouldin)

            centers = _to_array(logs.get("cluster_centers"))
            if centers is not None:
                self.cluster_centers_history.append(centers.copy())
                if len(self.cluster_centers_history) > _HISTORY_LIMIT:
                    del self.cluster_centers_history[
                        : len(self.cluster_centers_history) - _HISTORY_LIMIT
                    ]
                self._compute_cluster_stability()

            self.cache_metrics(
                f"clustering_epoch_{context.epoch}",
                {
                    "silhouette_score": silhouette,
                    "calinski_harabasz_score": calinski_harabasz,
                    "davies_bouldin_score": davies_bouldin,
                    "num_clusters": int(unique_labels.size),
                    "epoch": context.epoch,
                },
            )

            self.logger.debug(
                "Clustering metrics at epoch %s: silhouette=%.4f, CH=%.2f",
                context.epoch,
                silhouette,
                calinski_harabasz,
            )
        except Exception as exc:
            self.logger.error("Failed to compute clustering metrics: %s", exc)

    def _compute_cluster_stability(self) -> None:
        if len(self.cluster_centers_history) < 2:
            return

        current_centers = self.cluster_centers_history[-1]
        previous_centers = self.cluster_centers_history[-2]
        if current_centers.shape != previous_centers.shape:
            self.logger.warning("Cluster centers shape mismatch; stability not computed")
            return

        movement = float(
            np.mean(np.linalg.norm(current_centers - previous_centers, axis=1))
        )
        stability_score = 1.0 / (1.0 + movement)
        _append_bounded(self.cluster_stability_scores, stability_score)

    def get_clustering_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "epochs_computed": len(self.silhouette_history),
            "stability_scores_count": len(self.cluster_stability_scores),
        }

        if self.silhouette_history:
            stats.update(
                {
                    "silhouette_mean": float(np.mean(self.silhouette_history)),
                    "silhouette_std": float(np.std(self.silhouette_history)),
                    "silhouette_latest": self.silhouette_history[-1],
                    "calinski_harabasz_mean": float(
                        np.mean(self.calinski_harabasz_history)
                    ),
                    "davies_bouldin_mean": float(np.mean(self.davies_bouldin_history)),
                }
            )

        if self.cluster_stability_scores:
            stats.update(
                {
                    "cluster_stability_mean": float(
                        np.mean(self.cluster_stability_scores)
                    ),
                    "cluster_stability_latest": self.cluster_stability_scores[-1],
                }
            )

        return stats

class DimensionalityReductionMetricsCallback(NoOpMemoryOptimizedCallback):
    """Monitor dimensionality-reduction quality metrics."""

    def __init__(
        self, compute_frequency: int = 1, original_data: np.ndarray | None = None
    ) -> None:
        super().__init__(cache_size=500)
        self.compute_frequency = max(1, compute_frequency)
        self.original_data = None if original_data is None else np.asarray(original_data)

        self.explained_variance_history: list[float] = []
        self.reconstruction_error_history: list[float] = []
        self.trustworthiness_history: list[float] = []
        self.continuity_history: list[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        embeddings = _to_array(logs.get("embeddings"))
        if embeddings is None:
            return

        try:
            explained = logs.get("explained_variance")
            if explained is not None:
                explained_value = _as_float(explained)
                if explained_value is None:
                    explained_arr = _to_array(explained)
                    if explained_arr is not None:
                        explained_value = float(np.sum(explained_arr))
                if explained_value is not None:
                    _append_bounded(self.explained_variance_history, explained_value)

            orig_data = self.original_data
            original_from_logs = _to_array(logs.get("original_data"))
            if original_from_logs is not None:
                orig_data = original_from_logs

            reconstructed = _to_array(logs.get("reconstructed_data"))
            if orig_data is not None and reconstructed is not None:
                try:
                    reconstruction_error = float(
                        np.mean(np.square(orig_data - reconstructed))
                    )
                    _append_bounded(
                        self.reconstruction_error_history,
                        reconstruction_error,
                    )
                except Exception as exc:
                    self.logger.warning("Failed reconstruction error computation: %s", exc)

            if orig_data is not None:
                try:
                    trustworthiness, continuity = self._compute_neighborhood_preservation(
                        orig_data,
                        embeddings,
                    )
                    _append_bounded(self.trustworthiness_history, trustworthiness)
                    _append_bounded(self.continuity_history, continuity)
                except Exception as exc:
                    self.logger.warning(
                        "Failed neighborhood preservation computation: %s", exc
                    )

            metrics_data: ObjectMap = {
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 1,
                "num_samples": int(embeddings.shape[0]),
                "epoch": context.epoch,
            }
            if self.explained_variance_history:
                metrics_data["explained_variance"] = self.explained_variance_history[-1]
            if self.reconstruction_error_history:
                metrics_data["reconstruction_error"] = self.reconstruction_error_history[-1]
            if self.trustworthiness_history:
                metrics_data["trustworthiness"] = self.trustworthiness_history[-1]
                metrics_data["continuity"] = self.continuity_history[-1]

            self.cache_metrics(f"dim_reduction_epoch_{context.epoch}", metrics_data)

        except Exception as exc:
            self.logger.error("Failed dimensionality-reduction metrics: %s", exc)

    def _compute_neighborhood_preservation(
        self, original_data: np.ndarray, embeddings: np.ndarray, k: int = 10
    ) -> tuple[float, float]:
        orig_distances = squareform(pdist(original_data))
        emb_distances = squareform(pdist(embeddings))

        n_samples = len(original_data)
        if n_samples <= 1:
            return 0.0, 0.0

        k_eff = min(k, n_samples - 1)
        trustworthiness = 0.0
        continuity = 0.0

        for i in range(n_samples):
            orig_neighbors = np.argsort(orig_distances[i])[1 : k_eff + 1]
            emb_neighbors = np.argsort(emb_distances[i])[1 : k_eff + 1]

            false_negatives = len(set(orig_neighbors) - set(emb_neighbors))
            false_positives = len(set(emb_neighbors) - set(orig_neighbors))
            trustworthiness += false_negatives / k_eff
            continuity += false_positives / k_eff

        return 1.0 - (trustworthiness / n_samples), 1.0 - (continuity / n_samples)

    def get_dim_reduction_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "epochs_computed": len(self.explained_variance_history)
            or len(self.reconstruction_error_history)
        }

        if self.explained_variance_history:
            stats.update(
                {
                    "explained_variance_mean": float(
                        np.mean(self.explained_variance_history)
                    ),
                    "explained_variance_latest": self.explained_variance_history[-1],
                }
            )

        if self.reconstruction_error_history:
            stats.update(
                {
                    "reconstruction_error_mean": float(
                        np.mean(self.reconstruction_error_history)
                    ),
                    "reconstruction_error_std": float(
                        np.std(self.reconstruction_error_history)
                    ),
                    "reconstruction_error_latest": self.reconstruction_error_history[-1],
                }
            )

        if self.trustworthiness_history:
            stats.update(
                {
                    "trustworthiness_mean": float(np.mean(self.trustworthiness_history)),
                    "trustworthiness_latest": self.trustworthiness_history[-1],
                    "continuity_mean": float(np.mean(self.continuity_history)),
                    "continuity_latest": self.continuity_history[-1],
                }
            )

        return stats

class EmbeddingQualityCallback(NoOpMemoryOptimizedCallback):
    """Assess learned embedding quality via multiple downstream signals."""

    def __init__(
        self,
        compute_frequency: int = 1,
        assessment_tasks: list[str] | None = None,
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = max(1, compute_frequency)
        self.assessment_tasks = assessment_tasks or [
            "clustering",
            "neighborhood",
            "downstream",
        ]
        self.embedding_quality_scores: dict[str, list[float]] = {}
        self.logger = logging.getLogger(__name__)

    def _record_quality_score(self, metric: str, value: float) -> None:
        history = self.embedding_quality_scores.setdefault(metric, [])
        _append_bounded(history, value)

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.compute_frequency != 0:
            return
        if logs is None:
            return

        embeddings = _to_array(logs.get("embeddings"))
        if embeddings is None:
            return

        quality_scores: ObjectMap = {}

        try:
            cluster_labels = _to_array(logs.get("cluster_labels"))
            if (
                cluster_labels is not None
                and cluster_labels.shape[0] == embeddings.shape[0]
                and np.unique(cluster_labels).size > 1
            ):
                silhouette = float(silhouette_score(embeddings, cluster_labels))
                quality_scores["clustering_silhouette"] = silhouette
                self._record_quality_score("clustering_silhouette", silhouette)

            if "neighborhood" in self.assessment_tasks:
                original_data = _to_array(logs.get("original_data"))
                if original_data is not None and original_data.shape[0] == embeddings.shape[0]:
                    trustworthiness, continuity = self._compute_neighborhood_preservation(
                        original_data,
                        embeddings,
                    )
                    quality_scores["trustworthiness"] = trustworthiness
                    quality_scores["continuity"] = continuity
                    self._record_quality_score("trustworthiness", trustworthiness)
                    self._record_quality_score("continuity", continuity)

            if "downstream" in self.assessment_tasks:
                labels = _to_array(logs.get("labels"))
                if labels is not None and labels.shape[0] == embeddings.shape[0]:
                    downstream_score = self._assess_downstream_performance(
                        embeddings,
                        labels,
                    )
                    quality_scores["downstream_score"] = downstream_score
                    self._record_quality_score("downstream_score", downstream_score)

            self.cache_metrics(
                f"embedding_quality_epoch_{context.epoch}",
                {
                    "epoch": context.epoch,
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 1,
                    **quality_scores,
                },
            )

        except Exception as exc:
            self.logger.error("Failed to assess embedding quality: %s", exc)

    def _compute_neighborhood_preservation(
        self, original_data: np.ndarray, embeddings: np.ndarray, k: int = 10
    ) -> tuple[float, float]:
        try:
            from sklearn.neighbors import NearestNeighbors

            n_samples = len(original_data)
            if n_samples <= 1:
                return 0.0, 0.0

            k_eff = min(k, n_samples - 1)
            orig_nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(original_data)
            emb_nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(embeddings)

            _, orig_indices = orig_nbrs.kneighbors(original_data)
            _, emb_indices = emb_nbrs.kneighbors(embeddings)

            trustworthiness = self._compute_trustworthiness(orig_indices, emb_indices, k_eff)
            continuity = self._compute_continuity(orig_indices, emb_indices, k_eff)
            return trustworthiness, continuity
        except Exception:
            return 0.0, 0.0

    def _compute_trustworthiness(
        self, orig_indices: np.ndarray, emb_indices: np.ndarray, k: int
    ) -> float:
        n_samples = len(orig_indices)
        if n_samples == 0 or k <= 0:
            return 0.0

        false_negative_count = 0
        for i in range(n_samples):
            orig_neighbors = set(orig_indices[i, 1 : k + 1])
            emb_neighbors = set(emb_indices[i, 1 : k + 1])
            false_negative_count += len(orig_neighbors - emb_neighbors)

        return 1.0 - (false_negative_count / (n_samples * k))

    def _compute_continuity(
        self, orig_indices: np.ndarray, emb_indices: np.ndarray, k: int
    ) -> float:
        n_samples = len(orig_indices)
        if n_samples == 0 or k <= 0:
            return 0.0

        false_positive_count = 0
        for i in range(n_samples):
            orig_neighbors = set(orig_indices[i, 1 : k + 1])
            emb_neighbors = set(emb_indices[i, 1 : k + 1])
            false_positive_count += len(emb_neighbors - orig_neighbors)

        return 1.0 - (false_positive_count / (n_samples * k))

    def _assess_downstream_performance(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> float:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            clf = LogisticRegression(max_iter=100, random_state=42)
            scores = cross_val_score(clf, embeddings, labels, cv=3, scoring="accuracy")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def get_embedding_quality_stats(self) -> ObjectMap:
        stats: ObjectMap = {"quality_metrics_count": len(self.embedding_quality_scores)}
        for metric_name, scores in self.embedding_quality_scores.items():
            if scores:
                stats.update(
                    {
                        f"{metric_name}_mean": float(np.mean(scores)),
                        f"{metric_name}_std": float(np.std(scores)),
                        f"{metric_name}_latest": scores[-1],
                        f"{metric_name}_count": len(scores),
                    }
                )
        return stats

class ConvergenceMonitorCallback(NoOpMemoryOptimizedCallback):
    """Monitor unsupervised training convergence based on tracked loss."""

    def __init__(
        self,
        convergence_threshold: float = 1e-4,
        patience: int = 10,
        monitor_frequency: int = 1,
    ):
        super().__init__(cache_size=1000)
        self.convergence_threshold = convergence_threshold
        self.patience = max(2, patience)
        self.monitor_frequency = max(1, monitor_frequency)

        self.loss_history: list[float] = []
        self.converged = False
        self.convergence_epoch = 0
        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.monitor_frequency != 0:
            return
        if logs is None:
            return

        current_loss = _as_float(logs.get("loss"))
        if current_loss is None:
            return

        _append_bounded(self.loss_history, current_loss)

        if len(self.loss_history) >= self.patience:
            recent_losses = self.loss_history[-self.patience :]
            loss_change = abs(recent_losses[-1] - recent_losses[0])
            if loss_change < self.convergence_threshold:
                if not self.converged:
                    self.logger.info("Convergence detected at epoch %s", context.epoch)
                self.converged = True
                self.convergence_epoch = context.epoch

    def has_converged(self) -> bool:
        return self.converged

    def get_convergence_info(self) -> ObjectMap:
        return {
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
            "loss_history_length": len(self.loss_history),
            "threshold": self.convergence_threshold,
        }

# Factory functions for easy instantiation

def create_clustering_metrics(**kwargs) -> ClusteringMetricsCallback:
    """Create clustering metrics callback with default settings."""
    defaults: ObjectMap = {"compute_frequency": 1, "max_samples": 5000}
    defaults.update(kwargs)
    return ClusteringMetricsCallback(**defaults)

def create_dim_reduction_metrics(**kwargs) -> DimensionalityReductionMetricsCallback:
    """Create dimensionality-reduction metrics callback with default settings."""
    defaults: ObjectMap = {"compute_frequency": 1}
    defaults.update(kwargs)
    return DimensionalityReductionMetricsCallback(**defaults)

def create_embedding_quality(**kwargs) -> EmbeddingQualityCallback:
    """Create embedding-quality callback with default settings."""
    defaults: ObjectMap = {
        "compute_frequency": 5,
        "assessment_tasks": ["clustering", "neighborhood"],
    }
    defaults.update(kwargs)
    return EmbeddingQualityCallback(**defaults)

def create_convergence_monitor(**kwargs) -> ConvergenceMonitorCallback:
    """Create convergence monitor callback with default settings."""
    defaults: ObjectMap = {
        "convergence_threshold": 1e-4,
        "patience": 10,
        "monitor_frequency": 1,
    }
    defaults.update(kwargs)
    return ConvergenceMonitorCallback(**defaults)

__all__ = [
    "ClusteringMetricsCallback",
    "DimensionalityReductionMetricsCallback",
    "EmbeddingQualityCallback",
    "ConvergenceMonitorCallback",
    "create_clustering_metrics",
    "create_dim_reduction_metrics",
    "create_embedding_quality",
    "create_convergence_monitor",
]
