#!/usr/bin/env python3
"""
Unsupervised Learning Callbacks.

This module provides callbacks optimized for unsupervised learning
tasks including clustering and dimensionality reduction.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    MemoryOptimizedCallback,
)


class ClusteringMetricsCallback(MemoryOptimizedCallback):
    """
    Clustering metrics callback.

    Computes and tracks clustering quality metrics including silhouette score,
    Calinski-Harabasz index, and Davies-Bouldin index.
    """

    def __init__(self, compute_frequency: int = 1, max_samples: int = 5000):
        super().__init__(cache_size=500)
        self.compute_frequency = compute_frequency
        self.max_samples = max_samples

        # Metrics history
        self.silhouette_history: List[float] = []
        self.calinski_harabasz_history: List[float] = []
        self.davies_bouldin_history: List[float] = []

        # Cluster stability tracking
        self.cluster_centers_history: List[np.ndarray] = []
        self.cluster_stability_scores: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Compute clustering metrics."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None or "embeddings" not in logs or "cluster_labels" not in logs:
            return

        embeddings = logs["embeddings"]
        cluster_labels = logs["cluster_labels"]

        # Subsample if too many samples
        if len(embeddings) > self.max_samples:
            indices = np.random.choice(len(embeddings), self.max_samples, replace=False)
            embeddings = embeddings[indices]
            cluster_labels = cluster_labels[indices]

        try:
            # Compute clustering metrics
            if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
                silhouette = silhouette_score(embeddings, cluster_labels)
                calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
                davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)

                # Store in history
                self.silhouette_history.append(silhouette)
                self.calinski_harabasz_history.append(calinski_harabasz)
                self.davies_bouldin_history.append(davies_bouldin)

                # Track cluster centers for stability
                if "cluster_centers" in logs:
                    centers = logs["cluster_centers"]
                    self.cluster_centers_history.append(centers.copy())
                    self._compute_cluster_stability()

                # Cache metrics
                metrics_key = f"clustering_epoch_{context.epoch}"
                self.cache_metrics(
                    metrics_key,
                    {
                        "silhouette_score": silhouette,
                        "calinski_harabasz_score": calinski_harabasz,
                        "davies_bouldin_score": davies_bouldin,
                        "num_clusters": len(np.unique(cluster_labels)),
                        "epoch": context.epoch,
                    },
                )

                self.logger.debug(
                    f"Clustering metrics - Silhouette: {silhouette:.4f}, "
                    f"CH: {calinski_harabasz:.2f} at epoch {context.epoch}"
                )
            else:
                self.logger.warning(
                    "Only one cluster found, skipping clustering metrics"
                )

        except Exception as e:
            self.logger.error(f"Failed to compute clustering metrics: {e}")

    def _compute_cluster_stability(self) -> None:
        """Compute cluster stability score based on center movement."""
        if len(self.cluster_centers_history) < 2:
            return

        # Compare current centers with previous centers
        current_centers = self.cluster_centers_history[-1]
        previous_centers = self.cluster_centers_history[-2]

        # Compute center movement (Euclidean distance)
        if current_centers.shape == previous_centers.shape:
            center_movement = np.mean(
                np.linalg.norm(current_centers - previous_centers, axis=1)
            )
            stability_score = 1.0 / (
                1.0 + center_movement
            )  # Higher score = more stable

            self.cluster_stability_scores.append(stability_score)
        else:
            self.logger.warning(
                "Cluster centers shape mismatch, skipping stability computation"
            )

    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering metrics statistics."""
        stats = {
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


class DimensionalityReductionMetricsCallback(MemoryOptimizedCallback):
    """
    Dimensionality reduction metrics callback.

    Monitors the quality of dimensionality reduction techniques
    including explained variance, reconstruction error, and preservation
    of local/global structure.
    """

    def __init__(
        self, compute_frequency: int = 1, original_data: Optional[np.ndarray] = None
    ):
        super().__init__(cache_size=500)
        self.compute_frequency = compute_frequency
        self.original_data = original_data

        # Metrics history
        self.explained_variance_history: List[float] = []
        self.reconstruction_error_history: List[float] = []
        self.trustworthiness_history: List[float] = []
        self.continuity_history: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Compute dimensionality reduction metrics."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None or "embeddings" not in logs:
            return

        embeddings = logs["embeddings"]

        try:
            # Compute explained variance (for PCA-like methods)
            if "explained_variance" in logs:
                explained_var = logs["explained_variance"]
                if isinstance(explained_var, (list, np.ndarray)):
                    explained_var = float(np.sum(explained_var))
                self.explained_variance_history.append(explained_var)

            # Compute reconstruction error if original data available
            if self.original_data is not None and "reconstructed_data" in logs:
                reconstructed = logs["reconstructed_data"]
                reconstruction_error = np.mean(
                    np.square(self.original_data - reconstructed)
                )
                self.reconstruction_error_history.append(reconstruction_error)

            # Compute neighborhood preservation metrics
            if self.original_data is not None:
                trustworthiness, continuity = self._compute_neighborhood_preservation(
                    self.original_data, embeddings
                )
                self.trustworthiness_history.append(trustworthiness)
                self.continuity_history.append(continuity)

            # Cache metrics
            metrics_key = f"dim_reduction_epoch_{context.epoch}"
            metrics_data = {
                "embedding_dim": embeddings.shape[1]
                if len(embeddings.shape) > 1
                else 1,
                "num_samples": len(embeddings),
                "epoch": context.epoch,
            }

            if self.explained_variance_history:
                metrics_data["explained_variance"] = self.explained_variance_history[-1]
            if self.reconstruction_error_history:
                metrics_data[
                    "reconstruction_error"
                ] = self.reconstruction_error_history[-1]
            if self.trustworthiness_history:
                metrics_data["trustworthiness"] = self.trustworthiness_history[-1]
                metrics_data["continuity"] = self.continuity_history[-1]

            self.cache_metrics(metrics_key, metrics_data)

            self.logger.debug(
                f"Dim reduction metrics computed for epoch {context.epoch}"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to compute dimensionality reduction metrics: {e}"
            )

    def _compute_neighborhood_preservation(
        self, original_data: np.ndarray, embeddings: np.ndarray, k: int = 10
    ) -> Tuple[float, float]:
        """Compute trustworthiness and continuity metrics."""
        try:
            # Compute pairwise distances
            orig_distances = squareform(pdist(original_data))
            emb_distances = squareform(pdist(embeddings))

            n_samples = len(original_data)
            trustworthiness = 0.0
            continuity = 0.0

            for i in range(n_samples):
                # Trustworthiness: measures how many points that are neighbors in
                # the original space are also neighbors in the embedding
                orig_neighbors = np.argsort(orig_distances[i])[: k + 1][
                    1:
                ]  # Exclude self
                emb_neighbors = np.argsort(emb_distances[i])[: k + 1][1:]

                # Points that are in original neighbors but not in embedding neighbors
                false_negatives = len(set(orig_neighbors) - set(emb_neighbors))
                trustworthiness += false_negatives / k

                # Continuity: measures how many points that are neighbors in
                # the embedding are also neighbors in the original space
                false_positives = len(set(emb_neighbors) - set(orig_neighbors))
                continuity += false_positives / k

            trustworthiness = 1.0 - (trustworthiness / n_samples)
            continuity = 1.0 - (continuity / n_samples)

            return trustworthiness, continuity

        except Exception as e:
            self.logger.warning(f"Failed to compute neighborhood preservation: {e}")
            return 0.0, 0.0

    def get_dim_reduction_stats(self) -> Dict[str, Any]:
        """Get dimensionality reduction metrics statistics."""
        stats = {"epochs_computed": len(self.explained_variance_history)}

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
                    "reconstruction_error_latest": self.reconstruction_error_history[
                        -1
                    ],
                }
            )

        if self.trustworthiness_history:
            stats.update(
                {
                    "trustworthiness_mean": float(
                        np.mean(self.trustworthiness_history)
                    ),
                    "trustworthiness_latest": self.trustworthiness_history[-1],
                    "continuity_mean": float(np.mean(self.continuity_history)),
                    "continuity_latest": self.continuity_history[-1],
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


class EmbeddingQualityCallback(MemoryOptimizedCallback):
    """
    Embedding quality assessment callback.

    Evaluates the quality of learned embeddings using various metrics
    including clustering quality, neighborhood preservation, and
    downstream task performance.
    """

    def __init__(
        self, compute_frequency: int = 5, assessment_tasks: Optional[List[str]] = None
    ):
        super().__init__(cache_size=1000)
        self.compute_frequency = compute_frequency
        self.assessment_tasks = assessment_tasks or [
            "clustering",
            "neighborhood",
            "downstream",
        ]

        # Quality metrics history
        self.embedding_quality_scores: Dict[str, List[float]] = {}
        self.downstream_performance: Dict[str, List[float]] = {}

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Assess embedding quality."""
        if context.epoch % self.compute_frequency != 0:
            return

        if logs is None or "embeddings" not in logs:
            return

        embeddings = logs["embeddings"]
        quality_scores = {}

        try:
            # Assess clustering quality
            if "clustering" in self.assessment_tasks and "cluster_labels" in logs:
                cluster_labels = logs["cluster_labels"]
                if len(np.unique(cluster_labels)) > 1:
                    silhouette = silhouette_score(embeddings, cluster_labels)
                    quality_scores["clustering_silhouette"] = silhouette

                    # Initialize history if needed
                    if "clustering_silhouette" not in self.embedding_quality_scores:
                        self.embedding_quality_scores["clustering_silhouette"] = []
                    self.embedding_quality_scores["clustering_silhouette"].append(
                        silhouette
                    )

            # Assess neighborhood preservation
            if "neighborhood" in self.assessment_tasks and "original_data" in logs:
                original_data = logs["original_data"]
                trustworthiness, continuity = self._compute_neighborhood_preservation(
                    original_data, embeddings
                )
                quality_scores["trustworthiness"] = trustworthiness
                quality_scores["continuity"] = continuity

                for metric in ["trustworthiness", "continuity"]:
                    if metric not in self.embedding_quality_scores:
                        self.embedding_quality_scores[metric] = []
                    self.embedding_quality_scores[metric].append(quality_scores[metric])

            # Assess downstream task performance (simplified)
            if "downstream" in self.assessment_tasks and "labels" in logs:
                labels = logs["labels"]
                downstream_score = self._assess_downstream_performance(
                    embeddings, labels
                )
                quality_scores["downstream_score"] = downstream_score

                if "downstream_score" not in self.embedding_quality_scores:
                    self.embedding_quality_scores["downstream_score"] = []
                self.embedding_quality_scores["downstream_score"].append(
                    downstream_score
                )

            # Cache quality scores
            quality_key = f"embedding_quality_epoch_{context.epoch}"
            quality_data = {
                "epoch": context.epoch,
                "embedding_dim": embeddings.shape[1]
                if len(embeddings.shape) > 1
                else 1,
                **quality_scores,
            }
            self.cache_metrics(quality_key, quality_data)

            self.logger.debug(
                f"Embedding quality assessed for epoch {context.epoch}: "
                f"{quality_scores}"
            )

        except Exception as e:
            self.logger.error(f"Failed to assess embedding quality: {e}")

    def _compute_neighborhood_preservation(
        self, original_data: np.ndarray, embeddings: np.ndarray, k: int = 10
    ) -> Tuple[float, float]:
        """Compute neighborhood preservation metrics."""
        # Simplified version - in practice would use more sophisticated methods
        try:
            from sklearn.neighbors import NearestNeighbors

            # Find k-nearest neighbors in original and embedding spaces
            orig_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(original_data)
            emb_nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)

            orig_distances, orig_indices = orig_nbrs.kneighbors(original_data)
            emb_distances, emb_indices = emb_nbrs.kneighbors(embeddings)

            # Compute trustworthiness and continuity
            trustworthiness = self._compute_trustworthiness(
                orig_indices, emb_indices, k
            )
            continuity = self._compute_continuity(orig_indices, emb_indices, k)

            return trustworthiness, continuity

        except Exception:
            return 0.0, 0.0

    def _compute_trustworthiness(
        self, orig_indices: np.ndarray, emb_indices: np.ndarray, k: int
    ) -> float:
        """Compute trustworthiness score."""
        n_samples = len(orig_indices)
        trustworthiness = 0.0

        for i in range(n_samples):
            orig_neighbors = set(orig_indices[i, 1 : k + 1])  # Exclude self
            emb_neighbors = set(emb_indices[i, 1 : k + 1])

            # Points that are neighbors in original but not in embedding
            false_negatives = len(orig_neighbors - emb_neighbors)
            trustworthiness += false_negatives

        return 1.0 - (trustworthiness / (n_samples * k))

    def _compute_continuity(
        self, orig_indices: np.ndarray, emb_indices: np.ndarray, k: int
    ) -> float:
        """Compute continuity score."""
        n_samples = len(orig_indices)
        continuity = 0.0

        for i in range(n_samples):
            orig_neighbors = set(orig_indices[i, 1 : k + 1])
            emb_neighbors = set(emb_indices[i, 1 : k + 1])

            # Points that are neighbors in embedding but not in original
            false_positives = len(emb_neighbors - orig_neighbors)
            continuity += false_positives

        return 1.0 - (continuity / (n_samples * k))

    def _assess_downstream_performance(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> float:
        """Assess downstream task performance (simplified linear probe)."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            # Simple linear probe with cross-validation
            clf = LogisticRegression(max_iter=100, random_state=42)
            scores = cross_val_score(clf, embeddings, labels, cv=3, scoring="accuracy")
            return float(np.mean(scores))

        except Exception:
            return 0.0

    def get_embedding_quality_stats(self) -> Dict[str, Any]:
        """Get embedding quality statistics."""
        stats = {"quality_metrics_count": len(self.embedding_quality_scores)}

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


class ConvergenceMonitorCallback(MemoryOptimizedCallback):
    """
    Convergence monitoring callback for unsupervised learning.

    Monitors convergence of unsupervised learning algorithms
    including changes in loss, cluster assignments, and embedding positions.
    """

    def __init__(
        self,
        convergence_threshold: float = 1e-4,
        patience: int = 10,
        monitor_frequency: int = 1,
    ):
        super().__init__()
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.monitor_frequency = monitor_frequency

        # Convergence tracking
        self.loss_history: List[float] = []
        self.convergence_detected = False
        self.convergence_epoch = 0
        self.patience_counter = 0

        # Change tracking
        self.previous_embeddings: Optional[np.ndarray] = None
        self.embedding_changes: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor convergence."""
        if context.epoch % self.monitor_frequency != 0:
            return

        if logs is None:
            return

        # Track loss convergence
        if "loss" in logs:
            current_loss = logs["loss"]
            self.loss_history.append(current_loss)

            # Check for loss convergence
            if len(self.loss_history) >= 3:
                recent_losses = self.loss_history[-3:]
                loss_change = abs(recent_losses[-1] - recent_losses[-2])
                if loss_change < self.convergence_threshold:
                    self.patience_counter += 1
                    if (
                        self.patience_counter >= self.patience
                        and not self.convergence_detected
                    ):
                        self.convergence_detected = True
                        self.convergence_epoch = context.epoch
                        self.logger.info(
                            f"Convergence detected at epoch {context.epoch}, "
                            f"loss change: {loss_change:.6f}"
                        )
                else:
                    self.patience_counter = 0

        # Track embedding changes
        if "embeddings" in logs:
            current_embeddings = logs["embeddings"]

            if self.previous_embeddings is not None:
                # Compute embedding change (simplified)
                if current_embeddings.shape == self.previous_embeddings.shape:
                    embedding_change = np.mean(
                        np.abs(current_embeddings - self.previous_embeddings)
                    )
                    self.embedding_changes.append(embedding_change)

                    if (
                        embedding_change < self.convergence_threshold
                        and not self.convergence_detected
                    ):
                        self.convergence_detected = True
                        self.convergence_epoch = context.epoch
                        self.logger.info(
                            f"Embedding convergence detected at epoch {context.epoch}, "
                            f"change: {embedding_change:.6f}"
                        )

            self.previous_embeddings = current_embeddings.copy()

    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence monitoring information."""
        return {
            "convergence_detected": self.convergence_detected,
            "convergence_epoch": self.convergence_epoch,
            "patience_counter": self.patience_counter,
            "loss_history_length": len(self.loss_history),
            "embedding_changes_count": len(self.embedding_changes),
            "convergence_threshold": self.convergence_threshold,
            "patience": self.patience,
        }

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
def create_clustering_metrics(**kwargs) -> ClusteringMetricsCallback:
    """Create clustering metrics callback with default settings."""
    defaults = {"compute_frequency": 1, "max_samples": 5000}
    defaults.update(kwargs)
    return ClusteringMetricsCallback(**defaults)


def create_dim_reduction_metrics(**kwargs) -> DimensionalityReductionMetricsCallback:
    """Create dimensionality reduction metrics callback with default settings."""
    defaults = {"compute_frequency": 1}
    defaults.update(kwargs)
    return DimensionalityReductionMetricsCallback(**defaults)


def create_embedding_quality(**kwargs) -> EmbeddingQualityCallback:
    """Create embedding quality callback with default settings."""
    defaults = {
        "compute_frequency": 5,
        "assessment_tasks": ["clustering", "neighborhood"],
    }
    defaults.update(kwargs)
    return EmbeddingQualityCallback(**defaults)


def create_convergence_monitor(**kwargs) -> ConvergenceMonitorCallback:
    """Create convergence monitor callback with default settings."""
    defaults = {"convergence_threshold": 1e-4, "patience": 10, "monitor_frequency": 1}
    defaults.update(kwargs)
    return ConvergenceMonitorCallback(**defaults)
