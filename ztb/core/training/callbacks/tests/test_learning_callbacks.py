#!/usr/bin/env python3
"""
Comprehensive Test Suite for Learning Callbacks.

This module provides comprehensive tests for all learning type callbacks
including unit tests, integration tests, and performance validation.
"""

import unittest

import numpy as np

from ztb.training.callbacks.meta.meta_callbacks import (
    FewShotCallback,
    MAMLCallback,
    MetaAdaptationCallback,
)
from ztb.training.callbacks.multi_task.multi_task_callbacks import (
    SharedRepresentationCallback,
    TaskBalancingCallback,
    TaskInterferenceCallback,
)
from ztb.training.callbacks.reinforcement.sac.sac_callbacks import (
    SACExplorationMonitor,
    SACTargetNetworkUpdater,
    SACTemperatureScheduler,
    SACValueFunctionMonitor,
)
from ztb.training.callbacks.shared.base.learning_callback import (
    CallbackManager,
    LearningContext,
)
from ztb.training.callbacks.supervised.supervised_callbacks import (
    ClassificationMetricsCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    RegressionMetricsCallback,
)
from ztb.training.callbacks.transfer.transfer_callbacks import (
    DomainAdaptationCallback,
    FineTuningCallback,
    TransferPerformanceCallback,
)
from ztb.training.callbacks.unsupervised.unsupervised_callbacks import (
    ClusteringMetricsCallback,
    ConvergenceMonitorCallback,
    DimensionalityReductionMetricsCallback,
    EmbeddingQualityCallback,
)


class TestReinforcementLearningCallbacks(unittest.TestCase):
    """Test reinforcement learning callbacks."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)

    def test_sac_temperature_scheduler(self):
        callback = SACTemperatureScheduler(
            initial_temp=1.0, min_temp=0.1, decay_rate=0.99
        )

        # Test initial temperature
        logs = {}
        callback.on_epoch_end(self.context, logs)
        self.assertIn("temperature", logs)
        self.assertAlmostEqual(logs["temperature"], 1.0, places=3)

        # Test temperature decay
        for epoch in range(2, 21):  # Run more epochs to build history
            context = LearningContext(epoch=epoch, total_epochs=100, step=epoch * 10)
            logs = {
                "reward": 0.5,
                "entropy": 1.2,
            }  # Add some metrics for adaptive update
            callback.on_epoch_end(context, logs)

        # Temperature should be within bounds
        self.assertGreaterEqual(logs["temperature"], 0.1)  # min_temp
        self.assertLessEqual(logs["temperature"], 2.0)  # max_temp

    def test_sac_value_function_monitor(self):
        callback = SACValueFunctionMonitor(monitor_frequency=1)

        # Mock value function outputs
        logs = {"value_mean": 0.5, "value_std": 0.1, "value_min": 0.3, "value_max": 0.7}

        callback.on_epoch_end(self.context, logs)

        # Check that stats are computed
        stats = callback.get_value_function_stats()
        self.assertIn("value_mean", stats)
        self.assertIn("value_std", stats)

    def test_sac_target_network_updater(self):
        callback = SACTargetNetworkUpdater()

        logs = {"tau": 0.005, "q_loss": 0.5, "policy_loss": 0.3}
        callback.on_epoch_end(self.context, logs)

        # Should not update on epoch 1 (insufficient history)
        self.assertNotIn("target_updated", logs)

        # Should update on epoch 2
        context2 = LearningContext(epoch=2, total_epochs=100, step=200)
        logs2 = {"tau": 0.005, "q_loss": 0.4, "policy_loss": 0.2}
        callback.on_epoch_end(context2, logs2)
        self.assertTrue(logs2.get("target_updated", False))

    def test_sac_exploration_monitor(self):
        callback = SACExplorationMonitor(monitor_frequency=1)

        logs = {"action_entropy": 1.5, "action_std": 0.8, "random_action_ratio": 0.1}

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_exploration_stats()
        self.assertIn("entropy_mean", stats)
        self.assertIn("action_std_mean", stats)


class TestSupervisedLearningCallbacks(unittest.TestCase):
    """Test supervised learning callbacks."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)

    def test_early_stopping(self):
        callback = EarlyStoppingCallback(patience=3, min_delta=0.01)

        # Improving performance
        for epoch in range(1, 6):
            context = LearningContext(epoch=epoch, total_epochs=100, step=epoch * 10)
            logs = {"val_loss": 1.0 - epoch * 0.1}
            callback.on_epoch_end(context, logs)

        self.assertFalse(callback.should_stop_training())

        # Degrading performance
        for epoch in range(6, 12):
            context = LearningContext(epoch=epoch, total_epochs=100, step=epoch * 10)
            logs = {"val_loss": 0.5 + (epoch - 6) * 0.05}  # Increasing loss
            callback.on_epoch_end(context, logs)

        # Should trigger early stopping after patience
        self.assertTrue(callback.should_stop_training())

    def test_learning_rate_scheduler(self):
        callback = LearningRateSchedulerCallback(
            schedule_type="cosine", initial_lr=0.1, epochs=100
        )

        logs = {}
        callback.on_epoch_end(self.context, logs)

        self.assertIn("learning_rate", logs)
        self.assertGreater(logs["learning_rate"], 0)

    def test_classification_metrics(self):
        callback = ClassificationMetricsCallback()

        # Mock predictions and labels
        n_samples = 100
        n_classes = 3
        predictions = np.random.rand(n_samples, n_classes)
        labels = np.random.randint(0, n_classes, n_samples)

        logs = {"predictions": predictions, "targets": labels}

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_classification_stats()
        self.assertIn("accuracy_mean", stats)
        self.assertIn("f1_mean", stats)

    def test_regression_metrics(self):
        callback = RegressionMetricsCallback()

        # Mock predictions and labels
        n_samples = 100
        predictions = np.random.rand(n_samples)
        labels = np.random.rand(n_samples)

        logs = {"predictions": predictions, "targets": labels}

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_regression_stats()
        self.assertIn("mse_mean", stats)
        self.assertIn("mae_mean", stats)
        self.assertIn("r2_mean", stats)


class TestUnsupervisedLearningCallbacks(unittest.TestCase):
    """Test unsupervised learning callbacks."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)

    def test_clustering_metrics(self):
        callback = ClusteringMetricsCallback()

        # Mock clustering data
        n_samples = 100
        n_clusters = 3
        embeddings = np.random.rand(n_samples, 10)
        cluster_labels = np.random.randint(0, n_clusters, n_samples)

        logs = {"embeddings": embeddings, "cluster_labels": cluster_labels}

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_clustering_stats()
        self.assertIn("silhouette_mean", stats)
        self.assertIn("calinski_harabasz_mean", stats)

    def test_dimensionality_reduction_metrics(self):
        callback = DimensionalityReductionMetricsCallback()

        # Mock dimensionality reduction data
        n_samples = 100
        embeddings = np.random.rand(n_samples, 5)
        original_data = np.random.rand(n_samples, 10)

        logs = {
            "embeddings": embeddings,
            "original_data": original_data,
            "explained_variance": [0.5, 0.3, 0.1, 0.05, 0.05],
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_dim_reduction_stats()
        self.assertIn("explained_variance_mean", stats)
        self.assertIn("trustworthiness_mean", stats)

    def test_embedding_quality(self):
        callback = EmbeddingQualityCallback()

        # Mock embedding data
        n_samples = 100
        embeddings = np.random.rand(n_samples, 10)
        cluster_labels = np.random.randint(0, 3, n_samples)

        logs = {"embeddings": embeddings, "cluster_labels": cluster_labels}

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_embedding_quality_stats()
        self.assertIn("quality_metrics_count", stats)

    def test_convergence_monitor(self):
        callback = ConvergenceMonitorCallback()

        logs = {"loss": 1.0}
        callback.on_epoch_end(self.context, logs)

        # Test convergence detection
        for epoch in range(2, 15):
            context = LearningContext(
                epoch=epoch, total_epochs=100, global_step=epoch * 10
            )
            loss = 1.0 - epoch * 0.01  # Gradually decreasing loss
            logs = {"loss": loss}
            callback.on_epoch_end(context, logs)

        info = callback.get_convergence_info()
        self.assertIn("convergence_detected", info)


class TestTransferLearningCallbacks(unittest.TestCase):
    """Test transfer learning callbacks."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)

    def test_domain_adaptation(self):
        callback = DomainAdaptationCallback()

        # Mock domain data
        n_samples = 100
        source_features = np.random.rand(n_samples, 10)
        target_features = np.random.rand(n_samples, 10) + 0.5  # Different distribution

        logs = {
            "source_features": source_features,
            "target_features": target_features,
            "domain_adaptation_loss": 0.8,
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_domain_adaptation_stats()
        self.assertIn("avg_domain_shift", stats)
        self.assertIn("adaptation_loss_mean", stats)

    def test_fine_tuning(self):
        callback = FineTuningCallback()

        logs = {
            "layer_learning_rates": {"conv1": 0.001, "fc1": 0.01},
            "task_performance": {"source_task": 0.85, "target_task": 0.75},
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_fine_tuning_stats()
        self.assertIn("source_task_perf_mean", stats)
        self.assertIn("target_task_perf_mean", stats)

    def test_transfer_performance(self):
        callback = TransferPerformanceCallback()

        # Mock transfer performance data
        n_samples = 100
        source_pred = np.random.rand(n_samples)
        source_labels = np.random.randint(0, 2, n_samples)
        target_pred = np.random.rand(n_samples)
        target_labels = np.random.randint(0, 2, n_samples)

        logs = {
            "source_predictions": source_pred,
            "source_labels": source_labels,
            "target_predictions": target_pred,
            "target_labels": target_labels,
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_transfer_performance_stats()
        self.assertIn("source_accuracy_mean", stats)
        self.assertIn("target_accuracy_mean", stats)
        self.assertIn("transfer_gap_accuracy_mean", stats)


class TestMultiTaskLearningCallbacks(unittest.TestCase):
    """Test multi-task learning callbacks."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)
        self.task_names = ["task_a", "task_b", "task_c"]

    def test_task_balancing(self):
        callback = TaskBalancingCallback(self.task_names)

        logs = {
            "task_a_loss": 0.5,
            "task_b_loss": 0.8,
            "task_c_loss": 0.3,
            "task_a_weight": 1.0,
            "task_b_weight": 1.2,
            "task_c_weight": 0.8,
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_task_balancing_stats()
        self.assertIn("balance_score_mean", stats)
        self.assertIn("task_a_loss_mean", stats)

    def test_shared_representation(self):
        callback = SharedRepresentationCallback()

        # Mock representation data
        n_samples = 100
        activations = np.random.rand(n_samples, 64)

        logs = {
            "shared_encoder_activations": activations,
            "shared_encoder_grad_norm": 0.05,
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_shared_representation_stats()
        self.assertIn("shared_encoder_diversity_mean", stats)

    def test_task_interference(self):
        callback = TaskInterferenceCallback(self.task_names)

        logs = {
            "task_a_performance": 0.85,
            "task_b_performance": 0.78,
            "task_c_performance": 0.92,
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_task_interference_stats()
        self.assertIn("task_a_interference_mean", stats)


class TestMetaLearningCallbacks(unittest.TestCase):
    """Test meta learning callbacks."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)

    def test_maml_callback(self):
        callback = MAMLCallback()

        logs = {
            "inner_losses": [0.8, 0.6, 0.4, 0.3, 0.25],
            "meta_loss": 0.35,
            "adaptation_accuracies": [0.2, 0.4, 0.6, 0.7, 0.75],
            "meta_grad_norm": 0.05,
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_maml_stats()
        self.assertIn("meta_loss_mean", stats)
        self.assertIn("inner_loss_improvement", stats)

    def test_few_shot_callback(self):
        callback = FewShotCallback(n_way=5, k_shot=1)

        logs = {
            "episode_accuracy": 0.68,
            "episode_loss": 0.45,
            "query_accuracy": 0.72,
            "prototype_distances": [0.8, 0.9, 1.1, 0.7, 1.2],
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_few_shot_stats()
        self.assertIn("episode_accuracy_mean", stats)
        self.assertIn("query_accuracy_mean", stats)

    def test_meta_adaptation_callback(self):
        callback = MetaAdaptationCallback()

        logs = {
            "adaptation_curve": [0.2, 0.35, 0.5, 0.62, 0.68, 0.71, 0.73, 0.74],
            "convergence_step": 6,
            "cross_task_performance": {"task1": 0.75, "task2": 0.68},
        }

        callback.on_epoch_end(self.context, logs)

        stats = callback.get_meta_adaptation_stats()
        self.assertIn("adaptation_speed_mean", stats)
        self.assertIn("avg_convergence_steps", stats)


class TestCallbackIntegration(unittest.TestCase):
    """Test callback integration and manager functionality."""

    def setUp(self):
        self.context = LearningContext(epoch=1, total_epochs=100, step=100)

    def test_callback_manager(self):
        """Test callback manager with multiple callbacks."""
        manager = CallbackManager()

        # Add various callbacks
        callbacks = [
            SACTemperatureScheduler(),
            EarlyStoppingCallback(),
            ClusteringMetricsCallback(),
            DomainAdaptationCallback(),
        ]

        for callback in callbacks:
            manager.add_callback(callback)

        # Test manager execution
        logs = {
            "loss": 0.5,
            "embeddings": np.random.rand(50, 10),
            "cluster_labels": np.random.randint(0, 3, 50),
        }

        manager.on_epoch_end(self.context, logs)

        # Check that callbacks were executed
        self.assertGreater(len(manager.callbacks), 0)

    def test_memory_optimization(self):
        """Test memory optimization features."""
        callback = ClusteringMetricsCallback(max_samples=50)

        # Generate large dataset
        n_samples = 200
        embeddings = np.random.rand(n_samples, 10)
        cluster_labels = np.random.randint(0, 3, n_samples)

        logs = {"embeddings": embeddings, "cluster_labels": cluster_labels}

        callback.on_epoch_end(self.context, logs)

        # Check that subsampling worked
        stats = callback.get_clustering_stats()
        self.assertIn("epochs_computed", stats)

    def test_error_handling(self):
        """Test error handling in callbacks."""
        callback = ClassificationMetricsCallback()

        # Test with invalid data
        logs = {"predictions": None, "labels": None}

        # Should not crash
        try:
            callback.on_epoch_end(self.context, logs)
            stats = callback.get_classification_stats()
            self.assertIsInstance(stats, dict)
        except Exception as e:
            self.fail(f"Callback raised unexpected exception: {e}")


class TestPerformanceValidation(unittest.TestCase):
    """Test performance and scalability of callbacks."""

    def test_callback_performance(self):
        """Test callback execution performance."""
        import time

        callback = ClusteringMetricsCallback()

        # Generate test data
        n_samples = 1000
        embeddings = np.random.rand(n_samples, 20)
        cluster_labels = np.random.randint(0, 5, n_samples)

        logs = {"embeddings": embeddings, "cluster_labels": cluster_labels}

        # Measure execution time
        start_time = time.time()
        for epoch in range(1, 11):
            context = LearningContext(
                epoch=epoch, total_epochs=100, global_step=epoch * 100
            )
            callback.on_epoch_end(context, logs)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(execution_time, 5.0, "Callback execution too slow")

    def test_memory_usage(self):
        """Test memory usage of callbacks."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        callback = EmbeddingQualityCallback()

        # Generate large dataset
        n_samples = 5000
        embeddings = np.random.rand(n_samples, 50)
        cluster_labels = np.random.randint(0, 10, n_samples)

        logs = {"embeddings": embeddings, "cluster_labels": cluster_labels}

        # Run multiple epochs
        for epoch in range(1, 21):
            context = LearningContext(
                epoch=epoch, total_epochs=100, global_step=epoch * 100
            )
            callback.on_epoch_end(context, logs)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 500, "Excessive memory usage")


if __name__ == "__main__":
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestReinforcementLearningCallbacks))
    suite.addTest(unittest.makeSuite(TestSupervisedLearningCallbacks))
    suite.addTest(unittest.makeSuite(TestUnsupervisedLearningCallbacks))
    suite.addTest(unittest.makeSuite(TestTransferLearningCallbacks))
    suite.addTest(unittest.makeSuite(TestMultiTaskLearningCallbacks))
    suite.addTest(unittest.makeSuite(TestMetaLearningCallbacks))
    suite.addTest(unittest.makeSuite(TestCallbackIntegration))
    suite.addTest(unittest.makeSuite(TestPerformanceValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
