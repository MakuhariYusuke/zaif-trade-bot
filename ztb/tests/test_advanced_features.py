#!/usr/bin/env python3
"""
Comprehensive tests for SAC v421 advanced features: Anomaly Detection, Meta Learning, and Federated Learning.
"""

import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")

from ztb.adaptation.continual_learning import (
    ContinualLearner,
    ContinualLearningConfig,
    ElasticWeightConsolidation,
    ProgressiveNetwork,
    RehearsalBuffer,
    TaskData,
)
from ztb.adaptation.meta_learning import MarketMetaLearner
from ztb.data.anomaly_detection import (
    AutoencoderAnomalyDetector,
    ComprehensiveAnomalyDetector,
    MLAnomalyDetector,
    StatisticalAnomalyDetector,
)
from ztb.training.federated_learning import (
    FedAvgServer,
    FederatedClient,
    FederatedConfig,
    MarketFederatedLearner,
)
from ztb.training.unified_trainer.config import UnifiedAlgorithm


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for anomaly detection system."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = np.random.normal(0, 1, (100, 5))
        self.anomaly_data = np.random.normal(5, 1, (10, 5))  # Anomalous data

    def test_statistical_anomaly_detector(self):
        """Test statistical anomaly detection."""
        detector = StatisticalAnomalyDetector(method="zscore", threshold=3.0)

        # Test normal data
        result = detector.detect(self.test_data[0])
        self.assertIsInstance(result.is_anomaly, bool)
        self.assertIsInstance(result.anomaly_score, float)

        # Test with anomaly
        result = detector.detect(self.anomaly_data[0])
        self.assertIsInstance(result.is_anomaly, bool)

    def test_ml_anomaly_detector(self):
        """Test ML-based anomaly detection."""
        detector = MLAnomalyDetector(method="isolation_forest")

        # Fit detector
        success = detector.fit(self.test_data)
        self.assertTrue(success)

        # Test detection
        result = detector.detect(self.test_data[0])
        self.assertIsInstance(result.is_anomaly, bool)
        self.assertIsInstance(result.anomaly_score, float)

    def test_autoencoder_anomaly_detector(self):
        """Test autoencoder-based anomaly detection."""
        detector = AutoencoderAnomalyDetector(input_dim=5)

        # Fit detector
        success = detector.fit(self.test_data, epochs=5)
        self.assertTrue(success)

        # Test detection
        result = detector.detect(self.test_data[0])
        self.assertIsInstance(result.is_anomaly, bool)
        self.assertIsInstance(result.anomaly_score, float)

    def test_comprehensive_anomaly_detector(self):
        """Test comprehensive anomaly detection system."""
        detector = ComprehensiveAnomalyDetector(
            statistical_methods=["zscore"],
            ml_methods=["isolation_forest"],
            voting_threshold=0.5,
        )

        # Fit ML detectors
        success = detector.fit_ml_detectors(self.test_data)
        self.assertTrue(success)

        # Test detection
        is_anomaly, results = detector.detect_anomalies(self.test_data[0])
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(results, dict)
        self.assertIn("anomaly_score", results)
        self.assertIn("method_results", results)

    def test_anomaly_stats(self):
        """Test anomaly detection statistics."""
        detector = ComprehensiveAnomalyDetector()

        # Run multiple detections
        for i in range(10):
            detector.detect_anomalies(self.test_data[i])

        stats = detector.get_stats()
        self.assertEqual(stats.total_samples, 10)
        self.assertIsInstance(stats.anomaly_rate, float)


class TestMetaLearning(unittest.TestCase):
    """Test cases for meta learning system."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.action_dim = 4

        # Create simple model
        self.base_model = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Tanh(),
        )

    def test_maml_algorithm(self):
        """Test MAML algorithm."""
        from ztb.adaptation.meta_learning import MAML, MetaLearningConfig

        config = MetaLearningConfig()
        maml = MAML(self.base_model, config)

        # Create dummy task data
        task_data = Mock()
        task_data.states = torch.randn(20, self.state_dim)
        task_data.actions = torch.randn(20, self.action_dim)
        task_data.rewards = torch.randn(20, 1)
        task_data.next_states = torch.randn(20, self.state_dim)
        task_data.dones = torch.randint(0, 2, (20, 1)).float()

        def dummy_loss(outputs, actions, rewards, next_outputs, dones):
            return torch.mean((outputs - actions) ** 2)

        # Test adaptation
        adapted_model = maml.adapt_to_task(task_data, dummy_loss)
        self.assertIsInstance(adapted_model, nn.Module)

        # Test meta update
        losses = [torch.tensor(1.0), torch.tensor(0.8)]
        meta_loss = maml.meta_update(losses)
        self.assertIsInstance(meta_loss, float)

    def test_reptile_algorithm(self):
        """Test Reptile algorithm."""
        from ztb.adaptation.meta_learning import MetaLearningConfig, Reptile

        config = MetaLearningConfig()
        reptile = Reptile(self.base_model, config)

        # Create dummy task data
        task_data = Mock()
        task_data.states = torch.randn(20, self.state_dim)
        task_data.actions = torch.randn(20, self.action_dim)
        task_data.rewards = torch.randn(20, 1)
        task_data.next_states = torch.randn(20, self.state_dim)
        task_data.dones = torch.randint(0, 2, (20, 1)).float()

        def dummy_loss(outputs, actions, rewards, next_outputs, dones):
            return torch.mean((outputs - actions) ** 2)

        # Test adaptation
        adapted_model, task_info = reptile.adapt_to_task(task_data, dummy_loss)
        self.assertIsInstance(adapted_model, nn.Module)
        self.assertIn("adaptation_losses", task_info)

    def test_market_meta_learner(self):
        """Test market-specific meta learning."""
        meta_learner = MarketMetaLearner(
            state_dim=self.state_dim, action_dim=self.action_dim
        )

        # Add market data
        market_data = {
            "states": np.random.randn(50, self.state_dim),
            "actions": np.random.randn(50, self.action_dim),
            "rewards": np.random.randn(50, 1),
            "next_states": np.random.randn(50, self.state_dim),
            "dones": np.random.randint(0, 2, (50, 1)),
        }

        meta_learner.add_market_data("test_market", **market_data)

        # Test training
        history = meta_learner.train_on_markets(num_epochs=2)
        self.assertIsInstance(history, dict)
        self.assertIn("meta_losses", history)

        # Test adaptation
        adapted_model = meta_learner.adapt_to_market("test_market", market_data)
        self.assertIsInstance(adapted_model, nn.Module)

        # Test prediction
        state = np.random.randn(self.state_dim)
        action = meta_learner.predict_market_action("test_market", state)
        self.assertEqual(action.shape, (self.action_dim,))


class TestFederatedLearning(unittest.TestCase):
    """Test cases for federated learning system."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))

        self.config = FederatedConfig(
            num_clients=3,
            num_rounds=2,
            local_epochs=1,
            enable_privacy=False,  # Disable for testing
        )

    def test_federated_client(self):
        """Test federated client functionality."""
        client = FederatedClient(0, self.base_model, self.config)

        # Create dummy data
        data = torch.randn(20, 10)
        targets = torch.randn(20, 4)
        dataset = torch.utils.data.TensorDataset(data, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        client.local_data = dataloader

        def dummy_loss(outputs, targets):
            return torch.nn.functional.mse_loss(outputs, targets)

        # Test local training
        update = client.train_local_model(dummy_loss)
        self.assertIsInstance(update, object)
        self.assertEqual(update.client_id, 0)
        self.assertIsInstance(update.model_state, dict)

    def test_fedavg_server(self):
        """Test FedAvg server functionality."""
        server = FedAvgServer(self.base_model, self.config)

        # Create clients
        clients = []
        dataloaders = []

        for i in range(3):
            client = FederatedClient(i, self.base_model, self.config)

            # Create client data
            data = torch.randn(20, 10)
            targets = torch.randn(20, 4)
            dataset = torch.utils.data.TensorDataset(data, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

            clients.append(client)
            dataloaders.append(dataloader)
            server.add_client(client)

        server.initialize_clients(dataloaders)

        def dummy_loss(outputs, targets):
            return torch.nn.functional.mse_loss(outputs, targets)

        # Test federated round
        result = server.run_federated_round(dummy_loss)
        self.assertIsInstance(result, object)
        self.assertEqual(result.round_number, 1)
        self.assertIsInstance(result.global_loss, float)

    def test_market_federated_learner(self):
        """Test market-based federated learning."""
        market_configs = {"market1": self.config, "market2": self.config}

        federated_learner = MarketFederatedLearner(self.base_model, market_configs)

        # Add clients to markets
        for market in ["market1", "market2"]:
            for i in range(2):
                data = torch.randn(20, 10)
                targets = torch.randn(20, 4)
                dataset = torch.utils.data.TensorDataset(data, targets)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

                federated_learner.add_market_client(market, dataloader, i)

        def dummy_loss(outputs, targets):
            return torch.nn.functional.mse_loss(outputs, targets)

        # Test training
        results = federated_learner.train_all_markets(dummy_loss)
        self.assertIsInstance(results, dict)
        self.assertIn("market1", results)
        self.assertIn("market2", results)

        # Test cross-market aggregation
        aggregated_model = federated_learner.aggregate_cross_market_knowledge()
        self.assertIsInstance(aggregated_model, nn.Module)

        # Test stats
        stats = federated_learner.get_federated_stats()
        self.assertIsInstance(stats, dict)


class TestAdvancedFeaturesIntegration(unittest.TestCase):
    """Test integration of advanced features in UnifiedTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "algorithm": "ppo",
            "total_timesteps": 100,
            "model_name": "test_advanced",
            "enable_anomaly_detection": True,
            "enable_meta_learning": True,
            "enable_federated": True,
            "federated_markets": True,
            "markets": ["market1", "market2"],
        }

    @patch("ztb.training.unified_trainer.trainer.create_algorithm_trainer")
    def test_advanced_features_setup(self, mock_create_trainer):
        """Test advanced features setup in UnifiedTrainer."""
        from ztb.training.unified_trainer.trainer import UnifiedTrainer

        # Mock algorithm trainer
        mock_trainer = Mock()
        mock_trainer.model = self.base_model
        mock_trainer.train.return_value = True
        mock_create_trainer.return_value = mock_trainer

        trainer = UnifiedTrainer(self.config, dry_run=True)

        # Test setup (would be called during training)
        trainer._setup_advanced_features()

        # Check if components were initialized
        self.assertIsNotNone(trainer.anomaly_detector)
        self.assertIsNotNone(trainer.meta_learner)
        self.assertIsNotNone(trainer.federated_learner)

    def test_config_validation(self):
        """Test configuration validation for advanced features."""
        from ztb.training.unified_trainer.config import UnifiedTrainerConfig

        config = UnifiedTrainerConfig(
            algorithm=UnifiedAlgorithm.PPO,
            enable_anomaly_detection=True,
            enable_meta_learning=True,
            federated_markets=True,
            markets=["test_market"],
        )

        # Check config attributes
        self.assertTrue(config.enable_anomaly_detection)
        self.assertTrue(config.enable_meta_learning)
        self.assertTrue(config.federated_markets)
        self.assertEqual(config.markets, ["test_market"])


class TestContinualLearning(unittest.TestCase):
    """Test cases for continual learning system."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))

        self.config = ContinualLearningConfig(
            method="ewc",
            ewc_lambda=0.1,
            rehearsal_buffer_size=100,
            max_tasks_in_memory=3,
        )

        # Create sample task data
        self.task_data = TaskData(
            task_id="test_task",
            states=torch.randn(50, 10),
            actions=torch.randn(50, 4),
            rewards=torch.randn(50, 1),
            next_states=torch.randn(50, 10),
            dones=torch.randint(0, 2, (50,)).float(),
            num_samples=50,
        )

    def test_elastic_weight_consolidation(self):
        """Test EWC functionality."""
        ewc = ElasticWeightConsolidation(self.model, self.config)

        def dummy_loss(outputs, actions, rewards, next_outputs, dones):
            return torch.nn.functional.mse_loss(outputs, actions)

        # Consolidate task
        stats = ewc.consolidate_task(self.task_data, dummy_loss)
        self.assertIsInstance(stats, dict)
        self.assertIn("task_id", stats)

        # Test regularization loss
        current_params = {name: param for name, param in self.model.named_parameters()}
        reg_loss = ewc.regularization_loss(current_params)
        self.assertIsInstance(reg_loss, torch.Tensor)

    def test_rehearsal_buffer(self):
        """Test rehearsal buffer functionality."""
        buffer = RehearsalBuffer(self.config)

        # Add samples
        buffer.add_samples(self.task_data)

        # Get rehearsal batch
        batch = buffer.get_rehearsal_batch(batch_size=10)
        self.assertIsNotNone(batch)
        self.assertIn("states", batch)

        # Get buffer stats
        stats = buffer.get_buffer_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_samples", stats)

    def test_progressive_network(self):
        """Test progressive network functionality."""
        progressive_net = ProgressiveNetwork(self.model, self.config)

        # Add task network
        task_net = progressive_net.add_task_network("task1")
        self.assertIsInstance(task_net, nn.Module)

        # Test forward with lateral connections
        x = torch.randn(5, 10)
        output = progressive_net.forward_with_lateral("task1", x)
        self.assertEqual(output.shape, (5, 4))

        # Get network stats
        stats = progressive_net.get_network_stats()
        self.assertIsInstance(stats, dict)

    def test_continual_learner_ewc(self):
        """Test continual learner with EWC."""
        learner = ContinualLearner(self.model, self.config)

        def dummy_loss(outputs, actions, rewards, next_outputs, dones):
            return torch.nn.functional.mse_loss(outputs, actions)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Learn task
        stats = learner.learn_task(self.task_data, dummy_loss, optimizer, num_epochs=2)
        self.assertIsInstance(stats, dict)
        self.assertIn("final_loss", stats)

        # Test prediction
        state = torch.randn(1, 10)
        output = learner.predict_with_continual(state)
        self.assertEqual(output.shape, (1, 4))

        # Get stats
        learner_stats = learner.get_continual_stats()
        self.assertIsInstance(learner_stats, dict)

    def test_continual_learner_rehearsal(self):
        """Test continual learner with rehearsal."""
        config = ContinualLearningConfig(method="rehearsal", rehearsal_buffer_size=50)
        learner = ContinualLearner(self.model, config)

        def dummy_loss(outputs, actions, rewards, next_outputs, dones):
            return torch.nn.functional.mse_loss(outputs, actions)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Learn task
        stats = learner.learn_task(self.task_data, dummy_loss, optimizer, num_epochs=2)
        self.assertIsInstance(stats, dict)

        # Check rehearsal buffer
        learner_stats = learner.get_continual_stats()
        self.assertIn("rehearsal_stats", learner_stats)

    def test_continual_learner_progressive(self):
        """Test continual learner with progressive networks."""
        config = ContinualLearningConfig(method="progressive")
        learner = ContinualLearner(self.model, config)

        def dummy_loss(outputs, actions, rewards, next_outputs, dones):
            return torch.nn.functional.mse_loss(outputs, actions)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Learn task
        stats = learner.learn_task(self.task_data, dummy_loss, optimizer, num_epochs=2)
        self.assertIsInstance(stats, dict)

        # Check progressive stats
        learner_stats = learner.get_continual_stats()
        self.assertIn("progressive_stats", learner_stats)


class TestContinualLearningIntegration(unittest.TestCase):
    """Test integration of continual learning in UnifiedTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "algorithm": "ppo",
            "total_timesteps": 100,
            "model_name": "test_continual",
            "enable_continual_learning": True,
            "continual_method": "ewc",
            "continual_ewc_lambda": 0.1,
            "continual_buffer_size": 100,
        }

    @patch("ztb.training.unified_trainer.trainer.create_algorithm_trainer")
    def test_continual_learning_setup(self, mock_create_trainer):
        """Test continual learning setup in UnifiedTrainer."""
        from ztb.training.unified_trainer.trainer import UnifiedTrainer

        # Mock algorithm trainer
        mock_trainer = Mock()
        mock_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))
        mock_trainer.model = mock_model
        mock_trainer.train.return_value = True
        mock_create_trainer.return_value = mock_trainer

        trainer = UnifiedTrainer(self.config, dry_run=True)
        trainer.algorithm_trainer = mock_trainer  # Set algorithm trainer for testing

        # Test setup
        trainer._setup_advanced_features()

        # Check if continual learner was initialized
        self.assertIsNotNone(trainer.continual_learner)

    def test_config_validation(self):
        """Test configuration validation for continual learning."""
        from ztb.training.unified_trainer.config import UnifiedTrainerConfig

        config = UnifiedTrainerConfig(
            algorithm=UnifiedAlgorithm.PPO,
            enable_continual_learning=True,
            continual_method="ewc",
            continual_ewc_lambda=0.1,
            continual_buffer_size=1000,
        )

        # Check config attributes
        self.assertTrue(config.enable_continual_learning)
        self.assertEqual(config.continual_method, "ewc")
        self.assertEqual(config.continual_ewc_lambda, 0.1)
