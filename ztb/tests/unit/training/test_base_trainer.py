"""
Unit tests for base_trainer.py module.

This module provides comprehensive testing for the BaseTrainer abstract class
and related protocols and utilities.
"""

from unittest.mock import Mock, patch

from ztb.training.config.trainer_params import TrainerParams
from ztb.training.core.base_trainer import BaseTrainer


class TestBaseTrainer:
    """Test cases for BaseTrainer abstract class."""

    def test_trainer_params_initialization(self):
        """Test TrainerParams initialization with valid config."""
        from ztb.training.config.ppo_config import PPOConfig

        config = PPOConfig()
        params = TrainerParams(
            data_path="/path/to/data.csv",
            config=config,
            checkpoint_dir="/path/to/checkpoints",
        )

        assert params.data_path == "/path/to/data.csv"
        assert params.config == config
        assert params.checkpoint_dir == "/path/to/checkpoints"
        assert params.checkpoint_interval == 10000

    def test_trainer_params_defaults(self):
        """Test TrainerParams with default values."""
        from ztb.training.config.ppo_config import PPOConfig

        config = PPOConfig()
        params = TrainerParams(
            data_path="/path/to/data.csv",
            config=config,
            checkpoint_dir="/path/to/checkpoints",
        )

        assert params.data_path == "/path/to/data.csv"
        assert params.config == config
        assert params.checkpoint_dir == "/path/to/checkpoints"
        assert params.checkpoint_interval == 10000

    @patch("ztb.training.base_trainer.EvalGates")
    @patch("ztb.training.base_trainer.get_logger")
    def test_base_trainer_initialization(self, mock_logger, mock_eval_gates):
        """Test BaseTrainer initialization."""
        from ztb.training.config.ppo_config import PPOConfig

        # Create a concrete implementation for testing
        class ConcreteTrainer(BaseTrainer):
            def train(self, session_id: str):
                return Mock()

            def get_reward_stats(self):
                return {"mean_reward": 1.0}

            def _create_callback(self):
                return Mock()

        config = PPOConfig()
        params = TrainerParams(
            data_path="/path/to/data.csv",
            config=config,
            checkpoint_dir="/path/to/checkpoints",
        )

        trainer = ConcreteTrainer(params)

        assert trainer.data_path == "/path/to/data.csv"
        assert trainer.checkpoint_dir.name == "checkpoints"
        assert trainer.checkpoint_interval == 10000

    @patch("ztb.training.base_trainer.EvalGates")
    def test_evaluation_gate_checking(self, mock_eval_gates):
        """Test evaluation gate checking functionality."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, session_id: str):
                return Mock()

            def get_reward_stats(self):
                return {"mean_reward": 1.0}

        config = {}
        trainer = ConcreteTrainer(config)

        # Mock eval gates
        mock_gates_instance = Mock()
        mock_gates_instance.check_gates.return_value = Mock(
            passed=True, status="success"
        )
        mock_eval_gates.return_value = mock_gates_instance

        # Test gate checking
        result = trainer._check_evaluation_gates(1000, {"mean_reward": 2.0})

        assert result.passed is True
        mock_gates_instance.check_gates.assert_called_once_with(
            1000, {"mean_reward": 2.0}
        )

    def test_reward_stats_tracking(self):
        """Test reward statistics tracking."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, session_id: str):
                return Mock()

            def get_reward_stats(self):
                return {"mean_reward": 1.5, "std_reward": 0.5}

        config = {}
        trainer = ConcreteTrainer(config)

        # Test initial state
        assert len(trainer.reward_history) == 0

        # Add some rewards
        trainer._update_reward_stats(1.0)
        trainer._update_reward_stats(2.0)
        trainer._update_reward_stats(3.0)

        assert len(trainer.reward_history) == 3
        assert trainer.reward_history[0] == 1.0
        assert trainer.reward_history[1] == 2.0
        assert trainer.reward_history[2] == 3.0

    def test_checkpoint_management(self):
        """Test checkpoint saving and loading."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, session_id: str):
                return Mock()

            def get_reward_stats(self):
                return {"mean_reward": 1.0}

        config = {"checkpoint_freq": 1000}
        trainer = ConcreteTrainer(config)

        # Mock checkpoint manager
        with patch.object(trainer, "checkpoint_manager") as mock_cm:
            mock_model = Mock()
            trainer._save_checkpoint(mock_model, 5000, "test_session")

            mock_cm.save_checkpoint.assert_called_once_with(
                mock_model, 5000, "test_session"
            )

    def test_progress_tracking(self):
        """Test training progress tracking."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, session_id: str):
                return Mock()

            def get_reward_stats(self):
                return {"mean_reward": 1.0}

        config = {}
        trainer = ConcreteTrainer(config)

        # Test progress updates
        trainer._update_progress(100, 1000)
        trainer._update_progress(500, 1000)
        trainer._update_progress(1000, 1000)

        # Progress should be tracked (exact implementation depends on progress tracker)
        assert trainer.current_timestep == 1000


class TestTrainerProtocol:
    """Test cases for TrainerProtocol."""

    def test_protocol_compliance(self):
        """Test that classes can implement TrainerProtocol."""

        class MockTrainer:
            def train(self, session_id: str):
                return "trained_model"

            def get_reward_stats(self):
                return {"reward": 1.0}

        trainer = MockTrainer()

        # Test protocol methods exist
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "get_reward_stats")

        # Test method signatures
        result = trainer.train("test_session")
        assert result == "trained_model"

        stats = trainer.get_reward_stats()
        assert stats == {"reward": 1.0}


class TestStatisticsTracker:
    """Test cases for StatisticsTracker protocol."""

    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        from ztb.types.generics import StatisticsTracker

        class MockStatsTracker(StatisticsTracker):
            def __init__(self):
                self.stats = {}

            def update(self, key: str, value: float):
                self.stats[key] = value

            def get(self, key: str) -> float:
                return self.stats.get(key, 0.0)

            def get_all(self) -> Dict[str, float]:
                return self.stats.copy()

        tracker = MockStatsTracker()

        # Test updating and retrieving stats
        tracker.update("mean_reward", 1.5)
        tracker.update("std_reward", 0.3)

        assert tracker.get("mean_reward") == 1.5
        assert tracker.get("std_reward") == 0.3
        assert tracker.get("nonexistent") == 0.0

        all_stats = tracker.get_all()
        assert all_stats == {"mean_reward": 1.5, "std_reward": 0.3}
