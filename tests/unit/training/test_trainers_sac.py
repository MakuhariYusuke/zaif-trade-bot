from unittest.mock import MagicMock, patch

from ztb.training.core.config_manager import ConfigManager
from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer


@patch("ztb.training.trainers.sac_trainer.AlgorithmFactory")
def test_sac_trainer_train_creates_and_saves_model(mock_algo_factory, tmp_path):
    # Arrange
    cm = ConfigManager({})
    trainer = SACAlgorithmTrainer(config_manager=cm)

    # Mock data loader to return a small dataframe-like object
    dummy_df = MagicMock()
    dummy_df.__len__.return_value = 10

    # Provide a lightweight DummyEnv to avoid HeavyTradingEnv initialization/type checks
    class DummyObsSpace:
        def __init__(self):
            self.shape = (4,)

    class DummyEnv:
        def __init__(self, df=None, config=None):
            # Accept any df (we mock load_csv to return a MagicMock)
            self.df = df
            self.action_space = "Continuous(1)"
            self.observation_space = DummyObsSpace()

    # Patch data loader and HeavyTradingEnv at their origin modules to avoid heavy initialization
    # Also patch DummyVecEnv used in trainer to avoid stable-baselines3 environment validation
    with patch(
        "ztb.utils.data_utils.load_csv_data_optimized", return_value=dummy_df
    ), patch(
        "ztb.training.trainers.sac_trainer.SACAlgorithm.get_default_config",
        return_value={},
    ), patch("ztb.training.trainers.sac_trainer.HeavyTradingEnv", new=DummyEnv), patch(
        "ztb.training.trainers.sac_trainer.DummyVecEnv", new=lambda fns: fns[0]()
    ):
        # Mock SAC algorithm and model
        mock_algo = MagicMock()
        mock_model = MagicMock()
        mock_algo.create_model.return_value = mock_model
        mock_algo.train.return_value = mock_model
        mock_algo.save.return_value = None
        mock_algo.get_default_config.return_value = {}

        mock_algo_factory.create.return_value = mock_algo

        # Prepare unified_config
        unified_config = {
            "model_name": "test_sac",
            "session_id": "test_session",
            "total_timesteps": 1,
            "sac_hyperparameters": {},
            "data_path": "dummy.csv",
        }

        # Act
        result = trainer.train(unified_config)

        # Assert
        assert result["success"] is True
        assert "model_path" in result
        # Ensure create_model and save called
        mock_algo.create_model.assert_called()
        mock_algo.train.assert_called()
        mock_algo.save.assert_called()


class TestTrainingProgressCallbackDebugLogs:
    """Test TrainingProgressCallback DEBUG log enhancements."""

    def test_debug_log_format_and_content(self, caplog):
        """Test that DEBUG logs contain expected information in correct format."""
        import logging

        import numpy as np

        from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback

        caplog.set_level(logging.DEBUG)

        # Create callback
        callback = TrainingProgressCallback(check_freq=10, verbose=1)

        # Mock trainer
        callback.trainer = MagicMock()
        callback.trainer.policy = MagicMock()
        callback.trainer.policy.action_space = MagicMock()
        callback.trainer.policy.action_space.n = 3  # PPO discrete actions

        # Mock locals data
        mock_locals = {
            "actions": np.array([1]),  # BUY action
            "rewards": np.array([2.5]),
            "infos": [
                {
                    "portfolio_value": 150000.0,
                    "position": 0.8,
                    "pnl": 750.0,
                    "market_regime": "bull_trend",
                }
            ],
        }

        callback.locals = mock_locals
        callback.n_calls = 5

        # Call _on_step to trigger DEBUG logs
        callback._on_step()

        # Check DEBUG logs
        debug_logs = [
            record for record in caplog.records if record.levelname == "DEBUG"
        ]

        # Should have action conversion log
        action_logs = [
            record for record in debug_logs if "action" in record.message.lower()
        ]
        assert len(action_logs) > 0, "Should have action-related DEBUG logs"

        # Should have detailed state log
        state_logs = [
            record
            for record in debug_logs
            if "Step" in record.message and "Action=" in record.message
        ]
        assert len(state_logs) > 0, "Should have detailed state DEBUG logs"

        # Check content of state log
        state_log = state_logs[0].message
        assert "Action=1" in state_log, "Should contain action information"
        assert "Reward=2.5000" in state_log, "Should contain reward information"
        assert "PnL=750.00" in state_log, "Should contain PnL information"
        assert (
            "Portfolio=150000.00" in state_log
        ), "Should contain portfolio information"
        assert "Position=0.8000" in state_log, "Should contain position information"
        assert "Regime=bull_trend" in state_log, "Should contain regime information"

    def test_sac_continuous_action_debug_log(self, caplog):
        """Test DEBUG logs for SAC continuous actions."""
        import logging

        import numpy as np

        from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback

        caplog.set_level(logging.DEBUG)

        # Create callback
        callback = TrainingProgressCallback(check_freq=10, verbose=1)

        # Mock trainer for SAC (no discrete action space)
        callback.trainer = MagicMock()
        callback.trainer.policy = None  # SAC case

        # Mock continuous action data
        mock_locals = {
            "actions": np.array([0.7]),  # Continuous action
            "rewards": np.array([1.2]),
            "infos": [
                {
                    "portfolio_value": 120000.0,
                    "position": -0.3,
                    "pnl": -200.0,
                    "market_regime": "bear_trend",
                }
            ],
        }

        callback.locals = mock_locals
        callback.n_calls = 3

        # Call _on_step
        callback._on_step()

        # Check DEBUG logs
        debug_logs = [
            record for record in caplog.records if record.levelname == "DEBUG"
        ]

        # Should have SAC action conversion log
        sac_logs = [record for record in debug_logs if "SAC action" in record.message]
        assert len(sac_logs) > 0, "Should have SAC action conversion DEBUG logs"

        # Should have detailed state log
        state_logs = [
            record
            for record in debug_logs
            if "Step" in record.message and "Action=" in record.message
        ]
        assert len(state_logs) > 0, "Should have detailed state DEBUG logs"

        # Check content
        state_log = state_logs[0].message
        assert (
            "Action=2" in state_log
        ), "Should contain discrete action (SELL for negative)"
        assert "Reward=1.2000" in state_log, "Should contain reward information"
        assert "PnL=-200.00" in state_log, "Should contain negative PnL"
        assert "Position=-0.3000" in state_log, "Should contain negative position"
        assert "Regime=bear_trend" in state_log, "Should contain bear regime"

    def test_training_metrics_debug_log(self, caplog):
        """Test that training metrics are logged at DEBUG level."""
        import logging

        from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback

        caplog.set_level(logging.DEBUG)

        # Create callback with some mock training metrics
        callback = TrainingProgressCallback(check_freq=10, verbose=1)
        callback.actor_losses = [0.5, 0.4, 0.3]
        callback.critic_losses = [0.8, 0.7, 0.6]
        callback.ent_coefs = [0.9, 0.8, 0.7]
        callback.learning_rates = [0.001, 0.0009, 0.0008]
        callback.n_calls = 10

        # Call _log_progress
        callback._log_progress()

        # Check DEBUG logs
        debug_logs = [
            record for record in caplog.records if record.levelname == "DEBUG"
        ]

        # Should have training metrics log
        metrics_logs = [
            record for record in debug_logs if "Training metrics" in record.message
        ]
        assert len(metrics_logs) > 0, "Should have training metrics DEBUG logs"

        # Check content
        metrics_log = metrics_logs[0].message
        assert "ActorLoss=" in metrics_log, "Should contain actor loss"
        assert "CriticLoss=" in metrics_log, "Should contain critic loss"
        assert "EntCoef=" in metrics_log, "Should contain entropy coefficient"
        assert "LR=" in metrics_log, "Should contain learning rate"
        assert "SPS=" in metrics_log, "Should contain steps per second"


class TestSACTrainerInternalLogs:
    """Test SAC trainer internal logging enhancements."""

    @patch("ztb.training.unified_trainer.algorithms.sac_trainer.time.time")
    def test_sac_training_completion_debug_log(self, mock_time, caplog):
        """Test that SAC training completion logs detailed metrics at DEBUG level."""
        import logging
        from unittest.mock import MagicMock

        from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer

        caplog.set_level(logging.DEBUG)

        # Mock time for consistent timing
        mock_time.side_effect = [1000.0, 1050.0]  # 50 seconds training time

        # Create trainer with minimal config
        config = {
            "training": {"total_timesteps": 1000},
            "checkpoint_interval": 500,
            "checkpoint_dir": "models/checkpoints",
        }

        trainer = SACTrainer(config)

        # Mock the model and its logger
        mock_model = MagicMock()
        mock_logger = MagicMock()
        mock_logger.name_to_value = {
            "train/actor_loss": 0.123,
            "train/critic_loss": 0.456,
            "train/ent_coef": 0.789,
        }
        mock_model.logger = mock_logger

        trainer.model = mock_model

        # Mock callback for training stats
        mock_callback = MagicMock()
        mock_callback.callbacks = [MagicMock()]
        mock_callback.callbacks[0].reward_history = [1.0, 2.0, 3.0]
        mock_callback.reward_history = [1.0, 2.0, 3.0]

        # Call the completion logging method directly
        trainer._log_sac_training_completion(50.0, mock_callback)

        # Check DEBUG logs
        debug_logs = [
            record for record in caplog.records if record.levelname == "DEBUG"
        ]

        # Should have SAC training completion log
        completion_logs = [
            record
            for record in debug_logs
            if "SAC training completed" in record.message
        ]
        assert (
            len(completion_logs) > 0
        ), "Should have SAC training completion DEBUG logs"

        # Check content
        completion_log = completion_logs[0].message
        assert "Time=50.00s" in completion_log, "Should contain training time"
        assert "Steps=1000" in completion_log, "Should contain total steps"
        assert "SPS=20.00" in completion_log, "Should contain steps per second"
        assert (
            "Final ActorLoss=0.1230" in completion_log
        ), "Should contain final actor loss"
        assert "CriticLoss=0.4560" in completion_log, "Should contain final critic loss"
        assert (
            "EntCoef=0.7890" in completion_log
        ), "Should contain final entropy coefficient"

    def test_sac_trainer_uses_checkpoint_callback(self):
        """Test that SAC trainer properly integrates CheckpointCallback."""
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer

        config = {
            "training": {"total_timesteps": 1000},
            "checkpoint_interval": 500,
            "checkpoint_dir": "models/checkpoints",
        }

        trainer = SACTrainer(config)

        with patch(
            "stable_baselines3.common.callbacks.CheckpointCallback"
        ) as mock_checkpoint:
            checkpoint_instance = MagicMock(spec=CheckpointCallback)
            mock_checkpoint.return_value = checkpoint_instance

            callbacks = trainer._setup_callbacks()

        # Should return CallbackList
        assert isinstance(callbacks, CallbackList), "Should return CallbackList"
        assert checkpoint_instance in callbacks.callbacks

        mock_checkpoint.assert_called_once_with(
            save_freq=500,
            save_path="models/checkpoints",
            name_prefix="sac_checkpoint",
            verbose=1,
        )
