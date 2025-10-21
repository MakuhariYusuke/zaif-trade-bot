import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager


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
    with patch("ztb.utils.data_utils.load_csv_data_optimized", return_value=dummy_df), \
        patch("ztb.training.trainers.sac_trainer.SACAlgorithm.get_default_config", return_value={}), \
        patch("ztb.training.trainers.sac_trainer.HeavyTradingEnv", new=DummyEnv), \
        patch("ztb.training.trainers.sac_trainer.DummyVecEnv", new=lambda fns: fns[0]()):
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
