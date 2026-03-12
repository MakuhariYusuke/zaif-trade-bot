#!/usr/bin/env python3
"""
Tests for analyze_results method additions to PPOTrainer and SACTrainer.
"""

from unittest.mock import Mock

import pytest

from ztb.training.unified_trainer.algorithms.ppo_trainer import PPOTrainer
from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback


class TestAnalyzeResultsMethods:
    """Test analyze_results methods added to trainers."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = {
            "training": {
                "ppo_hyperparameters": {
                    "learning_rate": 0.0003,
                    "n_steps": 2048,
                    "batch_size": 64,
                }
            },
            "model_name": "test_model",
        }
        return config

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    def test_ppo_trainer_has_analyze_results(self, mock_config, mock_logger):
        """Test that PPOTrainer has analyze_results method."""
        trainer = PPOTrainer(config=mock_config, logger=mock_logger)

        # Check that method exists
        assert hasattr(trainer, "analyze_results")
        assert callable(getattr(trainer, "analyze_results"))

    def test_sac_trainer_has_analyze_results(self, mock_config, mock_logger):
        """Test that SACTrainer has analyze_results method."""
        trainer = SACTrainer(config=mock_config, logger=mock_logger)

        # Check that method exists
        assert hasattr(trainer, "analyze_results")
        assert callable(getattr(trainer, "analyze_results"))

    def test_ppo_analyze_results_structure(self, mock_config, mock_logger):
        """Test that PPO analyze_results returns correct structure."""
        trainer = PPOTrainer(config=mock_config, logger=mock_logger)

        # Mock training stats
        trainer.training_stats = {
            "total_steps": 1000,
            "training_time": 60.0,
        }

        # Mock callback with action data
        mock_callback = Mock(spec=TrainingProgressCallback)
        mock_callback.discrete_actions = [0, 1, -1, 0, 1]  # HOLD, BUY, SELL, HOLD, BUY
        trainer.training_stats["callback"] = mock_callback

        result = trainer.analyze_results()

        # Check result structure
        assert isinstance(result, dict)
        assert "algorithm" in result
        assert result["algorithm"] == "PPO"
        assert "final_action_distribution" in result
        assert "regime_distributions" in result
        assert "total_training_steps" in result
        assert "training_time" in result

        # Check action distribution calculation
        action_dist = result["final_action_distribution"]
        assert "HOLD" in action_dist
        assert "BUY" in action_dist
        assert "SELL" in action_dist

        # With 5 actions: 2 HOLD, 2 BUY, 1 SELL
        expected_hold = 2 / 5  # 40%
        expected_buy = 2 / 5  # 40%
        expected_sell = 1 / 5  # 20%

        assert abs(action_dist["HOLD"] - expected_hold) < 0.001
        assert abs(action_dist["BUY"] - expected_buy) < 0.001
        assert abs(action_dist["SELL"] - expected_sell) < 0.001

    def test_sac_analyze_results_structure(self, mock_config, mock_logger):
        """Test that SAC analyze_results returns correct structure."""
        trainer = SACTrainer(config=mock_config, logger=mock_logger)

        # Mock training stats
        trainer.training_stats = {
            "total_steps": 2000,
            "training_time": 120.0,
        }

        # Mock callback with continuous action data
        mock_callback = Mock(spec=TrainingProgressCallback)
        mock_callback.continuous_actions = [
            0.0,
            0.5,
            -0.5,
            0.8,
            -0.8,
        ]  # Various continuous actions
        # SAC should convert these to discrete actions: HOLD, BUY, SELL, BUY, SELL
        mock_callback.discrete_actions = [0, 1, 2, 1, 2]
        trainer.training_stats["callback"] = mock_callback

        result = trainer.analyze_results()

        # Check result structure
        assert isinstance(result, dict)
        assert "algorithm" in result
        assert result["algorithm"] == "SAC"
        assert "final_action_distribution" in result
        assert "regime_distributions" in result
        assert "total_training_steps" in result
        assert "training_time" in result

        # Check action distribution calculation
        action_dist = result["final_action_distribution"]
        assert "HOLD" in action_dist
        assert "BUY" in action_dist
        assert "SELL" in action_dist

    def test_analyze_results_without_training_stats(self, mock_config, mock_logger):
        """Test analyze_results when training_stats is not available."""
        trainer = PPOTrainer(config=mock_config, logger=mock_logger)

        # No training stats
        result = trainer.analyze_results()

        # Should return basic structure with None/default values
        assert isinstance(result, dict)
        assert "algorithm" in result
        assert "final_action_distribution" in result
        assert "total_training_steps" in result
        assert "training_time" in result

        # Action distribution should be zeros
        action_dist = result["final_action_distribution"]
        assert action_dist["HOLD"] == 0.0
        assert action_dist["BUY"] == 0.0
        assert action_dist["SELL"] == 0.0

    def test_analyze_results_with_regime_data(self, mock_config, mock_logger):
        """Test analyze_results with regime-specific action data."""
        trainer = PPOTrainer(config=mock_config, logger=mock_logger)

        # Mock training stats with regime data
        mock_callback = Mock()
        mock_callback.discrete_actions = [1, 2, 0]  # BUY, SELL, HOLD
        mock_callback.regime_action_counts = {
            "bull": [2, 0, 1],  # BUY:2, SELL:0, HOLD:1
            "bear": [1, 1, 0],  # BUY:1, SELL:1, HOLD:0
        }
        trainer.training_stats = {"callback": mock_callback}

        result = trainer.analyze_results()

        # Check regime distributions
        regime_dist = result["regime_distributions"]
        assert "bull" in regime_dist
        assert "bear" in regime_dist

        # Bull market: 3 total actions (2+0+1)
        bull_dist = regime_dist["bull"]
        assert bull_dist["HOLD"] == 1 / 3  # 1/3 ≈ 0.333
        assert bull_dist["BUY"] == 2 / 3  # 2/3 ≈ 0.667
        assert bull_dist["SELL"] == 0 / 3  # 0/3 = 0.0
        assert bull_dist["total_actions"] == 3

        # Bear market: 2 total actions (1+1+0)
        bear_dist = regime_dist["bear"]
        assert bear_dist["HOLD"] == 0 / 2  # 0/2 = 0.0
        assert bear_dist["BUY"] == 1 / 2  # 1/2 = 0.5
        assert bear_dist["SELL"] == 1 / 2  # 1/2 = 0.5
        assert bear_dist["total_actions"] == 2

    def test_analyze_results_error_handling(self, mock_config, mock_logger):
        """Test that analyze_results handles errors gracefully."""
        trainer = PPOTrainer(config=mock_config, logger=mock_logger)

        # Mock training stats that will cause errors
        trainer.training_stats = {"callback": None}  # This might cause issues

        result = trainer.analyze_results()

        # Should not crash, should return error info or basic structure
        assert isinstance(result, dict)
        # Either returns normal result or error dict
        assert "algorithm" in result or "error" in result


if __name__ == "__main__":
    pytest.main([__file__])
