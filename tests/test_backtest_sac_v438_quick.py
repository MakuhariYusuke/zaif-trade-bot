"""
Tests for backtest_sac_v438_quick.py

This module contains unit tests for the SAC v438.1 backtest functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add project root to path for testing
project_root = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from scripts.backtest.backtest_sac_v438_quick import (
    backtest_sac_v438_quick,
    calculate_backtest_summary,
    run_quick_backtest,
)


class TestBacktestSACv438Quick:
    """Test cases for SAC v438.1 backtest functionality."""

    def test_calculate_backtest_summary_basic(self):
        """Test basic summary calculation with valid data."""
        # Create mock dataframes
        results_df = pd.DataFrame(
            {
                "total_reward": [100, 200, 150],
                "total_trades": [5, 8, 6],
                "final_portfolio_value": [201000, 202000, 201500],
                "total_steps": [100, 100, 100],
                "avg_reward_per_step": [1.0, 2.0, 1.5],
                "trades_per_step": [0.05, 0.08, 0.06],
            }
        )

        portfolio_df = pd.DataFrame(
            {
                "step": [1, 2, 3, 1, 2, 3],
                "portfolio_value": [200000, 200100, 200200, 200000, 200200, 200150],
            }
        )

        trades_df = pd.DataFrame(
            {
                "episode": [1, 1, 2, 2, 2],
                "step": [10, 20, 15, 25, 35],
                "action": [1, -1, 1, -1, 1],
            }
        )

        summary = calculate_backtest_summary(results_df, portfolio_df, trades_df)

        assert summary["total_episodes"] == 3
        assert summary["avg_total_reward"] == 150.0
        assert summary["total_trades_all_episodes"] == 5
        assert summary["best_episode_reward"] == 200
        assert summary["worst_episode_reward"] == 100
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary

    def test_calculate_backtest_summary_empty_portfolio(self):
        """Test summary calculation with empty portfolio data."""
        results_df = pd.DataFrame(
            {
                "total_reward": [100],
                "total_trades": [5],
                "final_portfolio_value": [201000],
                "total_steps": [100],
                "avg_reward_per_step": [1.0],
                "trades_per_step": [0.05],
            }
        )

        portfolio_df = pd.DataFrame()
        trades_df = pd.DataFrame()

        summary = calculate_backtest_summary(results_df, portfolio_df, trades_df)

        assert summary["total_episodes"] == 1
        assert (
            "max_drawdown" not in summary
        )  # Should not be calculated for empty portfolio

    @patch("scripts.backtest.backtest_sac_v438_quick.Path")
    @patch("scripts.backtest.backtest_sac_v438_quick.SAC.load")
    @patch("scripts.backtest.backtest_sac_v438_quick.HeavyTradingEnv")
    @patch("scripts.backtest.backtest_sac_v438_quick.SACv427FeatureEngineer")
    @patch("scripts.backtest.backtest_sac_v438_quick.pd.read_csv")
    @patch("scripts.backtest.backtest_sac_v438_quick.os.makedirs")
    @patch("scripts.backtest.backtest_sac_v438_quick.logger")
    def test_backtest_sac_v438_quick_model_not_found(
        self,
        mock_logger,
        mock_makedirs,
        mock_read_csv,
        mock_feature_engineer,
        mock_env,
        mock_sac_load,
        mock_path,
    ):
        """Test backtest when model file is not found."""
        # Mock Path.exists() to return False
        mock_path.return_value.exists.return_value = False

        result = backtest_sac_v438_quick(
            model_path="nonexistent_model.zip", data_path="test_data.csv"
        )

        assert result is None
        mock_logger.error.assert_called()

    @patch("scripts.backtest.backtest_sac_v438_quick.Path")
    @patch("scripts.backtest.backtest_sac_v438_quick.SAC.load")
    @patch("scripts.backtest.backtest_sac_v438_quick.HeavyTradingEnv")
    @patch("scripts.backtest.backtest_sac_v438_quick.SACv427FeatureEngineer")
    @patch("scripts.backtest.backtest_sac_v438_quick.pd.read_csv")
    @patch("scripts.backtest.backtest_sac_v438_quick.os.makedirs")
    @patch("scripts.backtest.backtest_sac_v438_quick.logger")
    def test_backtest_sac_v438_quick_data_not_found(
        self,
        mock_logger,
        mock_makedirs,
        mock_read_csv,
        mock_feature_engineer,
        mock_env,
        mock_sac_load,
        mock_path,
    ):
        """Test backtest when data file is not found."""

        # Mock paths - model exists, data doesn't
        def mock_exists(path):
            if "model" in str(path):
                return True
            elif "data" in str(path):
                return False
            return True

        mock_path.return_value.exists.side_effect = mock_exists

        result = backtest_sac_v438_quick(
            model_path="existing_model.zip", data_path="nonexistent_data.csv"
        )

        assert result is None
        mock_logger.error.assert_called()

    @patch("scripts.backtest.backtest_sac_v438_quick.Path")
    @patch("scripts.backtest.backtest_sac_v438_quick.SAC.load")
    @patch("scripts.backtest.backtest_sac_v438_quick.HeavyTradingEnv")
    @patch("scripts.backtest.backtest_sac_v438_quick.SACv427FeatureEngineer")
    @patch("scripts.backtest.backtest_sac_v438_quick.pd.read_csv")
    @patch("scripts.backtest.backtest_sac_v438_quick.os.makedirs")
    @patch("scripts.backtest.backtest_sac_v438_quick.logger")
    @patch("scripts.backtest.backtest_sac_v438_quick.calculate_backtest_summary")
    def test_backtest_sac_v438_quick_success(
        self,
        mock_calc_summary,
        mock_logger,
        mock_makedirs,
        mock_read_csv,
        mock_feature_engineer,
        mock_env,
        mock_sac_load,
        mock_path,
    ):
        """Test successful backtest execution."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True

        # Mock data loading
        mock_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_read_csv.return_value = mock_df

        # Mock feature engineering
        mock_features_df = pd.DataFrame({"feature1": [0.1, 0.2, 0.3]})
        mock_feature_engineer.return_value.generate_v427_features.return_value = (
            mock_features_df
        )

        # Mock environment
        mock_env_instance = Mock()
        mock_env.return_value = mock_env_instance
        mock_env_instance.reset.return_value = (Mock(), {})
        mock_env_instance.step.return_value = (
            Mock(),
            10.0,
            True,
            False,
            {"portfolio_value": 200100, "trade_executed": True},
        )

        # Mock model
        mock_model = Mock()
        mock_sac_load.return_value = mock_model
        mock_model.predict.return_value = (Mock(), None)

        # Mock summary calculation
        mock_calc_summary.return_value = {"total_episodes": 1, "avg_total_reward": 10.0}

        with patch("builtins.open", create=True):
            result = backtest_sac_v438_quick(
                model_path="test_model.zip", data_path="test_data.csv", n_episodes=1
            )

        assert result is not None
        assert result["total_episodes"] == 1
        mock_logger.info.assert_called()
        mock_calc_summary.assert_called_once()

    def test_run_quick_backtest(self):
        """Test the convenience function for quick backtest."""
        with patch(
            "scripts.backtest.backtest_sac_v438_quick.backtest_sac_v438_quick"
        ) as mock_backtest:
            mock_backtest.return_value = {"success": True}

            result = run_quick_backtest()

            mock_backtest.assert_called_once_with(
                model_path="checkpoints/sac_v438_production_150000_steps.zip",
                data_path="data/btc_jpy_real_dataset.csv",
                output_dir="backtest_experiments/v438.1",
                n_episodes=3,
                deterministic=True,
            )
            assert result == {"success": True}


if __name__ == "__main__":
    pytest.main([__file__])
