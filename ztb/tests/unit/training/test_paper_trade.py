"""
Unit tests for paper_trade.py module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from ztb.training.paper_trade import PaperTrader


class TestPaperTrader:
    """Test cases for PaperTrader class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "reward_scaling": 1.0,
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "risk_free_rate": 0.0,
            "initial_portfolio_value": 10000.0,
            "verbose": 0,
        }

    @pytest.fixture
    def sample_test_data(self):
        """Sample test data for testing."""
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.random.randint(100, 1000, 100)
        }
        return pd.DataFrame(data)

    def test_initialization(self, sample_config, sample_test_data):
        """Test PaperTrader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temporary test data file
            test_data_path = Path(tmpdir) / "test_data.csv"
            sample_test_data.to_csv(test_data_path, index=False)

            # Mock model path (doesn't need to exist for init test)
            model_path = Path(tmpdir) / "dummy_model.zip"

            trader = PaperTrader(str(model_path), str(test_data_path), sample_config)

            assert trader.model_path == model_path
            assert trader.test_data_path == test_data_path
            assert trader.config == sample_config
            assert trader.portfolio_value == 10000.0
            assert trader.position == 0.0
            assert trader.trades == []

    def test_get_default_config(self):
        """Test default configuration."""
        trader = PaperTrader("dummy_path", "dummy_data")
        config = trader._get_default_config()

        expected_keys = [
            "reward_scaling", "transaction_cost", "max_position_size",
            "risk_free_rate", "initial_portfolio_value", "verbose"
        ]

        for key in expected_keys:
            assert key in config

        assert config["initial_portfolio_value"] == 10000.0
        assert config["transaction_cost"] == 0.001

    def test_load_test_data(self, sample_test_data):
        """Test loading test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data_path = Path(tmpdir) / "test_data.csv"
            sample_test_data.to_csv(test_data_path, index=False)

            trader = PaperTrader("dummy_model", str(test_data_path))
            trader._load_test_data()

            assert trader.test_df is not None
            assert len(trader.test_df) == 20  # Should use last 20% of data
            assert list(trader.test_df.columns) == list(sample_test_data.columns)

    def test_load_test_data_file_not_found(self):
        """Test loading test data when file doesn't exist."""
        trader = PaperTrader("dummy_model", "nonexistent_file.csv")
        trader._load_test_data()

        assert trader.test_df is None

    @patch('ztb.training.paper_trade.PPO')
    @patch('ztb.training.paper_trade.DummyVecEnv')
    def test_create_env(self, mock_vec_env, mock_ppo, sample_config, sample_test_data):
        """Test environment creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data_path = Path(tmpdir) / "test_data.csv"
            sample_test_data.to_csv(test_data_path, index=False)

            trader = PaperTrader("dummy_model", str(test_data_path), sample_config)
            trader._load_test_data()

            env = trader._create_env()

            assert env is not None
            mock_vec_env.assert_called_once()

    def test_calculate_statistics_empty_trades(self):
        """Test statistics calculation with no trades."""
        trader = PaperTrader("dummy_model", "dummy_data")

        rewards = [1.0, 2.0, 3.0]
        lengths = [10, 20, 30]

        stats = trader._calculate_statistics(rewards, lengths)

        assert stats["episodes"] == 3
        assert stats["mean_reward"] == 2.0
        assert stats["total_trades"] == 0
        assert "win_rate" not in stats

    def test_calculate_statistics_with_trades(self):
        """Test statistics calculation with trades."""
        trader = PaperTrader("dummy_model", "dummy_data")

        # Mock some trades
        trader.trades = [
            {"portfolio_change": 100.0, "action": [0.5]},
            {"portfolio_change": -50.0, "action": [-0.3]},
            {"portfolio_change": 200.0, "action": [0.8]},
        ]

        rewards = [1.0, 2.0, 3.0]
        lengths = [10, 20, 30]

        stats = trader._calculate_statistics(rewards, lengths)

        assert stats["total_trades"] == 3
        assert stats["win_rate"] == 2/3  # 2 profitable trades out of 3
        assert stats["avg_win"] == 150.0  # (100 + 200) / 2
        assert stats["avg_loss"] == -50.0

    @patch('ztb.training.paper_trade.ensure_dir')
    @patch('builtins.open', new_callable=Mock)
    @patch('json.dump')
    def test_save_trade_log(self, mock_json_dump, mock_open, mock_ensure_dir):
        """Test saving trade log."""
        trader = PaperTrader("dummy_model", "dummy_data")
        trader.trades = [{"test": "trade"}]

        stats = {"total_trades": 1, "mean_reward": 1.0}

        trader._save_trade_log(stats)

        # Check that ensure_dir was called
        assert mock_ensure_dir.call_count == 2  # stats and trades files

        # Check that json.dump was called twice (stats and trades)
        assert mock_json_dump.call_count == 2

    def test_simulate_episode_requires_model_and_data(self):
        """Test that simulation requires loaded model and data."""
        trader = PaperTrader("dummy_model", "dummy_data")

        # Test without model
        with pytest.raises(ValueError, match="Model not loaded"):
            trader.simulate_trading(1)

        # Test without data
        trader.model = Mock()
        with pytest.raises(ValueError, match="Test data not loaded"):
            trader.simulate_trading(1)


class TestPaperTraderIntegration:
    """Integration tests for PaperTrader."""

    @patch('ztb.training.paper_trade.PPO.load')
    @patch('ztb.training.paper_trade.DummyVecEnv')
    @patch('ztb.training.paper_trade.HeavyTradingEnv')
    def test_full_simulation_workflow(self, mock_env_class, mock_vec_env, mock_ppo_load, sample_test_data):
        """Test full simulation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup test data
            test_data_path = Path(tmpdir) / "test_data.csv"
            sample_test_data.to_csv(test_data_path, index=False)

            # Mock model
            mock_model = Mock()
            mock_ppo_load.return_value = mock_model

            # Mock environment
            mock_env_instance = Mock()
            mock_env_class.return_value = mock_env_instance
            mock_vec_env_instance = Mock()
            mock_vec_env.return_value = mock_vec_env_instance

            # Mock environment methods
            mock_vec_env_instance.reset.return_value = np.array([[1.0, 2.0, 3.0]])
            mock_vec_env_instance.step.return_value = (
                np.array([[1.1, 2.1, 3.1]]),
                np.array([1.0]),
                np.array([False]),
                {}
            )

            trader = PaperTrader("dummy_model.zip", str(test_data_path))

            # Mock the environment's portfolio_value and position
            mock_env_instance.portfolio_value = 10500.0
            mock_env_instance.position = 0.5

            # Run simulation
            results = trader.simulate_trading(n_episodes=1)

            assert "episodes" in results
            assert "mean_reward" in results
            assert "total_trades" in results
            assert results["episodes"] == 1</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\ztb\tests\unit\training\test_paper_trade.py