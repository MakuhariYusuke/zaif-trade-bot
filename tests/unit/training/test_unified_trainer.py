
"""
Unified Trainer unit tests

Tests for UnifiedTrainer class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from ztb.training.unified_trainer.trainer import UnifiedTrainer


class TestUnifiedTrainer:
    """UnifiedTrainer tests"""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration"""
        return {
            "algorithm": "sac",
            "env_name": "HeavyTradingEnv",
            "total_timesteps": 100000,
            "batch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "target_entropy": "auto",
            "buffer_size": 1000000,
            "learning_starts": 1000,
            "train_freq": 1,
            "gradient_steps": 1,
            "verbose": 1,
            "device": "cpu"
        }

    @pytest.fixture
    def trainer(self, sample_config):
        """Trainer fixture"""
        return UnifiedTrainer(sample_config)

    @pytest.fixture
    def sample_backtest_data(self):
        """Sample backtest data"""
        return {
            "period_1": {
                "start_date": "2023-01-01",
                "end_date": "2023-01-31",
                "data": pd.DataFrame({
                    "timestamp": pd.date_range("2023-01-01", periods=100, freq="1H"),
                    "price": np.random.randn(100).cumsum() + 100,
                    "volume": np.random.randint(100, 1000, 100)
                })
            },
            "period_2": {
                "start_date": "2023-02-01",
                "end_date": "2023-02-28",
                "data": pd.DataFrame({
                    "timestamp": pd.date_range("2023-02-01", periods=100, freq="1H"),
                    "price": np.random.randn(100).cumsum() + 100,
                    "volume": np.random.randint(100, 1000, 100)
                })
            }
        }

    def test_initialization(self, trainer):
        """Initialization test"""
        assert trainer.config is not None
        assert trainer.algorithm == "sac"
        assert trainer.model is None

    def test_run_multi_period_backtest(self, trainer, sample_backtest_data):
        """Multi-period backtest execution test"""
        periods = [
            {
                "name": "period_1",
                "start_date": "2023-01-01",
                "end_date": "2023-01-31"
            },
            {
                "name": "period_2",
                "start_date": "2023-02-01",
                "end_date": "2023-02-28"
            }
        ]

        df = sample_backtest_data["period_1"]["data"]

        with (
            patch.object(trainer, "_create_backtest_environment", return_value=Mock()) as mock_env,
            patch.object(trainer, "_load_backtest_data", return_value=df),
            patch.object(
                trainer,
                "_run_single_period_backtest",
                return_value={"total_return_pct": 5.0, "total_actions": 100},
            ),
            patch.object(trainer, "_calculate_overall_backtest_metrics") as mock_overall,
            patch.object(trainer, "_analyze_backtest_regime_performance") as mock_regime,
            patch.object(trainer, "_generate_backtest_recommendations") as mock_recommend,
        ):

            mock_overall.return_value = {
                "total_periods": 2,
                "average_return": 0.04,
                "total_trades": 180
            }
            mock_regime.return_value = {
                "bull": {"average_return": 0.07}
            }
            mock_recommend.return_value = ["Recommendation 1", "Recommendation 2"]

            results = trainer.run_multi_period_backtest(periods)

            # Verify methods were called
            mock_env.assert_called()
            assert mock_overall.called
            assert mock_regime.called
            assert mock_recommend.called

            assert "period_results" in results
            assert "overall_metrics" in results
            assert "regime_performance" in results
            assert "recommendations" in results

    def test_create_backtest_environment(self, trainer):
        """Backtest environment creation test"""
        with patch.object(trainer, "_create_training_environment") as mock_create:
            mock_env = Mock()
            mock_create.return_value = mock_env

            env = trainer._create_backtest_environment()

            mock_create.assert_called_once()
            assert env == mock_env

    def test_load_backtest_data(self, trainer):
        """Backtest data loading test"""
        trainer.config.setdefault("training", {})["data_config"] = {"data_path": "dummy.csv"}
        mock_df = pd.DataFrame({"timestamp": ["2023-01-01T00:00:00"], "price": [100.0]})

        with patch(
            "ztb.training.unified_trainer.trainer.DataLoader.load_csv_optimized",
            return_value=mock_df.copy(),
        ) as mock_load:
            data = trainer._load_backtest_data()

            mock_load.assert_called_once_with("dummy.csv")
            assert not data.empty
            assert "timestamp" in data.columns

    def test_run_single_period_backtest(self, trainer):
        """Single period backtest execution test"""
        period_data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="1H"),
            "price": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "volume": [1000] * 10
        })

        mock_env = Mock()
        trainer.model = Mock()

        with patch.object(
            trainer,
            "_test_period_with_model",
            return_value={"period_name": "period_1", "total_return_pct": 5.0},
        ) as mock_test:
            results = trainer._run_single_period_backtest(
                trainer.model,
                mock_env,
                period_data,
                {
                    "name": "period_1",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-01 09:00:00",
                },
            )

            assert results["period_name"] == "period_1"
            assert results["total_return_pct"] == 5.0
            mock_test.assert_called_once()

    def test_calculate_overall_backtest_metrics(self, trainer):
        """Overall backtest metrics calculation test"""
        period_results = [
            {
                "total_return_pct": 5.0,
                "total_actions": 100,
            },
            {
                "total_return_pct": -1.0,
                "total_actions": 80,
            }
        ]

        metrics = trainer._calculate_overall_backtest_metrics(period_results)

        assert "total_periods" in metrics
        assert "average_return" in metrics
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert metrics["total_periods"] == 2
        assert abs(metrics["average_return"] - 2.0) < 0.001
        assert metrics["total_trades"] == 180
        assert abs(metrics["win_rate"] - 50.0) < 0.001

    def test_analyze_backtest_regime_performance(self, trainer):
        """Backtest regime performance analysis test"""
        period_results = [
            {
                "performance_by_regime": {
                    "bull": {"return": 0.08, "win_rate": 0.7},
                    "bear": {"return": -0.02, "win_rate": 0.4}
                }
            },
            {
                "performance_by_regime": {
                    "bull": {"return": 0.06, "win_rate": 0.65},
                    "bear": {"return": -0.01, "win_rate": 0.45}
                }
            }
        ]

        regime_perf = trainer._analyze_backtest_regime_performance(period_results)

        assert "bull_market_performance" in regime_perf
        assert "bear_market_performance" in regime_perf
        assert regime_perf["bull_market_performance"]["average_return"] == 0.0
        assert regime_perf["bear_market_performance"]["win_rate"] == 0.0

    def test_generate_backtest_recommendations(self, trainer):
        """Backtest recommendations generation test"""
        results = {
            "overall_metrics": {
                "win_rate": 65.0,
                "average_return": 0.04
            },
            "regime_performance": {
                "bull": {"average_win_rate": 0.7},
                "bear": {"average_win_rate": 0.3}
            }
        }

        recommendations = trainer._generate_backtest_recommendations(results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("Strong overall performance" in rec for rec in recommendations)

    def test_error_handling(self, trainer):
        """Error handling test"""
        # Invalid period data
        periods = []

        results = trainer.run_multi_period_backtest(periods)

        assert "error" in results or results.get("period_results", []) == []

        # Empty period results
        metrics = trainer._calculate_overall_backtest_metrics([])
        assert metrics == {}
