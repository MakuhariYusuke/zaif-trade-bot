"""Unit tests for evaluation module"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.evaluation.evaluate import TradingEvaluator


class TestTradingEvaluator:
    """Test cases for TradingEvaluator class"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample trading data for testing"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        data = {
            "timestamp": dates,
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(105, 115, 100),
            "low": np.random.uniform(95, 105, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for evaluator"""
        return {
            "results_dir": "./test_results/",
            "n_eval_episodes": 2,
            "max_steps_per_episode": 10,
            "deterministic": True,
        }

    @patch("ztb.evaluation.evaluate.PPO")
    @patch("ztb.evaluation.evaluate.HeavyTradingEnv")
    @patch("ztb.evaluation.evaluate.SummaryWriter")
    def test_evaluator_initialization(
        self,
        mock_writer: Mock,
        mock_env: Mock,
        mock_ppo: Mock,
        sample_data: pd.DataFrame,
        sample_config: Dict[str, Any],
    ) -> None:
        """Test evaluator initialization"""
        # Create temporary files
        data_path = Path("./test_data.csv")
        model_path = Path("./test_model.zip")

        sample_data.to_csv(data_path, index=False)

        # Mock model loading
        mock_ppo.load.return_value = Mock()

        try:
            evaluator = TradingEvaluator(str(model_path), str(data_path), sample_config)

            assert evaluator.data_path == data_path
            assert evaluator.model_path == model_path
            assert evaluator.config == sample_config
            assert hasattr(evaluator, "env")
            assert hasattr(evaluator, "writer")

        finally:
            # Cleanup
            if data_path.exists():
                data_path.unlink()
            if model_path.exists():
                model_path.unlink()

    def test_calculate_sharpe_ratio(self, sample_config: Dict[str, Any]) -> None:
        """Test Sharpe ratio calculation"""
        evaluator = TradingEvaluator.__new__(
            TradingEvaluator
        )  # Create without __init__

        # Test with normal returns
        returns = [0.01, 0.02, -0.01, 0.03, -0.02]
        sharpe = evaluator._calculate_sharpe_ratio(returns)  # type: ignore
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)

        # Test with empty returns
        sharpe_empty = evaluator._calculate_sharpe_ratio([])  # type: ignore
        assert sharpe_empty == 0.0

        # Test with constant returns
        returns_constant = [0.01] * 5
        sharpe_constant = evaluator._calculate_sharpe_ratio(returns_constant)  # type: ignore
        assert isinstance(sharpe_constant, float)

    def test_calculate_sortino_ratio(self, sample_config: Dict[str, Any]) -> None:
        """Test Sortino ratio calculation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        returns = [0.01, 0.02, -0.01, 0.03, -0.02]
        sortino = evaluator._calculate_sortino_ratio(returns)  # type: ignore
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_calculate_max_drawdown(self, sample_config: Dict[str, Any]) -> None:
        """Test maximum drawdown calculation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Test with declining returns
        returns = [0.1, 0.05, 0.08, 0.02, 0.01]
        mdd = evaluator._calculate_max_drawdown(returns)  # type: ignore
        assert isinstance(mdd, float)
        assert mdd >= 0  # Should be positive

        # Test with empty returns
        mdd_empty = evaluator._calculate_max_drawdown([])  # type: ignore
        assert mdd_empty == 0.0

    def test_calculate_profit_factor(self, sample_config: Dict[str, Any]) -> None:
        """Test profit factor calculation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        pnls = [0.01, -0.005, 0.02, -0.01, 0.03]
        pf = evaluator._calculate_profit_factor(pnls)  # type: ignore
        assert isinstance(pf, float)
        assert pf > 0

        # Test with no losses
        pnls_profit_only = [0.01, 0.02, 0.03]
        pf_profit = evaluator._calculate_profit_factor(pnls_profit_only)  # type: ignore
        assert pf_profit == float("inf")

        # Test with no profits
        pnls_loss_only = [-0.01, -0.02, -0.03]
        pf_loss = evaluator._calculate_profit_factor(pnls_loss_only)  # type: ignore
        assert pf_loss == 0.0

    def test_calculate_hold_ratio_penalty(self, sample_config: Dict[str, Any]) -> None:
        """Test hold ratio penalty calculation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Test with high hold ratio
        actions = [[0] * 10, [0] * 9 + [1]]  # 100% and 90% hold -> avg 95%
        penalty = evaluator._calculate_hold_ratio_penalty(actions)  # type: ignore
        assert isinstance(penalty, float)
        assert penalty > 0

        # Test with low hold ratio
        actions_low = [[1, 2, 1, 2, 1], [1, 2, 1, 2, 2]]  # 0% hold
        penalty_low = evaluator._calculate_hold_ratio_penalty(actions_low)  # type: ignore
        assert penalty_low == 0.0

    def test_calculate_data_quality_score(self, sample_config: Dict[str, Any]) -> None:
        """Test data quality score calculation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Mock stats
        stats = {
            "reward_stats": {
                "std_total_reward": 0.1,
                "mean_total_reward": 0.5,
                "sharpe_ratio": 1.5,
            },
            "trading_stats": {
                "win_rate": 0.6,
                "hold_ratio_penalty": 0.1,
                "profit_factor": 1.2,
            },
        }

        score = evaluator._calculate_data_quality_score(stats)  # type: ignore
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @patch("ztb.evaluation.evaluate.plt")
    def test_create_reward_analysis_plot(
        self, mock_plt: Mock, sample_config: Dict[str, Any]
    ) -> None:
        """Test reward analysis plot creation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)
        evaluator.results_dir = Path("./test_results/")

        # Mock plt.subplots
        mock_fig = Mock()
        mock_axes = [[Mock() for _ in range(2)] for _ in range(2)]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        stats = {
            "episode_rewards": [0.1, 0.2, -0.1, 0.3],
            "episode_lengths": [10, 15, 8, 12],
            "reward_stats": {
                "mean_total_reward": 0.125,
                "std_total_reward": 0.15,
            },
        }

        # Should not raise exception
        evaluator._create_reward_analysis_plot(stats)  # type: ignore

        # Verify plot calls
        assert mock_plt.subplots.called
        assert mock_plt.tight_layout.called
        assert mock_plt.savefig.called

    @patch("ztb.evaluation.evaluate.plt")
    def test_create_pnl_analysis_plot(
        self, mock_plt: Mock, sample_config: Dict[str, Any]
    ) -> None:
        """Test PnL analysis plot creation"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)
        evaluator.results_dir = Path("./test_results/")

        # Mock plt.subplots
        mock_fig = Mock()
        mock_axes = [[Mock() for _ in range(2)] for _ in range(2)]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        stats = {
            "episode_pnls": [0.01, 0.02, -0.01, 0.03],
            "pnl_stats": {
                "mean_total_pnl": 0.0125,
                "sharpe_ratio": 1.2,
            },
        }

        evaluator._create_pnl_analysis_plot(stats)  # type: ignore

        assert mock_plt.subplots.called
        assert mock_plt.savefig.called
