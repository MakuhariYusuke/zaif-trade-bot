
"""
V4XX Unified Analyzer unit tests

Tests for V4XXUnifiedAnalyzer class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import json

from ztb.analysis.v4xx_unified_analyzer import V4XXUnifiedAnalyzer


class TestV4XXUnifiedAnalyzer:
    """V4XXUnifiedAnalyzer tests"""

    @pytest.fixture
    def sample_backtest_results(self):
        """Sample backtest results"""
        return {
            "summary": {
                "total_episodes": 100,
                "average_reward": 150.5,
                "average_trades": 25,
                "win_rate": 0.65,
                "total_return": 0.085,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.12
            },
            "episodes": [
                {
                    "episode": 1,
                    "reward": 200,
                    "trades": 30,
                    "win_rate": 0.7,
                    "return": 0.15
                },
                {
                    "episode": 2,
                    "reward": 100,
                    "trades": 20,
                    "win_rate": 0.6,
                    "return": 0.02
                }
            ]
        }

    @pytest.fixture
    def temp_results_file(self, sample_backtest_results, tmp_path: Path):
        """Temporary results file"""
        temp_path = tmp_path / "backtest_results.json"
        temp_path.write_text(json.dumps(sample_backtest_results), encoding="utf-8")

        yield str(temp_path)

    @pytest.fixture
    def analyzer(self, temp_results_file):
        """Analyzer fixture"""
        return V4XXUnifiedAnalyzer(temp_results_file)

    def test_initialization(self, analyzer):
        """Initialization test"""
        assert analyzer.results_path.exists()
        assert analyzer.version is not None
        assert isinstance(analyzer.metrics, dict)

    def test_calculate_basic_metrics(self, analyzer):
        """Basic metrics calculation test"""
        metrics = analyzer.calculate_basic_metrics()

        assert "total_episodes" in metrics
        assert "average_reward" in metrics
        assert "win_rate" in metrics
        assert metrics["total_episodes"] == 100
        assert metrics["average_reward"] == 150.5
        assert metrics["win_rate"] == 0.65

    def test_analyze_multi_period_backtest(self, analyzer):
        """Multi-period backtest analysis test"""
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

        # Mock to avoid actual backtest
        with patch.object(analyzer, '_analyze_single_period', return_value={
            "period_name": "mock_period",
            "metrics": {
                "total_return": 0.05,
                "win_rate": 0.6,
                "total_trades": 100
            },
            "performance_by_regime": {
                "bull": {"return": 0.08, "win_rate": 0.7}
            }
        }):
            results = analyzer.analyze_multi_period_backtest(periods)

        assert "period_analysis" in results
        assert "overall_metrics" in results
        assert "regime_performance" in results
        assert "recommendations" in results
        assert len(results["period_analysis"]) == 2

    def test_calculate_overall_metrics(self, analyzer):
        """Overall metrics calculation test"""
        period_results = [
            {
                "metrics": {
                    "total_return": 0.05,
                    "total_trades": 100
                }
            },
            {
                "metrics": {
                    "total_return": 0.03,
                    "total_trades": 80
                }
            }
        ]

        metrics = analyzer._calculate_overall_metrics(period_results)

        assert "total_periods" in metrics
        assert "average_return" in metrics
        assert "total_trades" in metrics
        assert metrics["total_periods"] == 2
        assert abs(metrics["average_return"] - 0.04) < 0.001
        assert metrics["total_trades"] == 180

    def test_analyze_regime_performance(self, analyzer):
        """Regime performance analysis test"""
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

        regime_perf = analyzer._analyze_regime_performance(period_results)

        assert "bull" in regime_perf
        assert "bear" in regime_perf
        assert regime_perf["bull"]["average_return"] == 0.07
        assert regime_perf["bull"]["average_win_rate"] == 0.675

    def test_generate_multi_period_recommendations(self, analyzer):
        """Multi-period recommendations generation test"""
        results = {
            "overall_metrics": {
                "average_win_rate": 0.65
            },
            "regime_performance": {
                "bull": {"average_win_rate": 0.7},
                "bear": {"average_win_rate": 0.3}
            }
        }

        recommendations = analyzer._generate_multi_period_recommendations(results)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should contain recommendations for strong performing regime
        strong_regime_found = any("bull" in rec for rec in recommendations)
        assert strong_regime_found

    def test_error_handling(self, analyzer):
        """Error handling test"""
        # Invalid period data
        periods = []

        results = analyzer.analyze_multi_period_backtest(periods)

        assert "error" not in results or results.get("period_analysis", [])

        # Empty period results
        metrics = analyzer._calculate_overall_metrics([])
        assert metrics == {}
