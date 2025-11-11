
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
import tempfile

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
    def temp_results_file(self, sample_backtest_results):
        """Temporary results file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_backtest_results, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

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
        # Sample backtest results in the format expected by the analyzer
        backtest_results = {
            "4h_windows": {
                "summary": {
                    "overall": {
                        "total_periods": 10,
                        "avg_return": 0.05,
                        "win_rate": 0.6,
                        "sharpe_ratio": 1.2
                    },
                    "by_trend_type": {
                        "bull": {"return": 0.08, "win_rate": 0.7},
                        "bear": {"return": -0.02, "win_rate": 0.4}
                    }
                }
            },
            "8h_windows": {
                "summary": {
                    "overall": {
                        "total_periods": 8,
                        "avg_return": 0.03,
                        "win_rate": 0.55,
                        "sharpe_ratio": 0.9
                    },
                    "by_trend_type": {
                        "bull": {"return": 0.06, "win_rate": 0.65},
                        "bear": {"return": -0.01, "win_rate": 0.45}
                    }
                }
            }
        }

        results = analyzer.analyze_multi_period_backtest(backtest_results)

        assert "overall_performance" in results
        assert "regime_performance" in results
        assert "timeframe_comparison" in results
        assert "recommendations" in results
        assert "4h" in results["overall_performance"]
        assert "8h" in results["regime_performance"]

    def test_analyze_regime_performance(self, analyzer):
        """Regime performance analysis test"""
        # Sample backtest results
        backtest_results = {
            "4h_windows": {
                "summary": {
                    "by_trend_type": {
                        "bull": {"return": 0.08, "win_rate": 0.7},
                        "bear": {"return": -0.02, "win_rate": 0.4}
                    }
                }
            },
            "8h_windows": {
                "summary": {
                    "by_trend_type": {
                        "bull": {"return": 0.06, "win_rate": 0.65},
                        "bear": {"return": -0.01, "win_rate": 0.45}
                    }
                }
            }
        }

        regime_perf = analyzer._analyze_regime_performance(backtest_results)

        assert "4h" in regime_perf
        assert "8h" in regime_perf
        assert "bull" in regime_perf["4h"]
        assert "bear" in regime_perf["4h"]

    def test_generate_trading_recommendations(self, analyzer):
        """Trading recommendations generation test"""
        analysis = {
            "overall_performance": {
                "4h": {
                    "total_periods": 10,
                    "avg_return": 0.05,
                    "win_rate": 0.6,
                    "sharpe_ratio": 1.2
                }
            },
            "timeframe_comparison": {
                "best_timeframe": {
                    "return": "4h",
                    "win_rate": "4h",
                    "sharpe_ratio": "4h"
                }
            }
        }

        recommendations = analyzer._generate_trading_recommendations(analysis)

        assert "optimal_timeframe" in recommendations
        assert "regime_strategy" in recommendations
        assert "risk_management" in recommendations
        assert "implementation_priority" in recommendations

    def test_error_handling(self, analyzer):
        """Error handling test"""
        # Invalid backtest results data
        invalid_results = {}

        results = analyzer.analyze_multi_period_backtest(invalid_results)

        # Should handle empty results gracefully
        assert "overall_performance" in results
        assert "regime_performance" in results
        assert "timeframe_comparison" in results
        assert "recommendations" in results
