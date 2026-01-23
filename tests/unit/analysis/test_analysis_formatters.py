#!/usr/bin/env python3
"""
Unit tests for analysis_formatters.py

Tests for analysis result formatting and summary creation utilities.
"""

import pytest

from ztb.utils.analysis_formatters import create_result_summary


class TestResultSummary:
    """Test result summary creation functions."""

    def test_create_result_summary_empty(self):
        """Test creating summary from empty results dict."""
        results = {}
        summary = create_result_summary(results)

        assert summary == ""

    def test_create_result_summary_single_item(self):
        """Test creating summary from single result item."""
        results = {"accuracy": 0.95}
        summary = create_result_summary(results)

        assert summary == "accuracy: 0.9500"

    def test_create_result_summary_multiple_items(self):
        """Test creating summary from multiple result items."""
        results = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88
        }
        summary = create_result_summary(results)

        expected = "accuracy: 0.9500 | precision: 0.9200 | recall: 0.8800"
        assert summary == expected

    def test_create_result_summary_mixed_types(self):
        """Test creating summary with mixed data types."""
        results = {
            "accuracy": 0.95,
            "epochs": 100,
            "model_name": "ppo_v1",
            "converged": True
        }
        summary = create_result_summary(results)

        expected = "accuracy: 0.9500 | epochs: 100 | model_name: ppo_v1 | converged: True"
        assert summary == expected

    def test_create_result_summary_float_precision(self):
        """Test float precision formatting in result summary."""
        results = {
            "score": 0.123456,
            "loss": 0.000123456
        }
        summary = create_result_summary(results)

        expected = "score: 0.1235 | loss: 0.0001"
        assert summary == expected

    def test_create_result_summary_special_values(self):
        """Test summary creation with special float values."""
        results = {
            "normal": 1.0,
            "zero": 0.0,
            "negative": -0.5,
            "large": 1e6,
            "small": 1e-6
        }
        summary = create_result_summary(results)

        # Should not raise any exceptions and format correctly
        assert "normal: 1.0000" in summary
        assert "zero: 0.0000" in summary
        assert "negative: -0.5000" in summary
        assert "large: 1000000.0000" in summary
        assert "small: 0.0000" in summary


class TestAnalysisFormattersIntegration:
    """Integration tests for analysis formatters."""

    def test_result_summary_with_optimization_results(self):
        """Test result summary with typical optimization results."""
        optimization_results = {
            "best_score": -0.0234,
            "execution_time": 45.67,
            "trials_completed": 100,
            "converged": True,
            "optimizer": "bayesian"
        }

        summary = create_result_summary(optimization_results)

        expected_parts = [
            "best_score: -0.0234",
            "execution_time: 45.6700",
            "trials_completed: 100",
            "converged: True",
            "optimizer: bayesian"
        ]

        for part in expected_parts:
            assert part in summary

    def test_result_summary_with_trading_metrics(self):
        """Test result summary with trading performance metrics."""
        trading_metrics = {
            "total_return": 0.1567,
            "sharpe_ratio": 1.234,
            "max_drawdown": -0.089,
            "win_rate": 0.612,
            "total_trades": 150
        }

        summary = create_result_summary(trading_metrics)

        expected_parts = [
            "total_return: 0.1567",
            "sharpe_ratio: 1.2340",
            "max_drawdown: -0.0890",
            "win_rate: 0.6120",
            "total_trades: 150"
        ]

        for part in expected_parts:
            assert part in summary


if __name__ == "__main__":
    pytest.main([__file__])