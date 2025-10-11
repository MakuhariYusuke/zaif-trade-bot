"""
Unit tests for ztb.analysis.transaction_cost_analysis module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from ztb.analysis.transaction_cost_analysis import simulate_transaction_costs
except ImportError:
    pytest.skip("ztb.analysis.transaction_cost_analysis module not available", allow_module_level=True)


class TestTransactionCostAnalysis:
    """Test cases for transaction cost analysis functions."""

    @patch("ztb.analysis.transaction_cost_analysis.TradingEvaluator")
    @patch("ztb.analysis.transaction_cost_analysis.plt")
    @patch("ztb.analysis.transaction_cost_analysis.pd.DataFrame.to_csv")
    def test_simulate_transaction_costs_normal(self, mock_to_csv, mock_plt, mock_evaluator_class):
        """Test normal transaction cost simulation."""
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_model.return_value = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "max_drawdown": 0.08,
            "total_trades": 100,
            "avg_trade_return": 0.0015,
        }
        mock_evaluator_class.return_value = mock_evaluator
        
        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Mock Path.mkdir
        with patch.object(Path, "mkdir"):
            cost_range = [0.001, 0.002]
            result = simulate_transaction_costs(
                Path("model.pkl"), cost_range, Path("data.csv"), Path("output")
            )
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 0.001 in result
        assert 0.002 in result
        
        for cost in cost_range:
            assert "total_return" in result[cost]
            assert "sharpe_ratio" in result[cost]
            assert result[cost]["total_return"] == 0.15

    @patch("ztb.analysis.transaction_cost_analysis.TradingEvaluator")
    @patch("ztb.analysis.transaction_cost_analysis.plt")
    @patch("ztb.analysis.transaction_cost_analysis.pd.DataFrame.to_csv")
    def test_simulate_transaction_costs_evaluator_error(self, mock_to_csv, mock_plt, mock_evaluator_class):
        """Test transaction cost simulation with evaluator error."""
        # Mock evaluator to raise exception
        mock_evaluator_class.side_effect = Exception("Evaluation failed")
        
        # Mock plt.subplots
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        with patch.object(Path, "mkdir"):
            cost_range = [0.001]
            result = simulate_transaction_costs(
                Path("model.pkl"), cost_range, Path("data.csv"), Path("output")
            )
        
        assert isinstance(result, dict)
        assert 0.001 in result
        assert "error" in result[0.001]

    @patch("ztb.analysis.transaction_cost_analysis.safe_operation")
    def test_simulate_transaction_costs_safe_operation_failure(self, mock_safe_operation):
        """Test transaction cost simulation when safe_operation fails."""
        mock_safe_operation.return_value = {}
        
        result = simulate_transaction_costs(
            Path("model.pkl"), [0.001], Path("data.csv"), Path("output")
        )
        
        assert result == {}
        mock_safe_operation.assert_called_once()