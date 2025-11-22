"""
Unit tests for LongTermMetrics (SAC v448 Layer 1).

Tests:
- Sharpe ratio calculation
- Max drawdown calculation
- Action balance stability
- Transaction cost efficiency
- Sustainable profitability score
- Edge cases

Version: 1.0
Created: 2025-11-22
"""

import pytest
import numpy as np
from ztb.trading.environment.components.reward.metrics import LongTermMetrics


class TestSharpeRatio:
    """Sharpe ratio calculation tests."""
    
    def test_positive_sharpe(self):
        """Test with positive returns."""
        metrics = LongTermMetrics()
        returns = np.array([0.05, 0.03, 0.04, 0.06, 0.02])
        
        sharpe = metrics.sharpe_ratio(returns, risk_free_rate=0.0)
        
        assert sharpe > 0, "Expected positive Sharpe for positive returns"
        assert np.isfinite(sharpe)
    
    def test_negative_sharpe(self):
        """Test with negative returns."""
        metrics = LongTermMetrics()
        returns = np.array([-0.05, -0.03, -0.04, -0.06, -0.02])
        
        sharpe = metrics.sharpe_ratio(returns, risk_free_rate=0.0)
        
        assert sharpe < 0, "Expected negative Sharpe for negative returns"
    
    def test_zero_volatility(self):
        """Test with zero volatility (constant returns)."""
        metrics = LongTermMetrics()
        returns = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        
        sharpe = metrics.sharpe_ratio(returns, risk_free_rate=0.0)
        
        assert sharpe == 0.0, "Sharpe should be 0 for zero volatility"
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        metrics = LongTermMetrics()
        returns = np.array([0.05])  # Only 1 return
        
        sharpe = metrics.sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_risk_free_rate(self):
        """Test with non-zero risk-free rate."""
        metrics = LongTermMetrics()
        returns = np.array([0.05, 0.03, 0.04, 0.06, 0.02])
        
        sharpe_no_rf = metrics.sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = metrics.sharpe_ratio(returns, risk_free_rate=0.02)
        
        # Sharpe should be lower with positive risk-free rate
        assert sharpe_with_rf < sharpe_no_rf


class TestMaxDrawdown:
    """Max drawdown calculation tests."""
    
    def test_no_drawdown(self):
        """Test with monotonically increasing portfolio."""
        metrics = LongTermMetrics()
        portfolio = np.array([100, 110, 120, 130, 140])
        
        max_dd = metrics.max_drawdown(portfolio)
        
        assert max_dd == 0.0, "Expected no drawdown for increasing portfolio"
    
    def test_simple_drawdown(self):
        """Test with simple drawdown."""
        metrics = LongTermMetrics()
        portfolio = np.array([100, 120, 90, 110, 130])
        # Peak at 120, trough at 90
        # Drawdown = (90 - 120) / 120 = -0.25 (25%)
        
        max_dd = metrics.max_drawdown(portfolio)
        
        expected = (90 - 120) / 120
        assert np.isclose(max_dd, expected, atol=0.01)
        assert max_dd < 0, "Drawdown should be negative"
    
    def test_multiple_drawdowns(self):
        """Test with multiple drawdowns (max should be returned)."""
        metrics = LongTermMetrics()
        portfolio = np.array([100, 120, 90, 110, 80, 130])
        # Peak at 120, drawdown to 90 = -25%
        # Peak at 120, drawdown to 80 = -33.3% (larger, from same peak)
        
        max_dd = metrics.max_drawdown(portfolio)
        
        # Should return the larger drawdown (from peak 120 to trough 80)
        expected = (80 - 120) / 120  # -0.333
        assert np.isclose(max_dd, expected, atol=0.01)
    
    def test_recovery_after_drawdown(self):
        """Test that recovery doesn't affect max drawdown."""
        metrics = LongTermMetrics()
        portfolio = np.array([100, 120, 60, 150])
        # Drawdown: 120 -> 60 = -50%
        # Recovery to 150 doesn't change max drawdown
        
        max_dd = metrics.max_drawdown(portfolio)
        
        expected = (60 - 120) / 120  # -0.5
        assert np.isclose(max_dd, expected, atol=0.01)
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        metrics = LongTermMetrics()
        portfolio = np.array([100])
        
        max_dd = metrics.max_drawdown(portfolio)
        
        assert max_dd == 0.0


class TestActionBalanceStability:
    """Action balance stability tests."""
    
    def test_perfect_stability(self):
        """Test with perfectly consistent action distribution."""
        metrics = LongTermMetrics()
        
        # 200 actions: always 50% BUY, 50% SELL
        actions = [1, 2] * 100  # Alternating BUY, SELL
        
        stability = metrics.action_balance_stability(actions, window=100)
        
        # Should be very stable (near 0)
        assert stability < 0.01, f"Expected high stability, got {stability}"
    
    def test_erratic_behavior(self):
        """Test with erratic action distribution."""
        metrics = LongTermMetrics()
        
        # First 100: mostly BUY
        # Second 100: mostly SELL
        actions = [1] * 90 + [2] * 10 + [2] * 90 + [1] * 10
        
        stability = metrics.action_balance_stability(actions, window=100)
        
        # Should show high variance (instability)
        assert stability > 0.10, f"Expected low stability, got {stability}"
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        metrics = LongTermMetrics()
        actions = [1, 2, 1, 2, 1]  # Only 5 actions
        
        stability = metrics.action_balance_stability(actions, window=100)
        
        assert stability == 0.0
    
    def test_gradual_shift(self):
        """Test with gradual shift in distribution."""
        metrics = LongTermMetrics()
        
        # Gradual shift from balanced to BUY-heavy
        actions = []
        for window_i in range(3):
            buy_ratio = 0.5 + window_i * 0.2  # 0.5, 0.7, 0.9
            n_buy = int(100 * buy_ratio)
            n_sell = 100 - n_buy
            actions.extend([1] * n_buy + [2] * n_sell)
        
        stability = metrics.action_balance_stability(actions, window=100)
        
        # Should show some variance but less than erratic
        # Gradual shifts may have lower variance than expected
        assert 0.01 < stability < 0.25


class TestTransactionCostEfficiency:
    """Transaction cost efficiency tests."""
    
    def test_low_cost_efficiency(self):
        """Test with low costs (good)."""
        metrics = LongTermMetrics()
        gross_pnl = 10000
        costs = 500  # 5% costs
        
        efficiency = metrics.transaction_cost_efficiency(gross_pnl, costs)
        
        assert efficiency == 0.05
    
    def test_high_cost_inefficiency(self):
        """Test with high costs (bad)."""
        metrics = LongTermMetrics()
        gross_pnl = 10000
        costs = 8000  # 80% costs!
        
        efficiency = metrics.transaction_cost_efficiency(gross_pnl, costs)
        
        assert efficiency == 0.80
    
    def test_costs_exceed_pnl(self):
        """Test when costs exceed PnL."""
        metrics = LongTermMetrics()
        gross_pnl = 5000
        costs = 8000  # Costs > PnL
        
        efficiency = metrics.transaction_cost_efficiency(gross_pnl, costs)
        
        assert efficiency > 1.0  # Inefficient
    
    def test_negative_pnl(self):
        """Test with negative PnL."""
        metrics = LongTermMetrics()
        gross_pnl = -5000  # Loss
        costs = 2000
        
        efficiency = metrics.transaction_cost_efficiency(gross_pnl, costs)
        
        # Should still calculate (using absolute value)
        assert efficiency == 2000 / 5000  # 0.4
    
    def test_zero_pnl(self):
        """Test with zero PnL."""
        metrics = LongTermMetrics()
        gross_pnl = 0
        costs = 1000
        
        efficiency = metrics.transaction_cost_efficiency(gross_pnl, costs)
        
        # Should return 1.0 (worst case)
        assert efficiency == 1.0


class TestSustainableProfitabilityScore:
    """Sustainable profitability score tests."""
    
    def test_excellent_sustainable_strategy(self):
        """Test with excellent metrics across board."""
        metrics = LongTermMetrics()
        
        score = metrics.sustainable_profitability_score(
            final_reward=9.0,      # High reward
            balance_stability=0.05,  # Very stable
            max_dd=-0.10,           # Small drawdown
            sharpe=2.0              # Excellent Sharpe
        )
        
        assert score > 0.7, f"Expected excellent score, got {score}"
    
    def test_poor_unsustainable_strategy(self):
        """Test with poor metrics."""
        metrics = LongTermMetrics()
        
        score = metrics.sustainable_profitability_score(
            final_reward=-5.0,      # Negative reward
            balance_stability=0.40,  # Unstable
            max_dd=-0.60,           # Large drawdown
            sharpe=-1.0             # Negative Sharpe
        )
        
        assert score < 0.3, f"Expected poor score, got {score}"
    
    def test_high_reward_but_unsustainable(self):
        """Test high reward with unsustainable characteristics."""
        metrics = LongTermMetrics()
        
        score = metrics.sustainable_profitability_score(
            final_reward=15.0,      # Very high reward
            balance_stability=0.35,  # Unstable (extreme bias)
            max_dd=-0.50,           # Large drawdown
            sharpe=0.5              # Low Sharpe
        )
        
        # Should be penalized for unsustainability despite high reward
        assert score < 0.6, f"Expected moderate score due to instability, got {score}"
    
    def test_moderate_balanced_strategy(self):
        """Test moderate but balanced strategy."""
        metrics = LongTermMetrics()
        
        score = metrics.sustainable_profitability_score(
            final_reward=8.0,       # Good reward
            balance_stability=0.08,  # Stable
            max_dd=-0.20,           # Moderate drawdown
            sharpe=1.2              # Good Sharpe
        )
        
        assert 0.5 < score < 0.8, f"Expected moderate-good score, got {score}"
    
    def test_custom_weights(self):
        """Test with custom weights."""
        metrics = LongTermMetrics()
        
        # Emphasize reward heavily
        custom_weights = {
            "reward": 0.70,
            "stability": 0.10,
            "drawdown": 0.10,
            "sharpe": 0.10,
        }
        
        score_default = metrics.sustainable_profitability_score(
            final_reward=10.0,
            balance_stability=0.30,
            max_dd=-0.40,
            sharpe=0.5
        )
        
        score_custom = metrics.sustainable_profitability_score(
            final_reward=10.0,
            balance_stability=0.30,
            max_dd=-0.40,
            sharpe=0.5,
            weights=custom_weights
        )
        
        # Custom weights emphasize reward, should give higher score
        assert score_custom > score_default


class TestEvaluateEpisode:
    """Comprehensive episode evaluation tests."""
    
    def test_complete_evaluation(self):
        """Test with all metrics available."""
        metrics = LongTermMetrics()
        
        episode_data = {
            "final_reward": 8.5,
            "episode_returns": np.array([0.05, 0.03, 0.04, 0.06, 0.02]),
            "portfolio_values": np.array([100, 110, 105, 115, 120]),
            "action_history": [1, 2, 1, 2] * 50,  # 200 actions
            "gross_pnl": 10000,
            "transaction_costs": 1000,
        }
        
        results = metrics.evaluate_episode(episode_data)
        
        # Check all metrics calculated
        assert "final_reward" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "balance_stability" in results
        assert "cost_efficiency" in results
        assert "sustainability_score" in results
        
        assert results["final_reward"] == 8.5
        assert np.isfinite(results["sharpe_ratio"])
        assert results["max_drawdown"] < 0  # Has some drawdown
        assert np.isfinite(results["balance_stability"])
        assert results["cost_efficiency"] == 0.1
    
    def test_partial_evaluation(self):
        """Test with partial data."""
        metrics = LongTermMetrics()
        
        episode_data = {
            "final_reward": 5.0,
            "portfolio_values": np.array([100, 110, 105, 115]),
        }
        
        results = metrics.evaluate_episode(episode_data)
        
        # Should have final_reward and max_drawdown
        assert "final_reward" in results
        assert "max_drawdown" in results
        
        # Should NOT have sustainability_score (missing other metrics)
        assert "sustainability_score" not in results


class TestRealWorldScenarios:
    """Realistic trading scenario tests."""
    
    def test_v447_typical_balanced_case(self):
        """Test metrics for typical v447 balanced case."""
        metrics = LongTermMetrics()
        
        # BUY=51%, SELL=44%, HOLD=5% - balanced
        actions = [1] * 51 + [2] * 44 + [0] * 5
        np.random.shuffle(actions)
        actions = actions * 4  # 400 actions
        
        stability = metrics.action_balance_stability(actions, window=100)
        
        # Should be stable
        assert stability < 0.15, f"Expected stable, got {stability}"
    
    def test_v447_bias_collapse_case(self):
        """Test metrics for v447 bias collapse case."""
        metrics = LongTermMetrics()
        
        # BUY=93%, SELL=4%, HOLD=3% - collapsed
        actions = [1] * 93 + [2] * 4 + [0] * 3
        actions = actions * 4  # 400 actions
        
        stability = metrics.action_balance_stability(actions, window=100)
        
        # Should show low variance (consistently biased)
        # BUT this is NOT good - it's consistently bad
        # This is why we need sustainability_score combining multiple metrics
        assert stability < 0.10  # Stable but biased
    
    def test_high_frequency_1min_trading(self):
        """Test cost efficiency for high-frequency 1-min trading."""
        metrics = LongTermMetrics()
        
        # 1500 trades/episode at 0.1% cost each
        # Initial balance: 200,000
        # Total costs: 200,000 * 0.001 * 1500 = 300,000 (150%)
        gross_pnl = 50000  # Gross profit
        costs = 300000
        
        efficiency = metrics.transaction_cost_efficiency(gross_pnl, costs)
        
        # Costs are 600% of profit!
        assert efficiency == 6.0
        assert efficiency > 1.0  # Highly inefficient


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
