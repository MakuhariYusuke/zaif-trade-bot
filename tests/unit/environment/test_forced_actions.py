#!/usr/bin/env python3
"""
Unit tests for forced action execution in trading environment.

Tests verify that:
- Forced action sequences produce expected inventory, average entry price, PnL, and fees
- BUY and SELL have symmetric fee/slippage application
- Position management is consistent across action sequences
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.environment.environment import HeavyTradingEnv


class TestForcedActions:
    """Test cases for forced action execution."""

    @pytest.fixture
    def simple_price_data(self) -> pd.DataFrame:
        """Create simple price data for deterministic testing."""
        # Simple linear price increase for predictable results
        dates = pd.date_range("2024-01-01", periods=100, freq="1min")
        prices = np.linspace(100.0, 110.0, 100)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.ones(100) * 1000,
        })
        
        return df

    @pytest.fixture
    def zero_fee_env(self, simple_price_data: pd.DataFrame) -> HeavyTradingEnv:
        """Create environment with zero fees for baseline testing."""
        config = {
            "transaction_cost": 0.0,
            "max_position_size": 1.0,
            "initial_portfolio_value": 10000.0,
            "curriculum_stage": "full",
            "reward_scaling": 1.0,
        }
        return HeavyTradingEnv(df=simple_price_data, config=config)

    @pytest.fixture
    def with_fee_env(self, simple_price_data: pd.DataFrame) -> HeavyTradingEnv:
        """Create environment with fees for fee testing."""
        config = {
            "transaction_cost": 0.001,  # 0.1% fee
            "max_position_size": 1.0,
            "initial_portfolio_value": 10000.0,
            "curriculum_stage": "full",
            "reward_scaling": 1.0,
        }
        return HeavyTradingEnv(df=simple_price_data, config=config)

    def test_hold_only_sequence(self, zero_fee_env: HeavyTradingEnv) -> None:
        """Test that HOLD-only sequence maintains initial state."""
        env = zero_fee_env
        env.reset()
        
        initial_portfolio = env.portfolio_value
        initial_position = env.position
        
        # Execute 10 HOLD actions
        for _ in range(10):
            obs, reward, done, truncated, info = env.step(0)  # 0 = HOLD
        
        # Portfolio should remain unchanged (no trades)
        assert env.portfolio_value == initial_portfolio, "HOLD should not change portfolio value"
        assert env.position == initial_position, "HOLD should not change position"
        assert env.position == 0.0, "Initial position should be 0"

    def test_buy_hold_sell_sequence(self, zero_fee_env: HeavyTradingEnv) -> None:
        """Test BUY -> HOLD -> SELL sequence with zero fees."""
        env = zero_fee_env
        env.reset()
        
        initial_portfolio = env.portfolio_value
        
        # Step 1: BUY
        env.step(1)  # 1 = BUY
        price_buy = env.df.iloc[env.current_step]["close"]
        
        # Should have position now
        assert env.position > 0, "BUY should create positive position"
        expected_position = env.config.max_position_size
        assert abs(env.position - expected_position) < 1e-6, f"Position should be {expected_position}"
        
        # Step 2: HOLD for a few steps
        for _ in range(3):
            env.step(0)  # HOLD
        
        # Position should remain
        assert abs(env.position - expected_position) < 1e-6, "HOLD should maintain position"
        
        # Step 3: SELL
        env.step(2)  # 2 = SELL
        price_sell = env.df.iloc[env.current_step]["close"]
        
        # Position should be closed
        assert env.position == 0.0, "SELL should close position"
        
        # Calculate expected PnL (with zero fees)
        expected_pnl = (price_sell - price_buy) * expected_position
        actual_pnl = env.portfolio_value - initial_portfolio
        
        assert abs(actual_pnl - expected_pnl) < 1e-2, \
            f"PnL mismatch: expected {expected_pnl:.2f}, got {actual_pnl:.2f}"

    def test_buy_sell_with_fees(self, with_fee_env: HeavyTradingEnv) -> None:
        """Test that fees are applied symmetrically for BUY and SELL."""
        env = with_fee_env
        env.reset()
        
        initial_portfolio = env.portfolio_value
        fee_rate = env.config.transaction_cost
        
        # BUY
        env.step(1)
        price_buy = env.df.iloc[env.current_step]["close"]
        position_size = env.position
        
        # Expected fee for BUY
        buy_fee = price_buy * position_size * fee_rate
        
        # SELL
        env.step(2)
        price_sell = env.df.iloc[env.current_step]["close"]
        
        # Expected fee for SELL
        sell_fee = price_sell * position_size * fee_rate
        
        # Calculate expected final portfolio
        gross_pnl = (price_sell - price_buy) * position_size
        total_fees = buy_fee + sell_fee
        expected_net_pnl = gross_pnl - total_fees
        expected_portfolio = initial_portfolio + expected_net_pnl
        
        actual_portfolio = env.portfolio_value
        
        assert abs(actual_portfolio - expected_portfolio) < 1e-2, \
            f"Portfolio mismatch: expected {expected_portfolio:.2f}, got {actual_portfolio:.2f}"
        
        # Verify fees were deducted
        assert actual_portfolio < initial_portfolio + gross_pnl, \
            "Fees should reduce final portfolio value"

    def test_multiple_round_trips(self, zero_fee_env: HeavyTradingEnv) -> None:
        """Test multiple BUY->SELL round trips."""
        env = zero_fee_env
        env.reset()
        
        initial_portfolio = env.portfolio_value
        position_size = env.config.max_position_size
        
        total_pnl = 0.0
        
        # Execute 3 round trips
        for _ in range(3):
            # BUY
            env.step(1)
            price_buy = env.df.iloc[env.current_step]["close"]
            
            # HOLD
            env.step(0)
            
            # SELL
            env.step(2)
            price_sell = env.df.iloc[env.current_step]["close"]
            
            # Accumulate PnL
            total_pnl += (price_sell - price_buy) * position_size
        
        expected_portfolio = initial_portfolio + total_pnl
        actual_portfolio = env.portfolio_value
        
        assert abs(actual_portfolio - expected_portfolio) < 1e-1, \
            f"Multi round-trip PnL mismatch: expected {expected_portfolio:.2f}, got {actual_portfolio:.2f}"

    def test_fee_symmetry(self, with_fee_env: HeavyTradingEnv) -> None:
        """Verify that BUY and SELL fees are applied at the same point in execution."""
        env = with_fee_env
        env.reset()
        
        initial_portfolio = env.portfolio_value
        
        # BUY
        env.step(1)
        portfolio_after_buy = env.portfolio_value
        price_buy = env.df.iloc[env.current_step]["close"]
        position = env.position
        
        # Fee should be deducted immediately on BUY
        expected_buy_fee = price_buy * position * env.config.transaction_cost
        # Note: In spot trading, BUY reduces cash immediately
        # Portfolio value change = -(cash spent on BUY + fee)
        # But position value is added, so net change should be just -fee
        
        # SELL
        env.step(2)
        portfolio_after_sell = env.portfolio_value
        price_sell = env.df.iloc[env.current_step]["close"]
        
        # Fee should be deducted immediately on SELL
        expected_sell_fee = price_sell * position * env.config.transaction_cost
        
        # Total fees
        total_fees = expected_buy_fee + expected_sell_fee
        
        # Gross PnL
        gross_pnl = (price_sell - price_buy) * position
        
        # Net PnL
        net_pnl = portfolio_after_sell - initial_portfolio
        expected_net_pnl = gross_pnl - total_fees
        
        assert abs(net_pnl - expected_net_pnl) < 1e-2, \
            f"Fee symmetry violated: expected net PnL {expected_net_pnl:.2f}, got {net_pnl:.2f}"

    def test_illegal_action_handling(self, zero_fee_env: HeavyTradingEnv) -> None:
        """Test that illegal actions are handled correctly."""
        env = zero_fee_env
        env.reset()
        
        # Get legal actions at start (should not allow SELL with no position)
        legal_actions = env.get_legal_actions()
        
        # SELL should be illegal with no position (assuming spot trading)
        if env.position == 0:
            # In spot trading, SELL without position should be masked
            # This depends on implementation - verify mask works
            assert legal_actions[2] == 0, "SELL should be illegal with no position in spot trading"
        
        # BUY to create position
        env.step(1)
        
        # Now check legal actions with position
        legal_actions = env.get_legal_actions()
        
        # SELL should now be legal
        assert legal_actions[2] == 1, "SELL should be legal with positive position"

    def test_action_sequence_consistency(self, zero_fee_env: HeavyTradingEnv) -> None:
        """Test that the same action sequence produces consistent results."""
        env1 = zero_fee_env
        env2 = HeavyTradingEnv(df=env1.df.copy(), config=env1.config.as_dict())
        
        env1.reset()
        env2.reset()
        
        action_sequence = [0, 1, 0, 0, 2, 0, 1, 2]  # HOLD, BUY, HOLD, HOLD, SELL, HOLD, BUY, SELL
        
        for action in action_sequence:
            obs1, r1, d1, t1, i1 = env1.step(action)
            obs2, r2, d2, t2, i2 = env2.step(action)
        
        assert env1.portfolio_value == env2.portfolio_value, \
            "Same action sequence should produce same portfolio value"
        assert env1.position == env2.position, \
            "Same action sequence should produce same position"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
