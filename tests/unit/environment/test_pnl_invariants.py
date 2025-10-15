#!/usr/bin/env python3
"""
PnL Accounting Invariants Tests (xfail - to be fixed in environment).

ENVIRONMENT SPECIFICATION (Critical Assumptions):
- HeavyTradingEnv supports SHORTING: position ∈ {-1.0, 0.0, +1.0}
- At position=0 (neutral), BOTH BUY and SELL are LEGAL actions
  - BUY: Opens long (+1.0) or closes short
  - SELL: Closes long + Opens short (-1.0)
- This is the "always-flip" design (NOT a bug)

These tests define the **correct** accounting behavior that the environment
should satisfy. Currently marked as xfail because the environment has a bug
where PnL is calculated immediately after position change, resulting in
pnl=0 at entry.

Once the environment is fixed (Issue #XXX), these tests should PASS.

Invariants:
1. Static prices + zero fees → realized_pnl == 0, portfolio_value == initial
2. BUY->SELL round trip with price change → realized_pnl == Δ * position_size
3. Realized PnL only accumulates on position close
4. Portfolio value = initial + realized_pnl + unrealized_pnl
"""

import pandas as pd
import pytest

from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

class TestPnLInvariants:
    """PnL accounting invariant tests (xfail until environment is fixed)."""

    @pytest.fixture
    def static_price_data(self) -> pd.DataFrame:
        """Static price data for invariant testing."""
        price = 100.0
        df = pd.DataFrame({
            'close': [price] * 100,
            'open': [price] * 100,
            'high': [price] * 100,
            'low': [price] * 100,
            'volume': [1000.0] * 100,
        })
        return df

    @pytest.fixture
    def rising_price_data(self) -> pd.DataFrame:
        """Rising price data for profit testing."""
        df = pd.DataFrame({
            'close': [100.0] * 10 + [105.0] * 10 + [110.0] * 10,
            'open': [100.0] * 10 + [105.0] * 10 + [110.0] * 10,
            'high': [100.0] * 10 + [105.0] * 10 + [110.0] * 10,
            'low': [100.0] * 10 + [105.0] * 10 + [110.0] * 10,
            'volume': [1000.0] * 30,
        })
        return df

    @pytest.fixture
    def zero_fee_config(self) -> EnvironmentConfig:
        """Zero-fee configuration."""
        return EnvironmentConfig(
            curriculum_stage='full',
            transaction_cost=0.0,
            initial_portfolio_value=10000.0,
            max_position_size=1.0,
        )

    def test_static_price_zero_pnl(
        self,
        static_price_data: pd.DataFrame,
        zero_fee_config: EnvironmentConfig,
    ) -> None:
        """Static prices + zero fees → portfolio value unchanged."""
        env = HeavyTradingEnv(static_price_data, zero_fee_config)
        env.reset()
        
        initial_value = env.portfolio_value
        
        # Execute random actions
        env.step(ACTION_BUY)  # BUY
        env.step(ACTION_HOLD)  # HOLD
        env.step(ACTION_HOLD)  # HOLD
        env.step(ACTION_SELL)  # SELL
        env.step(ACTION_HOLD)  # HOLD
        env.step(ACTION_BUY)  # BUY
        env.step(ACTION_SELL)  # SELL

        final_value = env.portfolio_value
        
        # INVARIANT: Static prices → no profit/loss
        assert abs(final_value - initial_value) < 1e-6, \
            f"Static prices should not change portfolio value: " \
            f"initial={initial_value}, final={final_value}"

    def test_buy_sell_round_trip_pnl(
        self,
        rising_price_data: pd.DataFrame,
        zero_fee_config: EnvironmentConfig,
    ) -> None:
        """BUY->SELL round trip → realized_pnl == price_delta * position."""
        env = HeavyTradingEnv(rising_price_data, zero_fee_config)
        env.reset()
        
        initial_value = env.portfolio_value
        position_size = env.config.max_position_size
        
        # BUY at 100
        env.step(ACTION_BUY)
        price_buy = env.df.iloc[env.current_step]["close"]
        
        # Advance 10 steps (price rises to 105)
        for _ in range(10):
            env.step(ACTION_HOLD)
        
        # SELL at 105
        env.step(ACTION_SELL)
        price_sell = env.df.iloc[env.current_step]["close"]
        
        # INVARIANT: Realized PnL = (sell_price - buy_price) * position_size
        expected_pnl = (price_sell - price_buy) * position_size
        actual_pnl = env.portfolio_value - initial_value
        
        assert abs(actual_pnl - expected_pnl) < 1e-2, \
            f"Round trip PnL mismatch: expected={expected_pnl:.2f}, actual={actual_pnl:.2f}"

    def test_unrealized_pnl_not_accumulated(
        self,
        rising_price_data: pd.DataFrame,
        zero_fee_config: EnvironmentConfig,
    ) -> None:
        """Unrealized PnL should not be added to realized_pnl (only on position close)."""
        env = HeavyTradingEnv(rising_price_data, zero_fee_config)
        env.reset()
        
        # BUY at 100
        env.step(ACTION_BUY)
        
        # Check realized PnL after entry (should be 0 or negative due to entry cost)
        realized_after_entry = env.realized_pnl
        
        # HOLD while price rises (unrealized gain)
        for _ in range(10):
            env.step(ACTION_HOLD)
        
        # INVARIANT: realized_pnl should not change during HOLD (position not closed)
        assert env.realized_pnl == realized_after_entry, \
            f"Realized PnL should not change during HOLD: " \
            f"realized_pnl={env.realized_pnl}, expected={realized_after_entry}"
        
        # Now close position
        env.step(ACTION_SELL)
        
        # INVARIANT: After close, realized_pnl should reflect realized gain
        assert env.realized_pnl > realized_after_entry, \
            "After closing position, realized_pnl should increase"

    def test_portfolio_value_composition(
        self,
        rising_price_data: pd.DataFrame,
        zero_fee_config: EnvironmentConfig,
    ) -> None:
        """Portfolio value = initial + realized_pnl + unrealized_pnl."""
        env = HeavyTradingEnv(rising_price_data, zero_fee_config)
        env.reset()
        
        initial_value = env.portfolio_value
        
        # BUY at 100
        env.step(ACTION_BUY)
        entry_price = env.entry_price
        position = env.position
        
        # Advance to price=105
        for _ in range(10):
            env.step(ACTION_HOLD)
        
        current_price = env.df.iloc[env.current_step]["close"]
        
        # INVARIANT: Portfolio = initial + realized + unrealized
        unrealized_pnl = position * (current_price - entry_price)
        expected_value = initial_value + env.realized_pnl + unrealized_pnl
        actual_value = env.portfolio_value
        
        assert abs(actual_value - expected_value) < 1e-2, \
            f"Portfolio value composition incorrect: " \
            f"expected={expected_value:.2f}, actual={actual_value:.2f}"

    def test_fee_deduction_timing(
        self,
        static_price_data: pd.DataFrame,
    ) -> None:
        """Fees should be deducted immediately on trade execution."""
        config = EnvironmentConfig(
            curriculum_stage='full',
            transaction_cost=0.001,  # 0.1% fee
            initial_portfolio_value=10000.0,
            max_position_size=1.0,
        )
        env = HeavyTradingEnv(static_price_data, config)
        env.reset()
        
        initial_value = env.portfolio_value
        
        # BUY (should incur fee)
        env.step(ACTION_BUY)
        price = env.df.iloc[env.current_step]["close"]
        position = env.position
        expected_fee = price * position * config.transaction_cost
        
        # INVARIANT: Portfolio should decrease by exactly the fee
        # (no price change, so only fee impact)
        expected_value_after_buy = initial_value - expected_fee
        actual_value_after_buy = env.portfolio_value
        
        assert abs(actual_value_after_buy - expected_value_after_buy) < 1e-2, \
            f"Fee not deducted correctly on BUY: " \
            f"expected={expected_value_after_buy:.2f}, actual={actual_value_after_buy:.2f}"

    def test_symmetric_round_trip(
        self,
        static_price_data: pd.DataFrame,
        zero_fee_config: EnvironmentConfig,
    ) -> None:
        """BUY->SELL->BUY->SELL at static price → net zero PnL."""
        env = HeavyTradingEnv(static_price_data, zero_fee_config)
        env.reset()
        
        initial_value = env.portfolio_value
        
        # Multiple round trips at same price
        for _ in range(3):
            env.step(ACTION_BUY)
            env.step(ACTION_HOLD)
            env.step(ACTION_SELL)
            env.step(ACTION_HOLD)

        final_value = env.portfolio_value
        
        # INVARIANT: Symmetric round trips at static price → zero net PnL
        assert abs(final_value - initial_value) < 1e-6, \
            f"Symmetric round trips should have zero net PnL: " \
            f"initial={initial_value}, final={final_value}"
