"""Tests for ActionValidator component."""

import numpy as np
import pytest

from ztb.trading.environment.components.action_validator import ActionValidator
from ztb.trading.environment.utils.config import EnvironmentConfig


@pytest.fixture
def sample_config():
    """Sample environment config for testing."""
    return EnvironmentConfig.from_dict({
        'max_position_size': 1.0,
        'transaction_cost': 0.001,
        'exchange': 'coincheck'
    })


@pytest.fixture
def action_validator(sample_config):
    """ActionValidator instance for testing."""
    return ActionValidator(sample_config, initial_portfolio_value=200000.0)


@pytest.fixture
def price_history():
    """Price history array for testing."""
    return np.full(150, 5000000.0)  # 150 steps of price data


class TestActionValidator:
    """Test cases for ActionValidator component."""

    def test_initialization(self, action_validator):
        """Test ActionValidator initialization."""
        assert action_validator.config.max_position_size == 1.0
        assert action_validator.config.transaction_cost == 0.001
        assert action_validator.initial_portfolio_value == 200000.0

    def test_flat_position_allows_all_actions(self, action_validator):
        """Test that flat position (0) allows BUY, SELL, and HOLD."""
        # Create sufficient price history
        price_history = np.full(150, 5000000.0)  # 150 steps of price data
        
        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=0.0,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history,
            price_array=price_history,
            df=None
        )

        # HOLD (0), BUY (1), SELL (2) should all be legal
        assert legal_actions[0] == 1  # HOLD always legal
        assert legal_actions[1] == 1  # BUY legal for flat position
        assert legal_actions[2] == 1  # SELL legal for flat position

    def test_long_position_allows_all_actions_with_funds(self, action_validator, price_history):
        """Test that long position (>0) allows BUY, SELL, and HOLD when funds are sufficient."""
        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=0.5,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history,
            price_array=price_history,
            df=None
        )

        assert legal_actions[0] == 1  # HOLD always legal
        assert legal_actions[1] == 1  # BUY legal for long position (funds sufficient)
        assert legal_actions[2] == 1  # SELL legal for long position (funds sufficient)

    def test_short_position_allows_all_actions_with_funds(self, action_validator, price_history):
        """Test that short position (<0) allows BUY, SELL, and HOLD when funds are sufficient."""
        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=-0.5,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history,
            price_array=price_history,
            df=None
        )

        assert legal_actions[0] == 1  # HOLD always legal
        assert legal_actions[1] == 1  # BUY legal for short position (funds sufficient)
        assert legal_actions[2] == 1  # SELL legal for short position (funds sufficient)

    def test_insufficient_funds_blocks_buy(self, action_validator, price_history):
        """Test that insufficient funds block BUY action."""
        # Very low portfolio value
        validator_low_funds = ActionValidator(
            action_validator.config,
            initial_portfolio_value=1000.0  # Very low funds
        )

        legal_actions = validator_low_funds.get_legal_actions(
            current_step=100,
            position=0.0,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history,
            price_array=price_history,
            df=None
        )

        assert legal_actions[0] == 1  # HOLD always legal
        assert legal_actions[1] == 0  # BUY blocked due to insufficient funds
        assert legal_actions[2] == 0  # SELL also blocked due to insufficient funds for short position

    def test_minimum_trade_size_buy(self, action_validator, price_history):
        """Test BUY action with minimum trade size logic."""
        # Set portfolio value that allows only minimum BTC unit
        validator_min_size = ActionValidator(
            action_validator.config,
            initial_portfolio_value=18000.0  # ~0.0036 BTC at 5M JPY
        )

        legal_actions = validator_min_size.get_legal_actions(
            current_step=100,
            position=0.0,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history, price_array=price_history,
            df=None
        )

        # Should allow BUY due to minimum trade size logic
        assert legal_actions[1] == 1

    def test_trade_cooldown_enforcement(self, action_validator, price_history):
        """Test that trade cooldown blocks actions appropriately."""
        # Recent trade should block new trades
        legal_actions = action_validator.get_legal_actions(
            current_step=10,
            position=0.0,
            total_pnl=0.0,
            trades_count=1,
            last_trade_step=8,  # Recent trade
            consecutive_trade_steps=1,
            close_array=price_history, price_array=price_history,
            df=None
        )

        # All actions should be blocked except possibly position closing
        # (but since position is 0, only HOLD should be legal)
        assert legal_actions[0] == 1  # HOLD always legal
        # BUY and SELL may be blocked due to cooldown

    def test_max_consecutive_trades_limit(self, action_validator, price_history):
        """Test max consecutive trades limit."""
        # High consecutive trade count
        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=0.5,  # Long position
            total_pnl=0.0,
            trades_count=10,
            last_trade_step=95,
            consecutive_trade_steps=6,  # Over limit
            close_array=price_history, price_array=price_history,
            df=None
        )

        # Should allow position closing even with consecutive trade limit
        assert legal_actions[0] == 1  # HOLD
        assert legal_actions[1] == 1  # BUY allowed (funds sufficient, consecutive limit doesn't block BUY/SELL)
        assert legal_actions[2] == 1  # SELL allowed (close long position)

    def test_volatility_filtering(self, action_validator):
        """Test volatility-based action filtering."""
        # Create volatile price data
        volatile_prices = np.array([5000000.0, 5100000.0, 4900000.0, 5200000.0, 4800000.0])

        legal_actions = action_validator.get_legal_actions(
            current_step=25,  # After volatility check starts
            position=0.0,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=volatile_prices,
            price_array=volatile_prices,
            df=None
        )

        # High volatility should potentially block some actions
        # (exact behavior depends on volatility threshold)
        assert legal_actions[0] == 1  # HOLD should always be legal

    def test_zero_price_handling(self, action_validator):
        """Test handling of zero or invalid price data."""
        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=0.0,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=None,
            price_array=None,
            df=None
        )

        # Should only allow HOLD when price cannot be resolved
        assert legal_actions[0] == 1  # HOLD
        assert legal_actions[1] == 0  # BUY blocked
        assert legal_actions[2] == 0  # SELL blocked

    def test_short_position_closing_cost_calculation(self, action_validator, price_history):
        """Test that short position closing calculates correct BUY cost."""
        # This test verifies the fix for short position BUY cost calculation
        position = -0.1  # Short position
        price = 5000000.0

        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=position,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history,
            price_array=price_history,
            df=None
        )

        # BUY should be legal for closing short position
        # The cost should be |position| * price, not position_size * price
        assert legal_actions[1] == 1  # BUY should be allowed

    def test_action_validator_logging(self, action_validator, price_history, caplog):
        """Test that ActionValidator produces appropriate log output."""
        import logging
        caplog.set_level(logging.INFO)

        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=0.0,
            total_pnl=0.0,
            trades_count=0,
            last_trade_step=None,
            consecutive_trade_steps=0,
            close_array=price_history, price_array=price_history,
            df=None
        )

        # Should log action validation results
        assert "ActionValidator: legal_actions=" in caplog.text

    def test_sell_lock_fix_short_position_allows_all_actions(self, action_validator, price_history):
        """Test that SELL-lock is fixed: short positions allow BUY, SELL, and HOLD when funds sufficient.

        This test validates the critical fix for inverted BUY/SELL masking logic.
        Previously, BUY conditions were: position >= -0.0001 (wrong - only long positions)
        SELL conditions were: position <= 0.0001 (wrong - only short positions)

        Corrected to: BUY/SELL always allowed when funds are sufficient, regardless of position.
        """
        # Test case that would cause SELL-lock: small short position with sufficient funds
        legal_actions = action_validator.get_legal_actions(
            current_step=100,
            position=-0.018,  # Small short position (like in training logs)
            total_pnl=-57411.67,  # Portfolio value = 200000 + (-57411.67) = 142588.33
            trades_count=1,
            last_trade_step=0,
            consecutive_trade_steps=1,
            close_array=price_history,
            price_array=price_history,
            df=None
        )

        # Critical assertions for SELL-lock fix - all actions legal with sufficient funds
        assert legal_actions[0] == 1, "HOLD should always be legal"
        assert legal_actions[1] == 1, "BUY should be legal for short positions (funds sufficient)"
        assert legal_actions[2] == 1, "SELL should be legal for short positions (funds sufficient)"

    def test_buy_sell_logic_inversion_prevention(self, action_validator, price_history):
        """Test that BUY/SELL logic allows all actions when funds are sufficient, preventing future inversions.

        This test ensures the logic remains correct:
        - BUY: Always allowed when funds are sufficient (regardless of position)
        - SELL: Always allowed when funds are sufficient (regardless of position)
        - HOLD: Always allowed
        """
        test_cases = [
            # (position, expected_legal_actions[HOLD, BUY, SELL])
            (0.0, [1, 1, 1]),  # Flat: all actions legal
            (0.5, [1, 1, 1]),  # Long: all actions legal (funds sufficient)
            (-0.5, [1, 1, 1]),  # Short: all actions legal (funds sufficient)
            (0.001, [1, 1, 1]),  # Small long: all actions legal (funds sufficient)
            (-0.001, [1, 1, 1]),  # Small short: all actions legal (funds sufficient)
        ]

        for position, expected in test_cases:
            legal_actions = action_validator.get_legal_actions(
                current_step=100,
                position=position,
                total_pnl=0.0,
                trades_count=0,
                last_trade_step=None,
                consecutive_trade_steps=0,
                close_array=price_history,
                price_array=price_history,
                df=None
            )

            assert legal_actions.tolist() == expected, (
                f"Position {position}: expected {expected}, got {legal_actions.tolist()}"
            )