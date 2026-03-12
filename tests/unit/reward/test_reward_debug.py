#!/usr/bin/env python3
"""
Test reward calculation for different actions
"""

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


# Create reward calculator with default settings
config = EnvironmentConfig(
    curriculum_stage="strong_penalty_trading",
    max_position_size=1.0,
)
reward_settings = RewardSettings(
    reward_scale=100.0,
    trading_bonus=0.01,
    use_simple_reward=False,
)
calculator = RewardCalculator(config, reward_settings, 200000.0)

# Test different actions with same conditions
test_pnl = 100.0
test_position = 0.01
test_portfolio = 200000.0
test_atr = 500.0
test_price = 5000000.0

print("Testing reward calculation for different actions:")
print(
    f"Conditions: PnL={test_pnl}, Position={test_position}, Portfolio={test_portfolio}"
)

for action_name, action in [
    ("HOLD", ACTION_HOLD),
    ("BUY", ACTION_BUY),
    ("SELL", ACTION_SELL),
]:
    reward = calculator.calculate_reward(
        action=action,
        current_price=test_price,
        position=test_position,
        portfolio_value=test_portfolio,
        atr=test_atr,
        transaction_cost=10.0,
        reward_scaling=1.0,
        pnl=test_pnl,
        old_position=0.0,
        step=1,
        observation=None,
        reward_history=[],
        portfolio_value_history=[test_portfolio] * 30,
    )
    print(f"{action_name}: {reward:.4f}")
