"""Tests for balance penalty targets in BehavioralPenaltyCalculator."""


from ztb.trading.constants import ACTION_BUY, ACTION_SELL
from ztb.trading.environment.components.behavioral_penalty_calculator import (
    BehavioralPenaltyCalculator,
)
from ztb.trading.environment.utils.config import EnvironmentConfig


def test_balance_penalty_targets_prefer_buy():
    # Build environment config with reward_settings targets preferring BUY
    config = EnvironmentConfig(
        curriculum_stage="forced_balance",
        max_position_size=1.0,
        reward_settings={
            "balance_penalty_enabled": True,
            "balance_penalty": 50.0,
            "balance_penalty_min_actions": 1,
            "balance_penalty_targets": {
                "buy_target": 0.45,
                "sell_target": 0.30,
                "hold_target": 0.25,
            },
        },
    )

    calc = BehavioralPenaltyCalculator(config)

    # Simulate actions: skewed toward SELL
    for _ in range(15):
        calc.record_action(ACTION_SELL)

    # Now test hypothetical BUY action should reduce imbalance and therefore be rewarded
    buy_penalty = calc.calculate_balance_penalty(ACTION_BUY, action_bonus=0.0)
    sell_penalty = calc.calculate_balance_penalty(ACTION_SELL, action_bonus=0.0)

    assert buy_penalty >= sell_penalty, (
        f"BUY penalty should be >= SELL penalty when buy_target is higher; buy={buy_penalty}, sell={sell_penalty}"
    )
