#!/usr/bin/env python3
"""
Test script for reward calculation bug fixes.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings


class MockConfig:
    def __init__(self):
        self.curriculum_stage = "ultra_profit"
        self.max_position_size = 1.0
        self.transaction_cost = 0.001


def test_get_nested_setting_fix():
    """Test that _get_nested_setting works correctly after bug fix."""
    config = MockConfig()
    reward_settings = {
        "profit_weight": 1.0,
        "risk_weight": 1.0,
        "consistency_weight": 1.0,
        "ultra_profit_multiplier": 1.0,
        "ultra_risk_multiplier": 1.0,
        "reward_scale": 1.0,
        "reward_clip_min": -10.0,
        "reward_clip_max": 10.0,
        "use_simple_reward": False,
        "profit_multiplier": 0.01,
        "loss_penalty_multiplier": 0.01,
        "hold_penalty_rate": 0.01,
        "balance_penalty_tolerance": 0.15,
        "balance_penalty": 1.0,
        "trading_bonus_multiplier": 2.0,
        "trading_bonus": 0.01,
        "dynamic_reward_shaping": {"enabled": True, "regime_detection_window": 20},
    }

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Test nested setting access
    value = calculator._get_nested_setting("dynamic_reward_shaping.enabled")
    assert value is True

    value = calculator._get_nested_setting(
        "dynamic_reward_shaping.regime_detection_window"
    )
    assert value == 20

    # Test non-existent key
    value = calculator._get_nested_setting("nonexistent.key")
    assert value is None

    # Test partial path
    value = calculator._get_nested_setting("dynamic_reward_shaping")
    assert isinstance(value, dict)


def test_calculate_reward_simple_with_transaction_cost():
    """Test that calculate_reward_simple properly handles transaction_cost parameter."""
    config = MockConfig()
    reward_settings = {
        "profit_weight": 1.0,
        "risk_weight": 1.0,
        "consistency_weight": 1.0,
        "ultra_profit_multiplier": 1.0,
        "ultra_risk_multiplier": 1.0,
        "reward_scale": 1.0,
        "reward_clip_min": -10.0,
        "reward_clip_max": 10.0,
        "use_simple_reward": False,
        "profit_multiplier": 0.01,
        "loss_penalty_multiplier": 0.01,
        "hold_penalty_rate": 0.01,
        "balance_penalty_tolerance": 0.15,
        "balance_penalty": 1.0,
        "trading_bonus_multiplier": 2.0,
        "trading_bonus": 0.01,
        "hold_penalty_multiplier": 1.0,
        "trade_frequency_bonus": 0.0,
        "reward_scaling": 1.0,
        "reward_clip_value": 10.0,
    }

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Test with trade (transaction_cost parameter is accepted but not used in calculation
    # since it's already deducted in position_manager)
    reward = calculator.calculate_reward_simple(
        pnl=1000.0,
        portfolio_value=200000.0,
        position=0.5,
        old_position=0.0,  # Position changed, indicating a trade
        action=ACTION_BUY,
        reward_history=[],
        portfolio_value_history=[],
        current_price=5000000.0,
        step=1,
        transaction_cost=100.0,  # This parameter is accepted but not used in calculation
    )

    # Reward should be based on pnl * scaling, modified by various components
    # The exact value depends on dynamic shaping and other components
    # We just verify it's a reasonable finite number
    assert isinstance(reward, (int, float))
    assert np.isfinite(reward)
    # Since transaction_cost is not used in calculation, reward should be > 1000 * 0.01 (minimum scaling)
    assert reward > 10.0


def test_calculate_reward_calls_simple_with_transaction_cost():
    """Test that calculate_reward properly passes transaction_cost to calculate_reward_simple."""
    config = MockConfig()
    config.curriculum_stage = "test"  # Force use_simple_reward path

    reward_settings = RewardSettings(
        profit_weight=1.0,
        risk_weight=1.0,
        consistency_weight=1.0,
        ultra_profit_multiplier=1.0,
        ultra_risk_multiplier=1.0,
        reward_scale=1.0,
        reward_clip_min=-10.0,
        reward_clip_max=10.0,
        use_simple_reward=True,  # Force simple reward path
        profit_multiplier=0.01,
        loss_penalty_multiplier=0.01,
        hold_penalty_rate=0.01,
        balance_penalty_tolerance=0.15,
        balance_penalty=1.0,
        trading_bonus_multiplier=2.0,
        trading_bonus=0.01,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Mock the calculate_reward_simple method to check if transaction_cost is passed
    original_method = calculator.calculate_reward_simple
    passed_transaction_cost = None

    def mock_calculate_reward_simple(*args, **kwargs):
        nonlocal passed_transaction_cost
        passed_transaction_cost = kwargs.get(
            "transaction_cost", args[9] if len(args) > 9 else None
        )
        return 0.0

    calculator.calculate_reward_simple = mock_calculate_reward_simple

    try:
        calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=5000000.0,
            position=0.5,
            portfolio_value=200000.0,
            atr=50000.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=1000.0,
            old_position=0.0,
            step=1,
            observation=np.array([5000000.0, 0.5, 1000.0]),
            reward_history=[],
            portfolio_value_history=[],
        )

        assert (
            passed_transaction_cost == 0.001
        ), f"Expected transaction_cost=0.001, got {passed_transaction_cost}"
    finally:
        calculator.calculate_reward_simple = original_method


def test_no_stdout_prints():
    """Test that reward calculator doesn't print to stdout."""
    import io
    from contextlib import redirect_stdout

    config = MockConfig()
    reward_settings = {
        "profit_weight": 1.0,
        "risk_weight": 1.0,
        "consistency_weight": 1.0,
        "ultra_profit_multiplier": 1.0,
        "ultra_risk_multiplier": 1.0,
        "reward_scale": 1.0,
        "reward_clip_min": -10.0,
        "reward_clip_max": 10.0,
        "use_simple_reward": True,
        "profit_multiplier": 0.01,
        "loss_penalty_multiplier": 0.01,
        "hold_penalty_rate": 0.01,
        "balance_penalty_tolerance": 0.15,
        "balance_penalty": 1.0,
        "trading_bonus_multiplier": 2.0,
        "trading_bonus": 0.01,
    }

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Capture stdout during reward calculation
    f = io.StringIO()
    with redirect_stdout(f):
        reward = calculator.calculate_reward_simple(
            pnl=1000.0,
            portfolio_value=200000.0,
            position=0.5,
            old_position=0.0,
            action=ACTION_BUY,
            reward_history=[],
            portfolio_value_history=[],
            current_price=5000000.0,
            step=1,
            transaction_cost=0.0,
        )

    output = f.getvalue()
    assert output == "", f"Unexpected stdout output: {output}"


def test_get_current_regime():
    """Test get_current_regime method works correctly."""
    config = MockConfig()
    reward_settings = {
        "profit_weight": 1.0,
        "risk_weight": 1.0,
        "consistency_weight": 1.0,
        "ultra_profit_multiplier": 1.0,
        "ultra_risk_multiplier": 1.0,
        "reward_scale": 1.0,
        "reward_clip_min": -10.0,
        "reward_clip_max": 10.0,
        "use_simple_reward": False,
        "profit_multiplier": 0.01,
        "loss_penalty_multiplier": 0.01,
        "hold_penalty_rate": 0.01,
        "balance_penalty_tolerance": 0.15,
        "balance_penalty": 1.0,
        "trading_bonus_multiplier": 2.0,
        "trading_bonus": 0.01,
    }

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Test get_current_regime
    regime = calculator.get_current_regime(current_price=5000000.0, step=1)
    assert isinstance(regime, str)
    assert regime in ["bull", "bear", "sideways", "volatile", "unknown"]


if __name__ == "__main__":
    pytest.main([__file__])

    # Test scenarios
    test_cases = [
        {
            "name": "Profitable BUY",
            "action": ACTION_BUY,
            "pnl": 1000.0,
            "position": 0.5,
            "current_price": 5000000.0,
            "atr": 50000.0,
        },
        {
            "name": "Loss SELL",
            "action": ACTION_SELL,
            "pnl": -500.0,
            "position": -0.3,
            "current_price": 4800000.0,
            "atr": 45000.0,
        },
        {
            "name": "HOLD with position",
            "action": ACTION_HOLD,
            "pnl": 0.0,
            "position": 0.8,
            "current_price": 4900000.0,
            "atr": 40000.0,
        },
    ]

    print("Testing reward calculation with extreme aggressive parameters:")
    print("=" * 60)

    for test_case in test_cases:
        reward = calculator._calculate_ultra_profit_reward(
            action=test_case["action"],
            atr_normalised=test_case["pnl"] / test_case["atr"],
            portfolio_return=test_case["pnl"] / 200000.0,
            position=test_case["position"],
            effective_max_position=1.0,
            current_price=test_case["current_price"],
            atr=test_case["atr"],
            pnl=test_case["pnl"],
            reward_scaling=1.0,  # Fixed scaling
        )

        print(f"{test_case['name']}:")
        print(f"  Action: {test_case['action']} (0=HOLD, 1=BUY, 2=SELL)")
        print(f"  PnL: {test_case['pnl']:.2f}")
        print(f"  Position: {test_case['position']:.2f}")
        print(f"  Reward: {reward:.4f}")
        print()


if __name__ == "__main__":
    test_reward_calculation()
