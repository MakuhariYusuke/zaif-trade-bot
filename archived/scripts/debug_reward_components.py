#!/usr/bin/env python3
"""
Debug Reward Function Components - Detailed Analysis

Analyzes each component of the reward function to find the asymmetry source.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def debug_reward_components():
    """Debug each component of the reward function."""

    print("=" * 80)
    print("REWARD FUNCTION COMPONENT DEBUG")
    print("=" * 80)

    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,
        reward_scaling=1.0,
        reward_clip_value=1.0,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 1.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
            "buy_action_penalty": 0.0,
            "sell_action_penalty": 0.0,
            "hold_action_penalty": 0.0,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
        },
    )

    import pandas as pd

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)

    # Test cases that showed asymmetry
    test_cases = [
        {
            "name": "BUY with loss",
            "action": 1,
            "pnl": -50.0,
            "position": 0.01,
            "old_position": 0.0,
            "portfolio_value": 200000.0,
        },
        {
            "name": "SELL with loss",
            "action": 2,
            "pnl": -50.0,
            "position": 0.0,
            "old_position": 0.01,
            "portfolio_value": 200000.0,
        },
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")

        # Call reward calculator directly to get components
        reward = env.reward_calculator.calculate_reward_simple(
            pnl=case["pnl"],
            portfolio_value=case["portfolio_value"],
            position=case["position"],
            old_position=case["old_position"],
            action=case["action"],
            reward_history=[],
            portfolio_value_history=[200000.0] * 30,
        )

        print(f"Final reward: {reward:+.6f}")

        # Now let's manually calculate each component to find the difference
        reward_scale = 1.0
        pnl_reward = case["pnl"] * reward_scale
        print(f"1. PNL reward: {pnl_reward:+.6f}")

        # Risk adjusted reward (simplified)
        risk_adjusted_reward = pnl_reward * 0.1
        print(f"2. Risk adjusted: {risk_adjusted_reward:+.6f}")

        # Position size bonus
        position_size_bonus = env.reward_calculator._calculate_position_size_bonus(
            case["position"], case["old_position"]
        )
        print(f"3. Position size bonus: {position_size_bonus:+.6f}")

        # Convert action to discrete
        discrete_action = env.reward_calculator._convert_continuous_to_discrete_action(
            case["action"]
        )
        print(f"4. Discrete action: {discrete_action}")

        # Action balance bonus
        action_balance_bonus = env.reward_calculator._calculate_action_balance_bonus(
            discrete_action
        )
        print(f"5. Action balance bonus: {action_balance_bonus:+.6f}")

        # Win rate bonus
        win_rate_bonus = env.reward_calculator._calculate_win_rate_bonus(
            discrete_action, case["pnl"]
        )
        print(f"6. Win rate bonus: {win_rate_bonus:+.6f}")

        # Trading bonus
        trading_bonus = 0.0
        if discrete_action in [1, 2]:  # BUY or SELL
            position_change = abs(case["position"] - case["old_position"])
            if position_change > 0.001:
                trading_bonus = 2.0
        print(f"7. Trading bonus: {trading_bonus:+.6f}")

        # Drawdown penalty
        drawdown_penalty = env.reward_calculator._calculate_drawdown_penalty(
            case["portfolio_value"], [200000.0] * 30
        )
        print(f"8. Drawdown penalty: {drawdown_penalty:+.6f}")

        # Sum
        manual_total = (
            risk_adjusted_reward
            + position_size_bonus
            + action_balance_bonus
            + win_rate_bonus
            + trading_bonus
            + drawdown_penalty
        )
        print(f"Manual total: {manual_total:+.6f}")
        print(f"Difference: {abs(reward - manual_total):.6f}")


def check_action_conversion():
    """Check how continuous actions are converted to discrete."""

    print("\n" + "=" * 80)
    print("ACTION CONVERSION CHECK")
    print("=" * 80)

    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,
        reward_scaling=1.0,
        reward_clip_value=1.0,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 1.0,
            "reward_clip_min": -1.0,
            "reward_clip_max": 1.0,
        },
    )

    import pandas as pd

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)

    test_actions = [1, 2]  # BUY and SELL as integers

    for action in test_actions:
        discrete = env.reward_calculator._convert_continuous_to_discrete_action(action)
        print(f"Action {action} -> Discrete {discrete}")


if __name__ == "__main__":
    debug_reward_components()
    check_action_conversion()
