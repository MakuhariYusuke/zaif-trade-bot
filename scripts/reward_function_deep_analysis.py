#!/usr/bin/env python3
"""
Deep Analysis of SAC SELL Bias - Reward Function Investigation

Investigates the fundamental cause of SELL bias in SAC reward function.
Analyzes reward components for BUY vs SELL actions to identify asymmetries.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = get_logger(__name__)

def create_test_environment() -> HeavyTradingEnv:
    """Create environment with detailed reward logging."""
    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,
        reward_scaling=8000.0,
        reward_clip_value=80.0,
        reward_settings={
            "use_simple_reward": True,
            "reward_scale": 8000.0,
            "reward_clip_min": -80.0,
            "reward_clip_max": 80.0,
            "profit_bonus_multipliers": [1.0, 1.0, 0.8],  # BUY, SELL, HOLD
            "buy_action_penalty": -2.0,
            "sell_action_penalty": -2.0,
            "hold_action_penalty": 0.0,
            "base_action_penalty": 0.015,
            "loss_penalty_coeff": -0.2,
            "base_profit_bonus_atr_coeff": 1.5,
            "base_profit_bonus_portfolio_coeff": 1.2,
        }
    )

    # Load sample data
    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)
    return env

def analyze_reward_components(env: HeavyTradingEnv, action: int, pnl: float, position: float, old_position: float) -> Dict[str, Any]:
    """Analyze reward components for a specific action."""

    # Set up test conditions
    current_price = env._resolve_price()
    atr = env._resolve_atr()
    portfolio_value = env.portfolio_value
    reward_scaling = 8000.0

    # Enable debug logging in reward calculator
    env.reward_calculator.logger.setLevel(logging.DEBUG)

    # Calculate reward
    reward = env.reward_calculator.calculate_reward(
        action=action,
        current_price=current_price,
        position=position,
        portfolio_value=portfolio_value,
        atr=atr,
        transaction_cost=0.0,
        reward_scaling=reward_scaling,
        pnl=pnl,
        old_position=old_position,
        step=1,
        observation=None,
        reward_history=[],
        portfolio_value_history=[portfolio_value] * 30
    )

    # Extract components from reward calculator if possible
    components = {
        "action": action,
        "pnl": pnl,
        "position": position,
        "old_position": old_position,
        "current_price": current_price,
        "atr": atr,
        "portfolio_value": portfolio_value,
        "final_reward": reward,
        "scaled_reward": reward * reward_scaling
    }

    return components

def run_comprehensive_analysis():
    """Run comprehensive analysis of reward function asymmetries."""

    print("=" * 80)
    print("DEEP ANALYSIS: SAC SELL BIAS - REWARD FUNCTION INVESTIGATION")
    print("=" * 80)

    env = create_test_environment()

    # Test scenarios
    test_scenarios = [
        # Profit scenarios
        {"pnl": 100.0, "position": 0.01, "old_position": 0.0, "description": "BUY with profit"},
        {"pnl": 100.0, "position": 0.0, "old_position": 0.01, "description": "SELL with profit"},
        {"pnl": 0.0, "position": 0.01, "old_position": 0.01, "description": "HOLD with no change"},

        # Loss scenarios
        {"pnl": -100.0, "position": 0.01, "old_position": 0.0, "description": "BUY with loss"},
        {"pnl": -100.0, "position": 0.0, "old_position": 0.01, "description": "SELL with loss"},

        # Position changes
        {"pnl": 50.0, "position": 0.02, "old_position": 0.01, "description": "Increase position with profit"},
        {"pnl": 50.0, "position": 0.0, "old_position": 0.01, "description": "Close position with profit"},
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n--- Testing: {scenario['description']} ---")

        # Test BUY action
        buy_result = analyze_reward_components(
            env, action=1,  # BUY
            pnl=scenario["pnl"],
            position=scenario["position"],
            old_position=scenario["old_position"]
        )

        # Test SELL action
        sell_result = analyze_reward_components(
            env, action=2,  # SELL
            pnl=scenario["pnl"],
            position=scenario["position"],
            old_position=scenario["old_position"]
        )

        # Test HOLD action
        hold_result = analyze_reward_components(
            env, action=0,  # HOLD
            pnl=scenario["pnl"],
            position=scenario["position"],
            old_position=scenario["old_position"]
        )

        # Compare rewards
        buy_reward = buy_result["final_reward"]
        sell_reward = sell_result["final_reward"]
        hold_reward = hold_result["final_reward"]

        reward_diff_buy_sell = buy_reward - sell_reward
        reward_diff_sell_buy = sell_reward - buy_reward

        print(f"BUY reward:  {buy_reward:.4f}")
        print(f"SELL reward: {sell_reward:.4f}")
        print(f"HOLD reward: {hold_reward:.4f}")
        print(f"BUY-SELL diff: {reward_diff_buy_sell:.4f}")
        print(f"SELL-BUY diff: {reward_diff_sell_buy:.4f}")
        # Check for bias
        bias_threshold = 0.1
        if abs(reward_diff_buy_sell) > bias_threshold:
            if reward_diff_buy_sell > 0:
                print(f"⚠️  BUY bias detected: BUY gets {abs(reward_diff_buy_sell):.4f} more reward")
            else:
                print(f"⚠️  SELL bias detected: SELL gets {abs(reward_diff_buy_sell):.4f} more reward")

        result = {
            "scenario": scenario["description"],
            "pnl": scenario["pnl"],
            "position": scenario["position"],
            "old_position": scenario["old_position"],
            "buy_reward": buy_reward,
            "sell_reward": sell_reward,
            "hold_reward": hold_reward,
            "buy_sell_diff": reward_diff_buy_sell,
            "sell_buy_diff": reward_diff_sell_buy
        }
        results.append(result)

    # Summary analysis
    print("\n" + "=" * 80)
    print("SUMMARY ANALYSIS")
    print("=" * 80)

    total_buy_bias = 0
    total_sell_bias = 0
    bias_count = 0

    for result in results:
        diff = result["buy_sell_diff"]
        if abs(diff) > 0.1:
            bias_count += 1
            if diff > 0:
                total_buy_bias += diff
                print(f"BUY bias in scenario '{result['scenario']}': BUY gets {diff:.4f} more reward")
            else:
                total_sell_bias += abs(diff)
                print(f"SELL bias in scenario '{result['scenario']}': SELL gets {abs(diff):.4f} more reward")
    print(f"\nTotal scenarios with bias: {bias_count}/{len(results)}")
    if total_buy_bias > total_sell_bias:
        print(f"Overall BUY bias detected: {total_buy_bias:.4f} total bias points")
    elif total_sell_bias > total_buy_bias:
        print(f"Overall SELL bias detected: {total_sell_bias:.4f} total bias points")
    else:
        print("No significant overall bias detected")

    # Save detailed results
    output_file = "results/reward_function_deep_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_type": "reward_function_deep_analysis",
            "test_scenarios": test_scenarios,
            "results": results,
            "summary": {
                "total_scenarios": len(results),
                "biased_scenarios": bias_count,
                "total_buy_bias": total_buy_bias,
                "total_sell_bias": total_sell_bias
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")

    return results

if __name__ == "__main__":
    run_comprehensive_analysis()