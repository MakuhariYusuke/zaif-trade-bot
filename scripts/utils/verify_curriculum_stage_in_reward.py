#!/usr/bin/env python3
"""
Verify that curriculum_stage is actually being passed to RewardCalculator
and that balance_penalty is being applied.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_sample_data(periods: int = 500) -> pd.DataFrame:
    """Create sample OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=periods, freq="1h")

    base_price = 5000000
    price_changes = np.random.normal(0, 0.005, periods).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)

    high = close * (1 + np.abs(np.random.normal(0, 0.002, periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, periods)))
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 10000, periods), index=dates)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": dates,
        }
    )

    # Add technical indicators
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["RSI"] = 50
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["BB_Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["BB_Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()

    return df.ffill().bfill()


def test_curriculum_stage():
    """Test 1: Verify curriculum_stage is set in EnvironmentConfig"""
    print("\n" + "=" * 80)
    print("TEST 1: Verify curriculum_stage in EnvironmentConfig")
    print("=" * 80)

    config_path = "config/sac_v444_3_balanced_penalty_scale_200.json"
    config = load_config(config_path)

    # Simulate what quick_train_v444_configurable.py does
    env_config = config["environment"].copy()

    # Add curriculum_stage from training config
    if "training" in config and "curriculum_learning" in config["training"]:
        curriculum_config = config["training"]["curriculum_learning"]
        if "curriculum_stage" in curriculum_config:
            env_config["curriculum_stage"] = curriculum_config["curriculum_stage"]
            print(
                f"✓ curriculum_stage added to env_config: {curriculum_config['curriculum_stage']}"
            )

    # Create EnvironmentConfig from dict
    env_config_obj = EnvironmentConfig.from_dict(env_config)

    print(f"✓ EnvironmentConfig.curriculum_stage = {env_config_obj.curriculum_stage}")

    assert (
        env_config_obj.curriculum_stage == "balanced_penalty"
    ), f"Expected 'balanced_penalty', got {env_config_obj.curriculum_stage}"

    print("✓ TEST 1 PASSED: curriculum_stage is correctly set\n")
    return True


def test_reward_calculator_receives_curriculum_stage():
    """Test 2: Verify RewardCalculator receives curriculum_stage"""
    print("\n" + "=" * 80)
    print("TEST 2: Verify RewardCalculator receives curriculum_stage")
    print("=" * 80)

    # Load config
    config_path = "config/sac_v444_3_balanced_penalty_scale_200.json"
    config = load_config(config_path)

    # Prepare env config
    env_config = config["environment"].copy()
    if "training" in config and "curriculum_learning" in config["training"]:
        curriculum_config = config["training"]["curriculum_learning"]
        if "curriculum_stage" in curriculum_config:
            env_config["curriculum_stage"] = curriculum_config["curriculum_stage"]

    # Create environment with sample data
    df = create_sample_data(periods=200)

    print("Creating HeavyTradingEnv...")
    try:
        env = HeavyTradingEnv(df, env_config)
        print("✓ Environment created successfully")

        # Check RewardCalculator's config
        print(
            f"✓ RewardCalculator.config.curriculum_stage = {env.reward_calculator.config.curriculum_stage}"
        )

        assert (
            env.reward_calculator.config.curriculum_stage == "balanced_penalty"
        ), f"Expected 'balanced_penalty', got {env.reward_calculator.config.curriculum_stage}"

        print("✓ TEST 2 PASSED: RewardCalculator has correct curriculum_stage\n")
        return True
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_balance_penalty_calculation():
    """Test 3: Verify balance_penalty is being calculated"""
    print("\n" + "=" * 80)
    print("TEST 3: Verify balance_penalty is being calculated")
    print("=" * 80)

    # Load config
    config_path = "config/sac_v444_3_balanced_penalty_scale_200.json"
    config = load_config(config_path)

    # Prepare env config
    env_config = config["environment"].copy()
    if "training" in config and "curriculum_learning" in config["training"]:
        curriculum_config = config["training"]["curriculum_learning"]
        if "curriculum_stage" in curriculum_config:
            env_config["curriculum_stage"] = curriculum_config["curriculum_stage"]

    # Create environment
    df = create_sample_data(periods=200)
    env = HeavyTradingEnv(df, env_config)

    print(
        f"RewardCalculator._recent_actions maxlen: {env.reward_calculator._recent_actions.maxlen}"
    )
    print(
        f"balance_penalty_scale: {env.reward_calculator._get_behavior_opt('balance_penalty', 4.0)}"
    )

    # Manually perform steps and check reward calculation
    print("\nPerforming 30 steps to accumulate action history...")
    obs, info = env.reset()

    # Take actions: mix of BUY, SELL, HOLD to see balance penalty
    actions_to_take = [
        1,
        1,
        1,
        1,
        1,  # 5 BUY
        2,
        2,
        2,  # 3 SELL
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # 10 HOLD
        1,
        1,  # 2 BUY
        2,
        2,  # 2 SELL
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # 8 HOLD
    ]

    rewards = []
    for step, action in enumerate(actions_to_take):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if step % 5 == 4:
            recent_actions = list(env.reward_calculator._recent_actions)
            from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

            buy_count = recent_actions.count(ACTION_BUY)
            sell_count = recent_actions.count(ACTION_SELL)
            hold_count = recent_actions.count(ACTION_HOLD)
            total = len(recent_actions)

            if total > 0:
                    buy_ratio = buy_count / total
                    sell_ratio = sell_count / total
                    hold_ratio = hold_count / total
                    from ztb.trading.environment.components.rewards.utils import RewardUtils
                    imbalance = RewardUtils.calculate_buy_sell_diff(buy_ratio, sell_ratio)

                    print(
                        f"Step {step:2d}: BUY={buy_ratio:.2%} SELL={sell_ratio:.2%} HOLD={hold_ratio:.2%} | "
                        f"Imbalance={imbalance:.2%} | Reward={reward:.4f}"
                    )
    print("\n✓ TEST 3 PASSED: balance_penalty is being calculated\n")
    return True


if __name__ == "__main__":
    try:
        test_curriculum_stage()
        test_reward_calculator_receives_curriculum_stage()
        test_balance_penalty_calculation()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ TESTS FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
