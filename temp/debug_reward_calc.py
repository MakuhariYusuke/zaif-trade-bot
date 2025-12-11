import logging

import numpy as np

from backtest.data_generator import generate_synthetic_data
from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_reward():
    # 1. Generate Data
    df = generate_synthetic_data(n_periods=1000)

    # 2. Config
    env_config = EnvironmentConfig(
        initial_portfolio_value=1000000.0,
        transaction_cost=0.001,
        reward_scaling=1.0,
        max_position_size=0.5,
        stop_loss_threshold=0.05,
        use_continuous_actions=True,  # Enable continuous actions
        curriculum_learning={
            "enabled": True,
            "auto_progression": False,
            "initial_stage": "forced_balance",
            "forced_balance": {
                "penalty": {
                    "scale": 2.0,  # High penalty
                    "threshold_small": 0.05,
                    "value_small_deviation": 10.0,
                    "value_very_large_deviation": 100.0,
                },
                "bonus": {
                    "scale": 2.0,
                    "value_very_large_deviation": 100.0,
                },
            },
        },
        behavior_optimization={
            "balance_penalty": 1.0,
        },
    )

    # 3. Create Env
    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        feature_extractor=V4FeatureExtractor(),
    )

    obs = env.reset()

    print("\n--- Starting Debug ---")
    print(f"Curriculum Enabled: {env.reward_calculator.curriculum_manager.enabled}")
    print(
        f"Current Stage: {env.reward_calculator.curriculum_manager.get_current_stage()}"
    )

    # 4. Force Actions
    # We will send continuous actions that map to HOLD, BUY, SELL

    # HOLD for 150 steps (to fill buffer > 100)
    print("\n--- Phase 1: Force HOLD (150 steps) ---")
    for i in range(150):
        action = np.array([0.0])  # HOLD
        obs, reward, done, truncated, info = env.step(action)

        stage = info.get("stage", "unknown")
        base_reward = info.get("base_reward", 0.0)
        imbalance_penalty = info.get("imbalance_penalty", 0.0)

        if i % 20 == 0 or i > 140:
            print(
                f"Step {i}: Action=HOLD, Reward={reward:.4f}, Stage={stage}, Base={base_reward:.4f}, Penalty={imbalance_penalty:.4f}"
            )
            print(f"  Counts: {env.reward_calculator._action_counts}")

    # BUY for 50 steps
    print("\n--- Phase 2: Force BUY (50 steps) ---")
    for i in range(50):
        action = np.array([0.8])  # BUY
        obs, reward, done, truncated, info = env.step(action)

        stage = info.get("stage", "unknown")
        base_reward = info.get("base_reward", 0.0)
        corrective_bonus = info.get("corrective_bonus", 0.0)

        if i % 10 == 0:
            print(
                f"Step {150+i}: Action=BUY, Reward={reward:.4f}, Stage={stage}, Base={base_reward:.4f}, Bonus={corrective_bonus:.4f}"
            )
            print(f"  Counts: {env.reward_calculator._action_counts}")


if __name__ == "__main__":
    debug_reward()
