#!/usr/bin/env python3
"""
Debug SAC v433 Model Predictions

Analyze what actions the v433 model is predicting during backtest.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def debug_model_predictions():
    """Debug model predictions to understand why no trades are executed."""

    try:
        logger.info("🔍 Debugging SAC v433 Model Predictions")

        # Load model
        model_path = "checkpoints/sac_v433_production_migration.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = SAC.load(model_path)
        logger.info("✅ Model loaded successfully")

        # Load data
        data_path = "data/btc_jpy_real_dataset.csv"
        df = pd.read_csv(data_path)
        logger.info(f"✅ Data loaded: {len(df)} rows")

        # Create environment
        env_config = EnvironmentConfig(
            reward_scaling=1.0,
            transaction_cost=0.0015,
            max_position_size=1.0,
            reward_position_penalty_scale=0.1,
            use_continuous_actions=True,
        )

        env = HeavyTradingEnv(df=df, config=env_config, random_start=False)
        logger.info("✅ Environment created")

        # Reset environment
        obs, info = env.reset()
        logger.info("✅ Environment reset")

        # Analyze first 100 steps
        actions_taken = []
        rewards_received = []
        action_counts = {"sell": 0, "hold": 0, "buy": 0}

        print("\n🎯 ANALYZING FIRST 100 PREDICTIONS:")
        print("-" * 50)

        for step in range(100):
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)

            # Convert action to readable format
            action_value = action[0] if isinstance(action, np.ndarray) else action

            # Classify action (using v433 thresholds: sell=-0.04, buy=0.04)
            if action_value <= -0.04:
                action_type = "sell"
                action_counts["sell"] += 1
            elif action_value >= 0.04:
                action_type = "buy"
                action_counts["buy"] += 1
            else:
                action_type = "hold"
                action_counts["hold"] += 1

            actions_taken.append(action_value)

            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            rewards_received.append(reward)

            # Print every 10 steps
            if step % 10 == 0:
                print(
                    f"Step {step:3d}: Action={action_value:+.6f} ({action_type}) Reward={reward:+.6f}"
                )
        print("-" * 50)
        print("📊 ACTION DISTRIBUTION (First 100 steps):")
        print(
            f"   SELL: {action_counts['sell']} ({action_counts['sell']/100*100:.1f}%)"
        )
        print(
            f"   HOLD: {action_counts['hold']} ({action_counts['hold']/100*100:.1f}%)"
        )
        print(f"   BUY:  {action_counts['buy']} ({action_counts['buy']/100*100:.1f}%)")

        print("\n🔧 MODEL CONFIGURATION:")
        print(f"   Policy Network: {model.policy.net_arch}")
        print(f"   Learning Rate: {model.learning_rate}")
        print(f"   Tau: {model.tau:.6f}")
        print(f"   Gamma: {model.gamma}")
        print(f"   Buffer Size: {model.replay_buffer.buffer_size}")

        print("\n⚙️  TRAINING CONFIGURATION (v433):")
        print("   HOLD Bonus: -0.002 (very low - discourages holding)")
        print("   SELL/BUY Bonus: 0.4")
        print("   Action Thresholds: SELL ≤ -0.04, BUY ≥ 0.04")
        print("   HOLD Range: [-0.04, 0.04]")

        # Analyze reward structure
        print("\n💡 ANALYSIS:")
        if action_counts["hold"] > 80:
            print("   ❌ PROBLEM: Model predicts HOLD in 80%+ of cases")
            print("   ❌ REASON: HOLD bonus (-0.002) is too low, model avoids holding")
            print(
                "   ❌ RESULT: No trades executed, portfolio decays due to transaction costs"
            )
        else:
            print("   ✅ Action distribution looks reasonable")

        return action_counts

    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        raise


if __name__ == "__main__":
    debug_model_predictions()
