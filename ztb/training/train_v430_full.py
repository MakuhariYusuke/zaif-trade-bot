#!/usr/bin/env python3
"""
SAC v430 Full Training - Standard Mode
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct imports to avoid complex dependencies
try:
    import pandas as pd
    from stable_baselines3 import SAC
    from ztb.utils.training_utils import create_checkpoint_callback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Import environment and data handling
    from ztb.trading.environment import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig
    from ztb.utils.logging_utils import get_logger
    from ztb.utils.training_utils import save_model
    from stable_baselines3.common.callbacks import BaseCallback

    logger = get_logger(__name__)

except ImportError as e:
    print(f"Import error: {e}")
    print("Required packages not available. Please install dependencies.")
    sys.exit(1)


class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress."""

    def __init__(self, verbose=0, eval_freq=1000):
        super().__init__(verbose)
        self.start_time = time.time()
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            elapsed = time.time() - self.start_time
            print(f"Step {self.n_calls}: {elapsed:.1f}s elapsed")

            # Log episode stats if available
            if len(self.episode_rewards) > 0:
                sum(self.episode_rewards[-10:]) / min(
                    10, len(self.episode_rewards)
                )
                print(".3f")
        return True

    def _on_rollout_end(self) -> None:
        """Called when a rollout ends."""
        if hasattr(self.locals, "episode_rewards"):
            self.episode_rewards.extend(self.locals["episode_rewards"])
        if hasattr(self.locals, "episode_lengths"):
            self.episode_lengths.extend(self.locals["episode_lengths"])




def run_full_training():
    """Run full SAC v430 training."""

    print("🚀 SAC v430 Full Training - Standard Mode")
    print("=" * 60)

    # Load optimized config
    config = load_optimized_config()
    training_config = config["training"]
    reward_config = config["reward_function"]
    sac_params = training_config

    print("📋 Training Configuration:")
    print(f"   Total timesteps: {training_config['total_timesteps']:,}")
    print(f"   Learning rate: {sac_params['learning_rate']:.6f}")
    print(f"   Batch size: {sac_params['batch_size']}")
    print(f"   Buffer size: {sac_params['buffer_size']:,}")
    print(f"   Gamma: {sac_params['gamma']:.4f}")
    print(f"   Tau: {sac_params['tau']:.4f}")
    print(f"   Reward scale: {reward_config['reward_scale']:.1f}")
    print()

    try:
        # Load data
        print("📊 Loading trading data...")
        data_path = "data/btc_jpy_real_dataset.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} rows of data")

        # Create environment config
        print("⚙️  Creating environment configuration...")
        env_config_obj = EnvironmentConfig(
            reward_scaling=reward_config["reward_scale"],
            transaction_cost=0.0005,  # Default transaction cost
            max_position_size=0.01,  # Default max position size
            reward_position_penalty_scale=reward_config["trading_bonus"],
            use_continuous_actions=True,  # Enable continuous actions for SAC
        )

        # Create environment
        print("🚀 Creating trading environment...")
        env = HeavyTradingEnv(df=df, config=env_config_obj, random_start=True)

        # Wrap environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Create SAC model
        print("🤖 Creating SAC model...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_params["learning_rate"],
            buffer_size=sac_params["buffer_size"],
            learning_starts=sac_params["learning_starts"],
            batch_size=sac_params["batch_size"],
            tau=sac_params["tau"],
            gamma=sac_params["gamma"],
            ent_coef=0.01,  # Convert from "auto_0.01"
            target_entropy=-2.0,  # Convert from "auto"
            verbose=1,
            device="auto",
            tensorboard_log="tensorboard/sac_v430_full",
        )

        # Create callbacks
        training_callback = TrainingCallback(eval_freq=5000)
        checkpoint_callback = create_checkpoint_callback(
            save_freq=10000,
            save_path="models/sac_v430_checkpoints",
            name_prefix="sac_v430",
        )

        # Run training
        print("🎯 Starting full training...")
        print(f"Target: {training_config['total_timesteps']:,} timesteps")
        start_time = time.time()

        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=[training_callback, checkpoint_callback],
            progress_bar=True,
        )

        training_time = time.time() - start_time

        print()
        print("=" * 60)
        print("✅ Full training completed successfully!")
        print(f"⏱️  Total training time: {training_time:.2f} seconds")
        print("📊 Final model saved to: models/sac_v430_full/final_model.zip")

        # Save final model
        model_path = "models/sac_v430_full/final_model.zip"
        save_model(model, model_path)

        # Check if model file exists
        if os.path.exists(model_path):
            print("📊 Final model verified: EXISTS")
        else:
            print("⚠️  Final model not found")

        print("=" * 60)
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Training failed!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        return False


def main():
    """Main function."""
    try:
        success = run_full_training()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
