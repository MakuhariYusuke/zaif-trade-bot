"""
Create a base PPO model for BC warmstart.

Initializes a fresh MaskablePPO model with the trading environment,
ready for behavioral cloning warmstart training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from sb3_contrib import MaskablePPO
from ztb.trading.environment.environment import HeavyTradingEnv


def create_base_model(
    data_path: Path,
    output_path: Path,
    policy: str = "MlpPolicy",
    learning_rate: float = 3e-4,
):
    """
    Create a fresh PPO model.
    
    Args:
        data_path: Path to training data CSV
        output_path: Path to save initialized model
        policy: Policy network type
        learning_rate: Learning rate
    """
    print("=" * 60)
    print("Creating Base PPO Model")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print(f"Policy: {policy}")
    print(f"Learning rate: {learning_rate}")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ["action", "timestamp"]]
    features = df[feature_cols].values
    
    print(f"  Features: {len(feature_cols)} columns")
    print()
    
    # Create environment
    print("Creating environment...")
    env = HeavyTradingEnv(
        features=features,
        initial_balance=100000,
        transaction_cost=0.001,
        max_position_size=1.0,
    )
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    print()
    
    # Create model
    print("Initializing PPO model...")
    model = MaskablePPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
    )
    print("  Model created with default PPO hyperparameters")
    print()
    
    # Save
    print("Saving model...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"✅ Saved to: {output_path}")
    print()
    
    print("Summary:")
    print(f"  Base model ready for BC warmstart")
    print(f"  Use with: python scripts/bc_warmstart.py --model {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a base PPO model for BC warmstart"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("ml-dataset-final.csv"),
        help="Training data CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/base_ppo.zip"),
        help="Output model path",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        help="Policy network type",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    
    args = parser.parse_args()
    
    create_base_model(
        data_path=args.data,
        output_path=args.output,
        policy=args.policy,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
