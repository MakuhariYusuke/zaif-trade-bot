#!/usr/bin/env python3
"""
Debug script to check what actions the trained model predicts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ztb.trading.environment import HeavyTradingEnv

def test_model_predictions():
    """Test what actions the trained model predicts."""

    # Load the latest trained model
    model_path = Path("models/test_high_entropy_ent_coef_0_200.zip")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = PPO.load(str(model_path))

    # Load data
    df = pd.read_csv('ml-dataset-enhanced.csv')

    # Create environment
    env_config = {
        'reward_scaling': 6.0,
        'transaction_cost': 0.001,
        'max_position_size': 1.0,
        'risk_free_rate': 0.02,
        'feature_set': 'full',
        'initial_portfolio_value': 1000000.0,
        'curriculum_stage': 'simple_portfolio',
        'reward_settings': {
            'enable_forced_diversity': False,
            'profit_bonus_multipliers': [1.0, 1.0, 1.0]
        },
    }

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68
    )

    print("=== Testing Model Predictions ===")

    # Reset environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Test predictions for first 10 steps
    actions_predicted = []
    action_probs = []

    for i in range(10):
        # Get action probabilities
        action, _ = model.predict(obs, deterministic=False)
        action = int(action)  # Convert to int
        actions_predicted.append(action)

        # Get action probabilities (need to use the policy directly)
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        actions_dist = model.policy.get_distribution(obs_tensor)
        if actions_dist.distribution is not None:
            probs = actions_dist.distribution.probs.detach().cpu().numpy()[0]
            action_probs.append(probs)
            print(f"Step {i+1}: Action={action}, Probs=[H:{probs[0]:.4f}, B:{probs[1]:.4f}, S:{probs[2]:.4f}]")
        else:
            print(f"Step {i+1}: Action={action}, Probs=Unable to get")

        # Take the action
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print("Episode ended")
            break

    print("\n=== Summary ===")
    print(f"Actions predicted: {actions_predicted}")
    print(f"Unique actions: {set(actions_predicted)}")

    # Average probabilities
    if action_probs:
        avg_probs = np.mean(action_probs, axis=0)
        print(f"Average probabilities: [H:{avg_probs[0]:.4f}, B:{avg_probs[1]:.4f}, S:{avg_probs[2]:.4f}]")

    env.close()

if __name__ == "__main__":
    test_model_predictions()