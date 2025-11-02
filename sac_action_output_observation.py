#!/usr/bin/env python3
"""
SAC Action Output Observation Script

Train SAC model for 5000 steps and observe the action outputs,
especially the final 100 steps to check for value sticking at -1.0 or 1.0.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_minimal_env():
    """Create a minimal continuous action environment for testing."""
    # Use Pendulum environment which has continuous action space [-2, 2]
    env = gym.make("Pendulum-v1")
    # Wrap in vectorized environment for SB3 compatibility
    env = make_vec_env(lambda: env, n_envs=1)
    return env


def observe_model_actions(model, env, n_steps=100):
    """Observe actions from a trained model for n steps."""
    actions_observed = []

    obs = env.reset()
    for step in range(n_steps):
        # Get action from model (deterministic=False to see stochastic behavior)
        action, _ = model.predict(obs, deterministic=False)
        actions_observed.append(float(action[0]))

        # Take step in environment
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()

    return actions_observed


def analyze_action_outputs(actions: List[float]) -> Dict[str, Any]:
    """Analyze the distribution of action outputs."""
    actions_array = np.array(actions)

    analysis = {
        "total_actions": len(actions),
        "mean": float(np.mean(actions_array)),
        "std": float(np.std(actions_array)),
        "min": float(np.min(actions_array)),
        "max": float(np.max(actions_array)),
        "median": float(np.median(actions_array)),
        "q25": float(np.percentile(actions_array, 25)),
        "q75": float(np.percentile(actions_array, 75)),
        "q90": float(np.percentile(actions_array, 90)),
        "q95": float(np.percentile(actions_array, 95)),
        "q99": float(np.percentile(actions_array, 99)),
    }

    # Check for value sticking
    unique_values = np.unique(actions_array)
    analysis["unique_values_count"] = len(unique_values)

    # Check how many values are exactly -1.0 or 1.0
    exact_minus_1 = np.sum(np.abs(actions_array + 1.0) < 1e-6)
    exact_plus_1 = np.sum(np.abs(actions_array - 1.0) < 1e-6)

    analysis["exact_minus_1_count"] = int(exact_minus_1)
    analysis["exact_plus_1_count"] = int(exact_plus_1)
    analysis["exact_minus_1_percentage"] = float(exact_minus_1 / len(actions) * 100)
    analysis["exact_plus_1_percentage"] = float(exact_plus_1 / len(actions) * 100)

    # Check values near boundaries
    near_minus_1 = np.sum((actions_array >= -1.0) & (actions_array <= -0.9))
    near_plus_1 = np.sum((actions_array <= 1.0) & (actions_array >= 0.9))

    analysis["near_minus_1_count"] = int(near_minus_1)
    analysis["near_plus_1_count"] = int(near_plus_1)
    analysis["near_minus_1_percentage"] = float(near_minus_1 / len(actions) * 100)
    analysis["near_plus_1_percentage"] = float(near_plus_1 / len(actions) * 100)

    return analysis


def main():
    """Main function to run SAC training and observe action outputs."""
    logger.info("Starting SAC action output observation test")

    # Create environment
    env = create_minimal_env()
    logger.info("Environment created")

    # Create SAC model with minimal config
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        verbose=1,
        tensorboard_log="./sac_action_test_logs"
    )

    # Train for 5000 steps
    logger.info("Starting training for 5000 steps...")
    model.learn(total_timesteps=5000)

    # After training, observe actions from the trained model
    logger.info("Training completed. Observing actions from trained model...")
    observed_actions = observe_model_actions(model, env, n_steps=100)

    # Analyze observed actions
    if observed_actions:
        logger.info(f"Observed {len(observed_actions)} actions from trained model")
        analysis = analyze_action_outputs(observed_actions)

        print("\n=== SAC Action Output Analysis (Trained Model) ===")
        print(f"Total actions observed: {analysis['total_actions']}")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(f"Unique values: {analysis['unique_values_count']}")

        print("\n=== Boundary Value Analysis ===")
        print(f"Exact -1.0 values: {analysis['exact_minus_1_count']} ({analysis['exact_minus_1_percentage']:.2f}%)")
        print(f"Exact +1.0 values: {analysis['exact_plus_1_count']} ({analysis['exact_plus_1_percentage']:.2f}%)")
        print(f"Near -1.0 values (≥-1.0, ≤-0.9): {analysis['near_minus_1_count']} ({analysis['near_minus_1_percentage']:.2f}%)")
        print(f"Near +1.0 values (≤1.0, ≥0.9): {analysis['near_plus_1_count']} ({analysis['near_plus_1_percentage']:.2f}%)")

        # Show distribution of observed actions
        print("\nObserved action values (last 20):")
        for i, action in enumerate(observed_actions[-20:]):
            print(".4f")

        # Save results to file
        import json
        with open("sac_action_output_analysis.json", "w") as f:
            json.dump({
                "analysis": analysis,
                "observed_actions": observed_actions
            }, f, indent=2)

        logger.info("Analysis saved to sac_action_output_analysis.json")

    else:
        logger.warning("No actions observed")

    env.close()
    logger.info("Test completed")

    env.close()
    logger.info("Test completed")


if __name__ == "__main__":
    main()