#!/usr/bin/env python3
"""
Environment Debug - Investigate why both SAC and PPO produce constant BUY bias
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def debug_environment():
    """Debug the HeavyTradingEnv to find the root cause of constant BUY bias."""

    print("=" * 80)
    print("ENVIRONMENT DEBUG - ROOT CAUSE ANALYSIS")
    print("=" * 80)

    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    # Create environment with zero transaction costs and neutral rewards
    config = EnvironmentConfig(
        max_position_size=0.01,
        transaction_cost=0.0,  # No transaction costs
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

    df = pd.read_csv("btc_jpy_real_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    env = HeavyTradingEnv(df=df, config=config)

    print("Environment Configuration:")
    print(f"- Max position size: {config.max_position_size}")
    print(f"- Transaction cost: {config.transaction_cost}")
    print(f"- Reward scaling: {config.reward_scaling}")
    print(f"- Action space: {env.action_space}")
    print(f"- Observation space: {env.observation_space}")
    print()

    # Test environment reset
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial position: {env.position}")
    print(f"Initial portfolio value: {env.portfolio_value}")
    print()

    # Test different actions and observe state transitions
    test_actions = [
        ("HOLD (0)", 0.0),
        ("SMALL BUY (0.1)", 0.1),
        ("MEDIUM BUY (0.5)", 0.5),
        ("LARGE BUY (0.9)", 0.9),
        ("SMALL SELL (-0.1)", -0.1),
        ("MEDIUM SELL (-0.5)", -0.5),
        ("LARGE SELL (-0.9)", -0.9),
    ]

    print("Testing Action Transitions:")
    print("-" * 60)

    for action_name, action_value in test_actions:
        # Reset environment for each test
        obs = env.reset()
        initial_position = env.position
        initial_portfolio = env.portfolio_value

        # Take action
        next_obs, reward, done, truncated, info = env.step(action_value)

        final_position = env.position
        final_portfolio = env.portfolio_value

        print(
            f"{action_name:15}: {initial_position:.4f} -> {final_position:.4f} | "
            f"Reward: {reward:+.6f} | Portfolio: {initial_portfolio:.0f} -> {final_portfolio:.0f}"
        )

    print()

    # Test reward function directly
    print("Direct Reward Function Testing:")
    print("-" * 60)

    reward_test_cases = [
        {
            "pnl": 100.0,
            "position": 0.01,
            "old_position": 0.0,
            "action": 1,
            "desc": "BUY with profit",
        },
        {
            "pnl": 100.0,
            "position": 0.0,
            "old_position": 0.01,
            "action": 2,
            "desc": "SELL with profit",
        },
        {
            "pnl": 0.0,
            "position": 0.01,
            "old_position": 0.01,
            "action": 0,
            "desc": "HOLD neutral",
        },
        {
            "pnl": -50.0,
            "position": 0.01,
            "old_position": 0.0,
            "action": 1,
            "desc": "BUY with loss",
        },
        {
            "pnl": -50.0,
            "position": 0.0,
            "old_position": 0.01,
            "action": 2,
            "desc": "SELL with loss",
        },
    ]

    for case in reward_test_cases:
        reward = env.reward_calculator.calculate_reward_simple(
            pnl=case["pnl"],
            portfolio_value=200000.0,
            position=case["position"],
            old_position=case["old_position"],
            action=case["action"],
            reward_history=[],
            portfolio_value_history=[200000.0] * 30,
        )
        print(f"{case['desc']:20}: {reward:+.6f}")

    print()

    # Test observation consistency
    print("Observation Analysis:")
    print("-" * 60)

    obs_array = obs[0] if isinstance(obs, tuple) else obs
    print(f"Observation shape: {np.array(obs_array).shape}")
    print(f"Observation values: {obs_array}")

    # Check if observations are consistent
    consistent_obs = []
    for i in range(10):
        obs = env.reset()
        consistent_obs.append(obs)

    obs_array = np.array(consistent_obs)
    print(f"Observation std across resets: {np.std(obs_array, axis=0)}")
    print(f"Are observations identical? {np.allclose(obs_array[0], obs_array[1])}")

    print()

    # Test action conversion logic
    print("Action Conversion Logic:")
    print("-" * 60)

    # Test the conversion from continuous to discrete actions
    continuous_actions = np.linspace(-1, 1, 21)
    buy_threshold = 0.1
    sell_threshold = -0.1

    print("Continuous -> Discrete Action Mapping:")
    for action in continuous_actions:
        if action > buy_threshold:
            discrete = "BUY"
        elif action < sell_threshold:
            discrete = "SELL"
        else:
            discrete = "HOLD"
        print(f"{action:+.2f} -> {discrete}")

    print()

    # Check for any environment biases
    print("Environment Bias Analysis:")
    print("-" * 60)

    # Test if environment naturally favors certain actions
    env.reset()
    random_actions = np.random.uniform(-1, 1, 100)
    rewards = []

    for action in random_actions:
        obs = env.reset()  # Reset for each action to test independently
        next_obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)

    rewards = np.array(rewards)
    print(
        f"Random action rewards - Mean: {np.mean(rewards):+.6f}, Std: {np.std(rewards):.6f}"
    )
    print(f"Reward range: {np.min(rewards):+.6f} to {np.max(rewards):+.6f}")

    # Check if certain action ranges get better rewards
    buy_actions = random_actions[random_actions > buy_threshold]
    sell_actions = random_actions[random_actions < sell_threshold]
    hold_actions = random_actions[
        (random_actions >= sell_threshold) & (random_actions <= buy_threshold)
    ]

    if len(buy_actions) > 0:
        buy_rewards = [env.step(a)[1] for a in buy_actions[:10]]  # Test first 10
        print(f"BUY action rewards (sample): {np.mean(buy_rewards):+.6f}")

    if len(sell_actions) > 0:
        sell_rewards = [env.step(a)[1] for a in sell_actions[:10]]  # Test first 10
        print(f"SELL action rewards (sample): {np.mean(sell_rewards):+.6f}")

    if len(hold_actions) > 0:
        hold_rewards = [env.step(a)[1] for a in hold_actions[:10]]  # Test first 10
        print(f"HOLD action rewards (sample): {np.mean(hold_rewards):+.6f}")


def main():
    debug_environment()

    print("\n" + "=" * 80)
    print("ROOT CAUSE HYPOTHESES")
    print("=" * 80)
    print("1. Environment state transitions favor BUY actions")
    print("2. Reward function has hidden BUY bias despite appearing neutral")
    print("3. Action conversion thresholds are too restrictive")
    print("4. Stable Baselines3 has issues with this environment")
    print("5. Dataset characteristics naturally favor BUY actions")
    print()
    print("Next steps:")
    print("- Test with different random seeds for environment")
    print("- Check if bias exists even with random reward function")
    print("- Investigate Stable Baselines3 source code")
    print("- Try different RL libraries (Ray RLlib, CleanRL)")
    print("=" * 80)


if __name__ == "__main__":
    main()
