#!/usr/bin/env python3
"""
Analyze action distribution from improved 5000-step trained SAC model
改良版5000ステップ学習済みSACモデルのアクション分布分析
"""

import json
import logging
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def create_improved_trading_env():
    """Create an improved trading environment with better reward function"""
    # Create simple price data
    np.random.seed(42)
    n_steps = 1000

    # Generate trending price data
    t = np.linspace(0, 4*np.pi, n_steps)
    trend = 0.1 * np.sin(t * 0.1)  # Long-term trend
    noise = np.random.normal(0, 0.005, n_steps)  # Short-term noise
    price_changes = trend + noise

    base_price = 5000000
    prices = base_price * (1 + np.cumsum(price_changes))

    # Create simple observation space (price, trend, position)
    class ImprovedTradingEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
            )
            self.current_step = 0
            self.balance = 200000.0
            self.position = 0.0
            self.prices = prices
            self.initial_balance = 200000.0
            self.prev_portfolio_value = self.initial_balance
            self.reset()

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0
            self.prev_portfolio_value = self.initial_balance
            return self._get_observation(), {}

        def step(self, action):
            # Simple trading logic
            action_value = float(action[0])

            # Execute trade
            price = self.prices[self.current_step]
            trade_executed = False

            if action_value > 0.05:  # Buy signal (lower threshold)
                if self.balance > price * 0.1:  # Can afford
                    self.position += 0.1
                    self.balance -= price * 0.1
                    trade_executed = True
            elif action_value < -0.05:  # Sell signal (lower threshold)
                if self.position > 0.1:
                    self.position -= 0.1
                    self.balance += price * 0.1
                    trade_executed = True

            # Calculate improved reward (portfolio return percentage)
            current_portfolio_value = self.balance + self.position * price
            portfolio_return = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value

            # Add small penalty for no action to encourage exploration
            action_penalty = 0.0
            if not trade_executed:
                action_penalty = -0.0001  # Small penalty for inaction

            reward = portfolio_return + action_penalty

            self.prev_portfolio_value = current_portfolio_value
            self.current_step += 1
            done = self.current_step >= len(self.prices) - 1

            return self._get_observation(), reward, done, False, {}

        def _get_observation(self):
            price = self.prices[self.current_step]
            # Simple features: price, trend, position, balance_ratio, step_ratio
            trend = (price - self.prices[max(0, self.current_step-10)]) / self.prices[max(0, self.current_step-10)]
            return np.array([
                price / 10000000,  # Normalized price
                trend,  # Price trend
                self.position,  # Current position
                self.balance / self.initial_balance,  # Balance ratio
                self.current_step / len(self.prices)  # Time progress
            ], dtype=np.float32)

    return ImprovedTradingEnv()


def analyze_improved_action_distribution(model_path, n_episodes=10):
    """Analyze action distribution from improved trained model"""
    logger.info(f"Analyzing action distribution from improved model: {model_path}")

    # Load model
    model = SAC.load(model_path)

    # Create environment
    env = create_improved_trading_env()

    actions = []
    rewards = []
    observations = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_actions = []
        episode_rewards = []
        episode_obs = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(float(action[0]))
            episode_obs.append(obs.copy())

            obs, reward, done, _, _ = env.step(action)
            episode_rewards.append(reward)

        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        observations.extend(episode_obs)

        logger.info(f"Episode {episode+1}/{n_episodes}: {len(episode_actions)} steps")

    # Convert to numpy arrays
    actions = np.array(actions)
    rewards = np.array(rewards)
    observations = np.array(observations)

    # Analyze action distribution with improved thresholds
    analysis = {
        "total_actions": len(actions),
        "action_stats": {
            "mean": float(np.mean(actions)),
            "std": float(np.std(actions)),
            "min": float(np.min(actions)),
            "max": float(np.max(actions)),
            "median": float(np.median(actions))
        },
        "action_distribution": {
            "buy_signals": int(np.sum(actions > 0.05)),  # Lower threshold
            "sell_signals": int(np.sum(actions < -0.05)),  # Lower threshold
            "hold_signals": int(np.sum((actions >= -0.05) & (actions <= 0.05))),
            "strong_buy": int(np.sum(actions > 0.5)),
            "strong_sell": int(np.sum(actions < -0.5))
        },
        "reward_stats": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "total": float(np.sum(rewards))
        }
    }

    # Calculate bias ratios
    total_signals = analysis["action_distribution"]["buy_signals"] + analysis["action_distribution"]["sell_signals"]
    if total_signals > 0:
        analysis["bias_analysis"] = {
            "buy_ratio": analysis["action_distribution"]["buy_signals"] / total_signals,
            "sell_ratio": analysis["action_distribution"]["sell_signals"] / total_signals,
            "buy_sell_ratio": analysis["action_distribution"]["buy_signals"] / max(analysis["action_distribution"]["sell_signals"], 1)
        }
    else:
        analysis["bias_analysis"] = {
            "buy_ratio": 0.0,
            "sell_ratio": 0.0,
            "buy_sell_ratio": 0.0
        }

    return analysis, actions, rewards, observations


def main():
    """Main analysis function"""
    model_path = "models/sac_improved_5000step_final.zip"

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return

    # Analyze action distribution
    analysis, actions, rewards, obs = analyze_improved_action_distribution(model_path)

    # Save analysis results
    analysis_path = "analysis/sac_improved_5000step_action_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Analysis saved to {analysis_path}")

    # Print summary
    print("\n" + "="*60)
    print("IMPROVED SAC 5000-STEP ACTION DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Total Actions Analyzed: {analysis['total_actions']}")
    print(f"Mean Action: {analysis['action_stats']['mean']:.4f}")
    print(f"Action Std: {analysis['action_stats']['std']:.4f}")
    print()

    print("Action Distribution (thresholds: ±0.05):")
    print(f"  Buy Signals (>0.05): {analysis['action_distribution']['buy_signals']}")
    print(f"  Sell Signals (<-0.05): {analysis['action_distribution']['sell_signals']}")
    print(f"  Hold Signals (-0.05 to 0.05): {analysis['action_distribution']['hold_signals']}")
    print(f"  Strong Buy (>0.5): {analysis['action_distribution']['strong_buy']}")
    print(f"  Strong Sell (<-0.5): {analysis['action_distribution']['strong_sell']}")
    print()

    print("Bias Analysis:")
    print(f"  Buy Ratio: {analysis['bias_analysis']['buy_ratio']:.3f}")
    print(f"  Sell Ratio: {analysis['bias_analysis']['sell_ratio']:.3f}")
    print(f"  Buy/Sell Ratio: {analysis['bias_analysis']['buy_sell_ratio']:.3f}")
    print()

    print("Reward Stats:")
    print(f"  Mean Reward: {analysis['reward_stats']['mean']:.6f}")
    print(f"  Reward Std: {analysis['reward_stats']['std']:.6f}")
    print(f"  Total Reward: {analysis['reward_stats']['total']:.4f}")
    print("="*60)

    # Issue detection with improved thresholds
    issues = []
    if analysis['bias_analysis']['sell_ratio'] > 0.7:
        issues.append("SELLバイアス検出: SELLシグナルの割合が70%以上")
    if analysis['bias_analysis']['buy_ratio'] < 0.1:
        issues.append("BUY不足: BUYシグナルの割合が10%未満")
    if analysis['action_stats']['std'] < 0.05:
        issues.append("探索不足: アクションの分散が小さい（<0.05）")
    if analysis['reward_stats']['mean'] < -0.001:
        issues.append("低パフォーマンス: 平均報酬が-0.001未満")
    if analysis['action_distribution']['hold_signals'] > analysis['total_actions'] * 0.95:
        issues.append("過度な保守性: HOLDシグナルが95%以上")

    if issues:
        print("\n検出された問題:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n重大な問題は検出されませんでした")

    print("="*60)


if __name__ == "__main__":
    main()