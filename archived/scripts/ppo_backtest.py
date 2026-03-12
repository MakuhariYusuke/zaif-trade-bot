#!/usr/bin/env python3
"""
Backtest PPO Model Script
"""

import json
import os
import sys

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_backtest(model_path, data_path, output_path=None):
    """Run backtest for a given PPO model and data"""
    try:
        import gymnasium as gym
        import numpy as np
        from stable_baselines3 import PPO

        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded data: {len(df)} rows")

        # Simple feature engineering - just use OHLCV
        features = ["open", "high", "low", "close", "volume"]
        df_features = df[features].copy()
        df_features = df_features.fillna(method="ffill").fillna(0)
        print(f"Using features: {features}")

        # Load model
        model = PPO.load(model_path)
        print(f"Loaded model: {model_path}")

        # Create simple environment
        class SimpleTradingEnv(gym.Env):
            def __init__(self, data):
                super().__init__()
                self.data = data
                self.current_step = 0
                self.action_space = gym.spaces.Discrete(3)  # HOLD, BUY, SELL
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32
                )
                self.position = 0
                self.entry_price = 0
                self.total_pnl = 0
                self.portfolio_value = 100000  # Initial balance

            def reset(self, seed=None, options=None):
                self.current_step = 0
                self.position = 0
                self.entry_price = 0
                self.total_pnl = 0
                self.portfolio_value = 100000
                return self._get_obs(), {}

            def _get_obs(self):
                return self.data.iloc[self.current_step][features].values.astype(
                    np.float32
                )

            def step(self, action):
                # Execute action
                current_price = self.data.iloc[self.current_step]["close"]

                if action == 1:  # BUY
                    if self.position <= 0:  # Can buy
                        self.position = 1
                        self.entry_price = current_price
                elif action == 2:  # SELL
                    if self.position >= 0:  # Can sell
                        self.position = -1
                        self.entry_price = current_price

                # Calculate PnL
                pnl = 0
                if self.position != 0:
                    pnl = self.position * (current_price - self.entry_price)

                self.total_pnl = pnl
                self.portfolio_value = 100000 + pnl

                # Move to next step
                self.current_step += 1
                done = self.current_step >= len(self.data) - 1

                reward = pnl * 0.01  # Simple reward

                return self._get_obs(), reward, done, False, {}

        env = SimpleTradingEnv(df_features)

        # Run backtest
        obs = env.reset()
        total_reward = 0
        actions_taken = []
        portfolio_values = []

        for step in range(len(df_features) - 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            total_reward += reward
            actions_taken.append(action)
            portfolio_values.append(env.portfolio_value)

            if done:
                break

        # Calculate statistics
        final_pnl = env.total_pnl
        total_return = (env.portfolio_value - 100000) / 100000 * 100

        results = {
            "model": str(model_path),
            "data_points": len(df_features),
            "final_pnl": final_pnl,
            "total_return_percent": total_return,
            "total_reward": total_reward,
            "action_distribution": {
                "HOLD": actions_taken.count(0),
                "BUY": actions_taken.count(1),
                "SELL": actions_taken.count(2),
            },
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")

        print("Backtest Results:")
        print(f"Final PnL: ¥{final_pnl:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Action Distribution: {results['action_distribution']}")

        return results

    except Exception as e:
        print(f"Error in backtest: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest PPO model")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument(
        "--data", default="btc_jpy_real_dataset.csv", help="Path to data file"
    )
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    run_backtest(args.model, args.data, args.output)
