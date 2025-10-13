#!/usr/bin/env python3
"""
Backtest Model Script for SAC Trading Models
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_backtest(model_path, data_path, output_path=None):
    """Run backtest for a given model and data"""
    try:
        from stable_baselines3 import SAC
        import gymnasium as gym
        import numpy as np

        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded data: {len(df)} rows")

        # Simple feature engineering - just use OHLCV
        features = ['open', 'high', 'low', 'close', 'volume']
        df_features = df[features].copy()
        df_features = df_features.fillna(method='ffill').fillna(0)
        print(f"Using features: {features}")

        # Load model
        model = SAC.load(model_path)
        print(f"Loaded model: {model_path}")

        # Create simple environment
        class SimpleTradingEnv(gym.Env):
            def __init__(self, data):
                super().__init__()
                self.data = data
                self.current_step = 0
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32)
                self.position = 0
                self.entry_price = 0
                self.total_pnl = 0

            def reset(self, seed=None, options=None):
                self.current_step = 0
                self.position = 0
                self.entry_price = 0
                self.total_pnl = 0
                return self._get_obs(), {}

            def _get_obs(self):
                return self.data.iloc[self.current_step].values.astype(np.float32)

            def step(self, action):
                current_price = self.data.iloc[self.current_step]['close']
                reward = 0

                # Simple trading logic
                if action[0] > 0.1 and self.position == 0:  # Buy
                    self.position = 1
                    self.entry_price = current_price
                elif action[0] < -0.1 and self.position == 1:  # Sell
                    pnl = current_price - self.entry_price
                    self.total_pnl += pnl
                    reward = pnl
                    self.position = 0
                    self.entry_price = 0

                self.current_step += 1
                done = self.current_step >= len(self.data) - 1

                return self._get_obs(), reward, done, False, {
                    'position': self.position,
                    'entry_price': self.entry_price,
                    'pnl': self.total_pnl
                }

        env = SimpleTradingEnv(df_features)

        # Run backtest
        obs, info = env.reset()
        done = False
        total_reward = 0
        trades = []
        step = 0

        while not done and step < len(df_features) - 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Record trade if position changed
            if info.get('position') != 0 or (step > 0 and info.get('pnl', 0) != 0):
                trade_info = {
                    'step': step,
                    'action': action.tolist() if hasattr(action, 'tolist') else action,
                    'reward': float(reward),
                    'position': float(info.get('position', 0)),
                    'entry_price': float(info.get('entry_price', 0)),
                    'pnl': float(info.get('pnl', 0))
                }
                trades.append(trade_info)

        # Calculate metrics
        if trades:
            pnl_values = [t['pnl'] for t in trades if t['pnl'] != 0]
            total_return = sum(pnl_values) if pnl_values else 0
            win_rate = sum(1 for pnl in pnl_values if pnl > 0) / len(pnl_values) * 100 if pnl_values else 0
            max_drawdown = min(pnl_values) if pnl_values else 0

            # Calculate Sharpe ratio (simplified)
            returns = np.array(pnl_values)
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            total_return = 0
            win_rate = 0
            max_drawdown = 0
            sharpe_ratio = 0

        results = {
            'metrics': {
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown),
                'total_trades': len(trades)
            },
            'trades': trades,
            'total_steps': step,
            'model_path': model_path,
            'data_path': data_path
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")

        return results

    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backtest SAC trading model')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--data', required=True, help='Path to data CSV file')
    parser.add_argument('--output', help='Output JSON file path')

    args = parser.parse_args()

    results = run_backtest(args.model, args.data, args.output)

    if results:
        print("\n=== BACKTEST RESULTS ===")
        metrics = results['metrics']
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
    else:
        sys.exit(1)