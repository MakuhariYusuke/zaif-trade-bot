import os

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC


class SimpleTradingEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.current_step = 0
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )  # Continuous action
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )  # OHLCV
        self.position = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.portfolio_value = 100000  # Initial balance
        self.initial_balance = self.portfolio_value

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.portfolio_value = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        return self.data.iloc[self.current_step][
            ["open", "high", "low", "close", "volume"]
        ].values.astype(np.float32)

    def step(self, action):
        # Action is continuous between -1 and 1
        # Convert to discrete: -1 to -0.1 = SELL, -0.1 to 0.1 = HOLD, 0.1 to 1 = BUY
        if action < -0.1:
            action_discrete = -1  # SELL
        elif action > 0.1:
            action_discrete = 1  # BUY
        else:
            action_discrete = 0  # HOLD

        current_price = self.data.iloc[self.current_step]["close"]

        # Execute trade
        if action_discrete == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = current_price
        elif action_discrete == -1 and self.position == 1:  # SELL
            pnl = (current_price - self.entry_price) * 1000  # Assume 1000 units
            self.total_pnl += pnl
            self.portfolio_value += pnl
            self.position = 0
            self.entry_price = 0

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Reward is portfolio value change
        reward = 0
        if done and self.position == 1:  # Close position at end
            pnl = (current_price - self.entry_price) * 1000
            self.total_pnl += pnl
            self.portfolio_value += pnl
            reward = pnl

        return self._get_obs(), reward, done, False, {}


def load_sac_model(zip_path):
    """Load SAC model from zip file"""
    # SAC.load can load directly from zip
    model = SAC.load(zip_path)
    return model


def run_backtest(model_path, data_path="btc_jpy_real_dataset.csv"):
    """Run backtest on SAC model"""
    # Load data
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Create environment
    env = SimpleTradingEnv(df)

    # Load model
    model = load_sac_model(model_path)

    # Run backtest
    obs, _ = env.reset()
    done = False
    total_reward = 0
    trades = 0
    buy_actions = 0
    sell_actions = 0
    hold_actions = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Count actions
        if action > 0.1:  # Buy
            buy_actions += 1
            trades += 1
        elif action < -0.1:  # Sell
            sell_actions += 1
            trades += 1
        else:  # Hold
            hold_actions += 1

    # Calculate final portfolio value
    final_value = env.portfolio_value
    initial_value = env.initial_balance
    return_pct = ((final_value - initial_value) / initial_value) * 100

    return {
        "model": os.path.basename(model_path),
        "return_pct": return_pct,
        "total_reward": total_reward,
        "trades": trades,
        "buy_actions": buy_actions,
        "sell_actions": sell_actions,
        "hold_actions": hold_actions,
        "final_value": final_value,
        "initial_value": initial_value,
    }


if __name__ == "__main__":
    # Test different V396 models
    v396_models = [
        "checkpoints/sac_session/sac_v396_50k_final.zip",
        "checkpoints/sac_session/sac_v396_optimized_final.zip",
        "checkpoints/sac_session/sac_v396_retrained_final.zip",
    ]

    results = []
    for model_path in v396_models:
        if os.path.exists(model_path):
            try:
                result = run_backtest(model_path)
                results.append(result)
                print(f"Model: {result['model']}")
                print(".2f")
                print(f"Trades: {result['trades']}")
                print(
                    f"Buy/Sell/Hold: {result['buy_actions']}/{result['sell_actions']}/{result['hold_actions']}"
                )
                print("---")
            except Exception as e:
                print(f"Error with {model_path}: {e}")
        else:
            print(f"Model not found: {model_path}")

    # Print summary
    if results:
        print("\nV396 Models Summary:")
        for r in results:
            print(".2f")
