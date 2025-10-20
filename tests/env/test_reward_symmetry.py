"""
Test reward symmetry using flipped environment and PnL-only rewards.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment import FlipHeavyTradingEnv, HeavyTradingEnv


class TestRewardSymmetry:
    """Test reward function symmetry and causes of action bias."""

    def test_flip_env_symmetry(self):
        """Test that flipped environment produces symmetric action distributions."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
        data = {
            "open": prices,
            "high": prices + np.random.rand(1000) * 2,
            "low": prices - np.random.rand(1000) * 2,
            "close": prices + np.random.randn(1000) * 0.05,
            "volume": np.random.randint(1000, 10000, 1000),
        }
        df = pd.DataFrame(data, index=dates)

        config = {
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
            "reward_scaling": 6.0,  # Optimal value found
        }

        # Train on normal environment
        env_normal = DummyVecEnv([lambda: HeavyTradingEnv(df=df, config=config)])
        model_normal = PPO("MlpPolicy", env_normal, verbose=0, seed=42)
        model_normal.learn(total_timesteps=10000)

        # Collect actions from normal env
        actions_normal = []
        obs = env_normal.reset()
        for _ in range(1000):
            action, _ = model_normal.predict(obs[0], deterministic=False)
            actions_normal.append(int(action))
            obs, _, done, _ = env_normal.step(action)
            if done:
                break

        # Train on flipped environment
        env_flipped = DummyVecEnv([lambda: FlipHeavyTradingEnv(df=df, config=config)])
        model_flipped = PPO("MlpPolicy", env_flipped, verbose=0, seed=42)
        model_flipped.learn(total_timesteps=10000)

        # Collect actions from flipped env
        actions_flipped = []
        obs = env_flipped.reset()
        for _ in range(1000):
            action, _ = model_flipped.predict(obs[0], deterministic=False)
            actions_flipped.append(int(action))
            obs, _, done, _ = env_flipped.step(action)
            if done:
                break

        # Analyze action distributions
        def analyze_actions(actions: list[int]) -> dict[str, float]:
            total = len(actions)
            hold = sum(1 for a in actions if a == 0)
            buy = sum(1 for a in actions if a == 1)
            sell = sum(1 for a in actions if a == 2)
            return {
                "hold_pct": hold / total * 100,
                "buy_pct": buy / total * 100,
                "sell_pct": sell / total * 100,
            }

        dist_normal = analyze_actions(actions_normal)
        dist_flipped = analyze_actions(actions_flipped)

        print(f"Normal env: {dist_normal}")
        print(f"Flipped env: {dist_flipped}")

        # Check if distributions are similar (within 10% points)
        # If reward is symmetric, flipped env should have similar distribution
        assert (
            abs(dist_normal["buy_pct"] - dist_flipped["sell_pct"]) < 10
        ), "BUY/SELL asymmetry detected"
        assert (
            abs(dist_normal["sell_pct"] - dist_flipped["buy_pct"]) < 10
        ), "SELL/BUY asymmetry detected"

    def test_pnl_only_reward(self):
        """Test action distribution with PnL-only reward (no penalties)."""
        # Create test data
        dates = pd.date_range("2023-01-01", periods=1000, freq="1min")
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
        data = {
            "open": prices,
            "high": prices + np.random.rand(1000) * 2,
            "low": prices - np.random.rand(1000) * 2,
            "close": prices + np.random.randn(1000) * 0.05,
            "volume": np.random.randint(1000, 10000, 1000),
        }
        df = pd.DataFrame(data, index=dates)

        # Custom environment with PnL-only reward
        class PnLOnlyEnv(HeavyTradingEnv):
            def _calculate_reward(self, *args, **kwargs) -> float:
                pnl = kwargs.get("pnl", 0.0)
                return float(pnl)  # Just return PnL

        config = {
            "transaction_cost": 0.001,
            "max_position_size": 1.0,
        }

        env = DummyVecEnv([lambda: PnLOnlyEnv(df=df, config=config)])
        model = PPO("MlpPolicy", env, verbose=0, seed=42)
        model.learn(total_timesteps=10000)

        # Collect actions
        actions = []
        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs[0], deterministic=False)
            actions.append(int(action))
            obs, _, done, _ = env.step(action)
            if done:
                break

        # Analyze distribution
        total = len(actions)
        hold = sum(1 for a in actions if a == 0) / total * 100
        buy = sum(1 for a in actions if a == 1) / total * 100
        sell = sum(1 for a in actions if a == 2) / total * 100

        print(f"PnL-only: HOLD={hold:.1f}%, BUY={buy:.1f}%, SELL={sell:.1f}%")

        # With pure PnL reward, should be more balanced than current biased reward
        # This tests if the bias comes from reward design
        assert buy > 5, "BUY actions should be present with PnL-only reward"
        assert sell > 5, "SELL actions should be present with PnL-only reward"
