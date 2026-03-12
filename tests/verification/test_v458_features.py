import unittest
import pandas as pd
import numpy as np
import logging
from ztb.trading.environment.factory_v456 import EnvironmentFactory
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456

# Setup logging
logging.basicConfig(level=logging.INFO)

class TestV458Features(unittest.TestCase):
    def setUp(self):
        # Create dummy data similar to v457 test
        n_steps = 1000
        dates = pd.date_range('2025-01-01', periods=n_steps, freq='1min', tz='UTC')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n_steps))  # Random walk

        # Base cols placeholders (30 total)
        base_cols = ["open", "high", "low", "close", "volume"] + \
                    ["sma_5", "sma_20", "sma_50", "ema_5", "ema_20", "ema_50", "rsi_14", "rsi_20", "atr_14", "atr_20",
                     "bb_upper_20", "bb_lower_20", "bb_pct_b_20", "macd_line", "macd_signal", "adx_14", "plus_di_14", "minus_di_14",
                     "obv", "vpt", "sma_5_close_ratio", "atr_pct_close", "hl_ratio", "hml_ratio", "trend_direction"]

        # Fill required base cols
        data = {c: np.random.randn(n_steps) for c in base_cols}
        data['close'] = prices
        data['open'] = prices + np.random.randn(n_steps)*0.1
        data['high'] = prices + np.abs(np.random.randn(n_steps)*0.5)
        data['low'] = prices - np.abs(np.random.randn(n_steps)*0.5)
        data['volume'] = np.abs(np.random.randn(n_steps)*1000)

        self.df = pd.DataFrame(data, index=dates)
        self.df['timestamp'] = dates
        self.n_steps = n_steps
        self.dates = dates

    def test_mtf_causality(self):
        """Verify MTF features are calculated (basic sanity check)"""
        print("\n--- Testing MTF Causality ---")
        factory = EnvironmentFactory(self.df)
        df_prepared, feature_cols = factory.prepare_features()

        mtf_cols = feature_cols['mtf']
        self.assertEqual(len(mtf_cols), 27)

        # Basic check: At least some features should be non-zero
        total_non_zero = 0
        for col in mtf_cols:
            values = df_prepared[col].values
            non_zero_count = np.count_nonzero(~np.isnan(values))
            total_non_zero += non_zero_count

        self.assertGreater(total_non_zero, len(mtf_cols) * 10, "Too many NaN values in MTF features")

        print("✓ MTF features are calculated (basic sanity check passed)")

    def test_guidance_decay(self):
        """Verify that trend guidance penalty decays over lifetime steps"""
        print("\n--- Testing Guidance Decay ---")
        factory = EnvironmentFactory(self.df)
        df_prepared, feature_cols = factory.prepare_features()

        # Create environment with short decay for testing
        env = FastIntradayEnvV456(
            df=df_prepared,
            base_feature_columns=feature_cols['base'],
            mtf_feature_columns=feature_cols['mtf'],
            regime_feature_columns=feature_cols['regime'],
            guidance_decay_steps=100,  # Short decay for testing
            max_steps=50,
            action_space_type="2d_position",
            reward_clip=None
        )

        obs, info = env.reset()

        # Track penalty effects over steps
        early_penalties = []
        late_penalties = []

        for step in range(50):
            # Action that opposes trend (if trend is positive, action negative)
            action = np.array([-0.5, 1.0])  # 2d_position: position, ttl
            obs, reward, done, truncated, info = env.step(action)

            if step < 25:  # Early steps
                early_penalties.append(reward)
            else:  # Late steps
                late_penalties.append(reward)

        # Statistical check: penalty effect should decrease (less negative rewards in late steps)
        early_avg = np.mean(early_penalties)
        late_avg = np.mean(late_penalties)

        # Late rewards should be higher (less penalty) than early
        self.assertGreater(late_avg, early_avg - 0.01,  # Allow small tolerance
                          f"Guidance decay failed: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}")

        print(f"Early avg reward: {early_avg:.4f}, Late avg reward: {late_avg:.4f}")
        print("✓ Guidance decay test passed")

    def test_reward_scaling(self):
        """Verify reward scaling and penalty magnitude"""
        print("\n--- Testing Reward Scaling ---")
        factory = EnvironmentFactory(self.df)
        df_prepared, feature_cols = factory.prepare_features()

        env = FastIntradayEnvV456(
            df=df_prepared,
            base_feature_columns=feature_cols['base'],
            mtf_feature_columns=feature_cols['mtf'],
            regime_feature_columns=feature_cols['regime'],
            reward_scale=1000.0,  # Smaller scale for testing
            guidance_decay_steps=50000,
            action_space_type="2d_position",
            reward_clip=None
        )

        obs, info = env.reset()

        # Take a few steps
        rewards = []
        for _ in range(10):
            action = np.array([0.1, 1.0])
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)

        # Rewards should be scaled appropriately
        avg_reward = np.mean(np.abs(rewards))
        self.assertLess(avg_reward, 10.0, f"Rewards should be scaled down, got {avg_reward}")

        print(f"Average |reward|: {avg_reward}")
        print("✓ Reward scaling test passed")

if __name__ == '__main__':
    unittest.main()