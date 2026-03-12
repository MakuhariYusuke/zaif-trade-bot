import copy
import unittest

from tests.helpers import make_schema_feature_env_config, make_trending_ohlcv_data
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


class TestEnvRandomizationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_trending_ohlcv_data(
            rows=24,
            seed=11,
            start="2023-01-01",
            freq="1min",
            start_price=100.5,
            end_price=100.5,
            noise_scale=0.0,
            volume_low=1000.0,
            volume_high=1000.0,
            include_timestamp=True,
        )

        cls.base_config = make_schema_feature_env_config(
            cls.df,
            continuous_to_discrete_threshold=0.01,
            exchange_profile={
                "name": "base",
                "maker_fee_rate": 0.0,
                "taker_fee_rate": 0.0,
                "slippage_rate": 0.0,
                "latency_ms": 0.0,
            },
            domain_randomization={
                "enabled": True,
                "maker_fee_range": [0.001, 0.002],
                "taker_fee_range": [0.002, 0.003],
                "slippage_range": [0.01, 0.02],
                "latency_range": [100.0, 200.0],
            },
        )

    def test_env_reset_randomizes_profile(self):
        """Test that environment reset randomizes the exchange profile."""
        try:
            env = HeavyTradingEnv(df=self.df, config=self.base_config)
        except Exception as e:
            print(f"Failed to instantiate env: {e}")
            # If we can't instantiate, we can't test integration directly.
            # But we can check if the logic is there by inspecting the class if needed.
            raise e

        # Initial reset
        env.reset()
        profile1 = env.config.exchange_profile

        # Verify profile1 is randomized (not 0.0)
        self.assertNotEqual(profile1.maker_fee_rate, 0.0)
        self.assertTrue(0.001 <= profile1.maker_fee_rate <= 0.002)

        # Second reset
        env.reset()
        profile2 = env.config.exchange_profile

        # Verify profile2 is randomized and likely different from profile1
        self.assertNotEqual(profile2.maker_fee_rate, 0.0)
        self.assertTrue(0.001 <= profile2.maker_fee_rate <= 0.002)

        self.assertNotEqual(profile1.maker_fee_rate, profile2.maker_fee_rate)

    def test_env_randomization_disabled(self):
        """Test that environment keeps base profile when randomization is disabled."""
        config = copy.deepcopy(self.base_config)
        config["domain_randomization"]["enabled"] = False

        env = HeavyTradingEnv(df=self.df, config=config)

        env.reset()
        profile1 = env.config.exchange_profile

        self.assertEqual(profile1.maker_fee_rate, 0.0)

        env.reset()
        profile2 = env.config.exchange_profile

        self.assertEqual(profile2.maker_fee_rate, 0.0)
