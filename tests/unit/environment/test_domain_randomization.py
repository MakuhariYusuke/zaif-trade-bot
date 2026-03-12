import unittest

from ztb.trading.environment.utils.domain_randomizer import (
    DomainRandomizationConfig,
    DomainRandomizer,
)
from ztb.trading.environment.utils.exchange_profile import ExchangeProfile
from ztb.utils.fee_model import FixedFeeModel


class TestDomainRandomization(unittest.TestCase):
    def test_randomization_logic(self):
        """Test that randomization produces different values within range."""
        config = DomainRandomizationConfig(
            enabled=True,
            maker_fee_range=(0.001, 0.002),
            taker_fee_range=(0.002, 0.003),
            slippage_range=(0.01, 0.02),
            latency_range=(100.0, 200.0),
        )
        randomizer = DomainRandomizer(config)

        base_profile = ExchangeProfile(
            name="base",
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_rate=0.0,
            latency_ms=0.0,
        )

        # Run multiple times to ensure randomness and range compliance
        for _ in range(10):
            randomized = randomizer.randomize_profile(base_profile)

            self.assertNotEqual(randomized.maker_fee_rate, 0.0)
            self.assertTrue(0.001 <= randomized.maker_fee_rate <= 0.002)

            self.assertNotEqual(randomized.taker_fee_rate, 0.0)
            self.assertTrue(0.002 <= randomized.taker_fee_rate <= 0.003)

            self.assertNotEqual(randomized.slippage_rate, 0.0)
            self.assertTrue(0.01 <= randomized.slippage_rate <= 0.02)

            self.assertNotEqual(randomized.latency_ms, 0.0)
            self.assertTrue(100.0 <= randomized.latency_ms <= 200.0)

            # Check fee model update
            self.assertIsInstance(randomized.fee_model, FixedFeeModel)
            # FixedFeeModel uses buy/sell, we map taker to both for simplicity in randomizer currently
            self.assertEqual(
                randomized.fee_model.buy_fee_rate, randomized.taker_fee_rate
            )

    def test_disabled_randomization(self):
        """Test that disabled randomization returns original profile."""
        config = DomainRandomizationConfig(enabled=False)
        randomizer = DomainRandomizer(config)

        base_profile = ExchangeProfile(name="base")
        randomized = randomizer.randomize_profile(base_profile)

        self.assertEqual(randomized, base_profile)


if __name__ == "__main__":
    unittest.main()
