import unittest
from unittest.mock import MagicMock


from ztb.trading.environment.components.threshold_manager import ThresholdManager


class TestThresholdManager(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.continuous_to_discrete_threshold = 0.01
        self.config.adaptive_threshold_mode = False
        self.config.threshold_volatility_multiplier = 1.0
        self.config.min_action_threshold = 0.001
        self.config.max_action_threshold = 0.05

    def test_static_threshold(self):
        manager = ThresholdManager(self.config)
        threshold = manager.get_threshold(volatility=100.0, current_price=1000.0)
        self.assertEqual(threshold, 0.01)

    def test_adaptive_threshold(self):
        self.config.adaptive_threshold_mode = True
        self.config.threshold_volatility_multiplier = 0.5
        manager = ThresholdManager(self.config)

        # Case 1: Low volatility
        # Price = 1000, ATR = 10 => Rel Vol = 0.01
        # Threshold = 0.01 + (0.01 * 0.5) = 0.015
        threshold = manager.get_threshold(volatility=10.0, current_price=1000.0)
        self.assertAlmostEqual(threshold, 0.015)

        # Case 2: High volatility
        # Price = 1000, ATR = 50 => Rel Vol = 0.05
        # Threshold = 0.01 + (0.05 * 0.5) = 0.035
        threshold = manager.get_threshold(volatility=50.0, current_price=1000.0)
        self.assertAlmostEqual(threshold, 0.035)

    def test_bounds(self):
        self.config.adaptive_threshold_mode = True
        self.config.threshold_volatility_multiplier = 10.0  # High multiplier
        manager = ThresholdManager(self.config)

        # Should cap at max_threshold (0.05)
        # Price = 1000, ATR = 100 => Rel Vol = 0.1
        # Calc = 0.01 + (0.1 * 10.0) = 1.01 -> clipped to 0.05
        threshold = manager.get_threshold(volatility=100.0, current_price=1000.0)
        self.assertEqual(threshold, 0.05)

        # Should floor at min_threshold (0.001)
        # Even if calc is lower (not possible with addition logic unless negative volatility?)
        # But let's test if base is very small
        self.config.continuous_to_discrete_threshold = 0.0001
        manager = ThresholdManager(self.config)
        threshold = manager.get_threshold(volatility=0.0, current_price=1000.0)
        self.assertEqual(threshold, 0.001)  # Min threshold

    def test_missing_data(self):
        self.config.adaptive_threshold_mode = True
        manager = ThresholdManager(self.config)

        # Missing volatility
        threshold = manager.get_threshold(volatility=None, current_price=1000.0)
        self.assertEqual(threshold, 0.01)

        # Missing price
        threshold = manager.get_threshold(volatility=10.0, current_price=None)
        self.assertEqual(threshold, 0.01)


if __name__ == "__main__":
    unittest.main()
