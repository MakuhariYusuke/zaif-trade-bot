import unittest
from types import SimpleNamespace


from ztb.trading.environment.components.threshold_manager import ThresholdManager


class TestThresholdManager(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            continuous_to_discrete_threshold=0.01,
            adaptive_threshold_mode=False,
            threshold_volatility_multiplier=1.0,
            min_action_threshold=0.001,
            max_action_threshold=0.05,
            dynamic_threshold_mode="fixed",
            z_score_window=100,
            z_score_threshold=2.0,
            z_score_method="std",
            regime_detection_window=50,
            threshold_adaptation_rate=0.1,
            performance_memory_size=100,
            regime_detection_config={},
            trend_detection_threshold=0.001,
            volatility_detection_threshold=0.02,
        )

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
