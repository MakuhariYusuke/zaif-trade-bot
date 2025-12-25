import unittest


from ztb.trading.signal.entry_system import IntegratedEntrySystem
from ztb.trading.types import MarketState


class TestIntegratedEntrySystem(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ewma_tau": 100.0,
            "n_min": 30.0,
            "fee_rate": 0.001,
            "c_spread": 0.3,
            "c_vol": 0.2,
            "c_imp": 0.5,
            "gamma": 0.5,
            "min_volume": 0.01,
            "latency_sec": 1.0,
            "order_size_btc": 0.01,
        }
        self.system = IntegratedEntrySystem(self.config)
        self.market_data: MarketState = {
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "atr": 1.0,
            "volume": 1000.0,
            "timestamp": "2023-01-01",
        }

    def test_process_signal_buy(self):
        # Test Buy Signal
        result = self.system.process_signal(
            0.8, self.market_data, "bull", order_size=0.01
        )
        self.assertIn("should_enter", result)
        self.assertIn("ev", result)
        self.assertIn("cost", result)

    def test_process_signal_sell(self):
        # Test Sell Signal (Short)
        result = self.system.process_signal(
            -0.8, self.market_data, "bear", order_size=0.01
        )
        self.assertIn("should_enter", result)

    def test_update_outcome(self):
        # Test updating stats
        self.system.update_outcome("bull", 0.8, 5.0, 100)

        # Verify stats updated
        stats = self.system.calibration_map.get_stats("bull", 0.8)
        self.assertGreater(stats["l1"]["avg_win"], 0.0)

    def test_save_load_state(self):
        import os
        import tempfile

        # Update some stats
        self.system.update_outcome("bull", 0.8, 10.0, 100)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name

        try:
            self.system.save_state(path)

            new_system = IntegratedEntrySystem(self.config)
            new_system.load_state(path)

            stats = new_system.calibration_map.get_stats("bull", 0.8)
            self.assertGreater(stats["l1"]["avg_win"], 0.0)

        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
