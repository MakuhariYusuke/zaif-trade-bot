"""
Unit tests for TrendDetector (SAC v448 Layer 1).

Tests:
- Basic trend detection (uptrend, downtrend, neutral)
- Signal normalization
- Edge cases (insufficient data, flat market)
- Statistics tracking

Version: 1.0
Created: 2025-11-22
"""

import pytest
import numpy as np
from ztb.trading.environment.components.reward.trend_detector import TrendDetector


class TestTrendDetectorBasics:
    """Basic functionality tests."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = TrendDetector(lookback=20)
        
        assert detector.lookback == 20
        assert detector.min_samples == 20
        assert len(detector.price_history) == 0
        assert detector.update_count == 0
        assert detector.last_signal == 0.0
    
    def test_initialization_custom_min_samples(self):
        """Test with custom min_samples."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        assert detector.lookback == 20
        assert detector.min_samples == 10
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Add only 5 prices
        for i in range(5):
            detector.update(1000 + i)
        
        signal = detector.get_trend_signal()
        assert signal == 0.0  # Not enough data


class TestTrendDetection:
    """Trend detection accuracy tests."""
    
    def test_uptrend_detection(self):
        """Test uptrend detection."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Strong uptrend: 1000 -> 1200 over 20 steps
        for i in range(20):
            price = 1000 + i * 10
            detector.update(price)
        
        signal = detector.get_trend_signal()
        assert signal > 0.3, f"Expected uptrend signal > 0.3, got {signal}"
        assert detector.get_trend_strength() in ["Uptrend", "Strong Uptrend"]
    
    def test_downtrend_detection(self):
        """Test downtrend detection."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Strong downtrend: 1200 -> 1000 over 20 steps
        for i in range(20):
            price = 1200 - i * 10
            detector.update(price)
        
        signal = detector.get_trend_signal()
        assert signal < -0.3, f"Expected downtrend signal < -0.3, got {signal}"
        assert detector.get_trend_strength() in ["Downtrend", "Strong Downtrend"]
    
    def test_neutral_flat_market(self):
        """Test neutral detection in flat market."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Flat market: constant price
        for i in range(20):
            detector.update(1000)
        
        signal = detector.get_trend_signal()
        assert signal == 0.0, f"Expected neutral signal 0.0, got {signal}"
        assert detector.get_trend_strength() == "Neutral"
    
    def test_neutral_sideways(self):
        """Test neutral detection in sideways market."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Sideways: oscillating around 1000
        for i in range(20):
            price = 1000 + (10 if i % 2 == 0 else -10)
            detector.update(price)
        
        signal = detector.get_trend_signal()
        assert -0.3 <= signal <= 0.3, f"Expected neutral signal, got {signal}"


class TestSignalNormalization:
    """Test signal normalization to [-1, 1]."""
    
    def test_signal_clipping(self):
        """Test that signal is always in [-1, 1]."""
        detector = TrendDetector(lookback=10, min_samples=5)
        
        # Extreme uptrend
        for i in range(10):
            detector.update(1000 + i * 100)
        
        signal = detector.get_trend_signal()
        assert -1.0 <= signal <= 1.0, f"Signal {signal} out of bounds"
    
    def test_multiple_scenarios(self):
        """Test signal bounds across multiple scenarios."""
        test_cases = [
            # (price_sequence, description)
            ([1000 + i * 50 for i in range(20)], "steep uptrend"),
            ([1000 - i * 50 for i in range(20)], "steep downtrend"),
            ([1000] * 20, "flat"),
            ([1000 + (i % 5) * 10 for i in range(20)], "noisy"),
        ]
        
        for prices, desc in test_cases:
            detector = TrendDetector(lookback=20, min_samples=10)
            for price in prices:
                detector.update(price)
            
            signal = detector.get_trend_signal()
            assert -1.0 <= signal <= 1.0, f"{desc}: signal {signal} out of bounds"


class TestEdgeCases:
    """Edge case handling tests."""
    
    def test_invalid_prices(self):
        """Test handling of invalid prices."""
        detector = TrendDetector(lookback=10)
        
        # Valid prices
        detector.update(1000)
        detector.update(1010)
        
        # Invalid prices (should be skipped)
        detector.update(np.inf)
        detector.update(np.nan)
        detector.update(-100)
        detector.update(0)
        
        # Should have only 2 valid prices
        assert len(detector.price_history) == 2
    
    def test_single_price(self):
        """Test with single price."""
        detector = TrendDetector(lookback=20, min_samples=1)
        detector.update(1000)
        
        signal = detector.get_trend_signal()
        assert signal == 0.0  # Can't determine trend from single point
    
    def test_two_prices_uptrend(self):
        """Test minimal uptrend (2 prices)."""
        detector = TrendDetector(lookback=20, min_samples=2)
        detector.update(1000)
        detector.update(1100)
        
        signal = detector.get_trend_signal()
        # Should detect uptrend even with minimal data
        assert signal > 0
    
    def test_two_prices_downtrend(self):
        """Test minimal downtrend (2 prices)."""
        detector = TrendDetector(lookback=20, min_samples=2)
        detector.update(1100)
        detector.update(1000)
        
        signal = detector.get_trend_signal()
        # Should detect downtrend
        assert signal < 0


class TestStatistics:
    """Statistics and utility methods tests."""
    
    def test_reset(self):
        """Test reset functionality."""
        detector = TrendDetector(lookback=10)
        
        # Add some data
        for i in range(10):
            detector.update(1000 + i)
        
        assert len(detector.price_history) > 0
        assert detector.update_count > 0
        
        # Reset
        detector.reset()
        
        assert len(detector.price_history) == 0
        assert detector.update_count == 0
        assert detector.last_signal == 0.0
    
    def test_statistics(self):
        """Test get_statistics()."""
        detector = TrendDetector(lookback=10, min_samples=5)
        
        # Add uptrend data
        for i in range(10):
            detector.update(1000 + i * 10)
        
        # Call get_trend_signal to update last_signal
        signal = detector.get_trend_signal()
        
        stats = detector.get_statistics()
        
        assert stats["samples"] == 10
        assert stats["update_count"] == 10
        assert stats["last_signal"] > 0  # Uptrend
        assert stats["trend_strength"] in ["Uptrend", "Strong Uptrend"]
        assert stats["price_range"] == 90  # 1090 - 1000
        assert stats["current_price"] == 1090
    
    def test_repr(self):
        """Test __repr__ method."""
        detector = TrendDetector(lookback=20)
        
        # Add some data
        for i in range(10):
            detector.update(1000 + i)
        
        repr_str = repr(detector)
        
        assert "TrendDetector" in repr_str
        assert "lookback=20" in repr_str
        assert "samples=" in repr_str


class TestRealWorldScenarios:
    """Realistic market scenario tests."""
    
    def test_crypto_volatile_uptrend(self):
        """Test detection in volatile crypto uptrend."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Volatile uptrend: overall up but with noise
        base_price = 1000
        for i in range(20):
            noise = np.random.randint(-20, 20)
            trend = i * 5
            price = base_price + trend + noise
            detector.update(price)
        
        signal = detector.get_trend_signal()
        # Should still detect uptrend despite noise
        assert signal > 0, f"Expected uptrend, got {signal}"
    
    def test_trend_reversal(self):
        """Test detection of trend reversal."""
        detector = TrendDetector(lookback=10, min_samples=5)
        
        # First: uptrend
        for i in range(10):
            detector.update(1000 + i * 10)
        
        signal1 = detector.get_trend_signal()
        assert signal1 > 0, "Expected uptrend initially"
        
        # Then: downtrend (will gradually shift as old data leaves window)
        for i in range(10):
            detector.update(1100 - i * 10)
        
        signal2 = detector.get_trend_signal()
        # Signal should have changed (either neutral or downtrend)
        assert signal2 < signal1, "Expected signal to decrease after reversal"


class TestIntegrationScenarios:
    """Integration with reward calculation scenarios."""
    
    def test_trend_aware_balance_adjustment(self):
        """Test typical usage for trend-aware balance adjustments."""
        detector = TrendDetector(lookback=20, min_samples=10)
        
        # Simulate strong uptrend market
        for i in range(20):
            detector.update(1000 + i * 15)
        
        signal = detector.get_trend_signal()
        
        # In uptrend, should favor BUY slightly
        if signal > 0.3:
            buy_adjustment = signal * 0.05  # Max 5% adjustment
            assert 0 < buy_adjustment <= 0.05
        
        # Verify strength classification
        strength = detector.get_trend_strength()
        assert strength in ["Uptrend", "Strong Uptrend"]
    
    def test_lookback_window_effect(self):
        """Test effect of different lookback windows."""
        short_detector = TrendDetector(lookback=5, min_samples=3)
        long_detector = TrendDetector(lookback=20, min_samples=10)
        
        # Add same data to both
        prices = [1000 + i * 5 for i in range(20)]
        for price in prices:
            short_detector.update(price)
            long_detector.update(price)
        
        short_signal = short_detector.get_trend_signal()
        long_signal = long_detector.get_trend_signal()
        
        # Both should detect uptrend
        assert short_signal > 0
        assert long_signal > 0
        
        # Signals may differ due to different windows
        # (short_detector only sees last 5 prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
