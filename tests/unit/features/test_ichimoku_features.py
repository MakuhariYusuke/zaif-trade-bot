"""
Unit tests for Ichimoku features
"""

import numpy as np
import pandas as pd
import pytest

from ztb.features.trend.ichimoku_cloud_expansion import compute_ichimoku_cloud_expansion
from ztb.features.trend.ichimoku_cloud_slope import compute_ichimoku_cloud_slope
from ztb.features.trend.ichimoku_ext import (
    calculate_ichimoku_extended,
    compute_ichimoku_chikou,
    compute_ichimoku_cloud_thickness,
    compute_ichimoku_composite_signal,
    compute_ichimoku_kijun,
    compute_ichimoku_price_cloud_distance,
    compute_ichimoku_senkou_a,
    compute_ichimoku_senkou_b,
    compute_ichimoku_tenkan,
    compute_ichimoku_trend,
)
from ztb.features.trend.ichimoku_momentum_confirmation import (
    compute_ichimoku_momentum_confirmation,
)
from ztb.features.trend.ichimoku_sanyaku_kouten import compute_ichimoku_sanyaku_kouten
from ztb.features.trend.ichimoku_time_theory import compute_ichimoku_time_theory
from ztb.features.trend.ichimoku_value_measurement import (
    compute_ichimoku_value_measurement,
)
from ztb.features.trend.ichimoku_wave_theory import compute_ichimoku_wave_theory


class TestIchimokuFeatures:
    """Test suite for Ichimoku Kinko Hyo features"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data for testing"""
        np.random.seed(42)
        n = 100

        # Create realistic OHLC data
        close = np.random.uniform(100, 200, n)
        high = close + np.random.uniform(0, 10, n)
        low = close - np.random.uniform(0, 10, n)
        volume = np.random.uniform(1000, 5000, n)

        df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})

        return df

    def test_ichimoku_basic_lines(self, sample_data):
        """Test basic Ichimoku lines computation"""
        tenkan = compute_ichimoku_tenkan(sample_data)
        kijun = compute_ichimoku_kijun(sample_data)
        senkou_a = compute_ichimoku_senkou_a(sample_data)
        senkou_b = compute_ichimoku_senkou_b(sample_data)
        chikou = compute_ichimoku_chikou(sample_data)

        # Check that all series have the correct length
        assert len(tenkan) == len(sample_data)
        assert len(kijun) == len(sample_data)
        assert len(senkou_a) == len(sample_data)
        assert len(senkou_b) == len(sample_data)
        assert len(chikou) == len(sample_data)

        # Check that values are reasonable (not all NaN)
        assert not tenkan.isna().all()
        assert not kijun.isna().all()
        assert not senkou_a.isna().all()
        assert not senkou_b.isna().all()
        assert not chikou.isna().all()

    def test_ichimoku_extended_features(self, sample_data):
        """Test extended Ichimoku features"""
        cloud_thickness = compute_ichimoku_cloud_thickness(sample_data)
        price_distance = compute_ichimoku_price_cloud_distance(sample_data)
        composite_signal = compute_ichimoku_composite_signal(sample_data)
        trend = compute_ichimoku_trend(sample_data)

        assert len(cloud_thickness) == len(sample_data)
        assert len(price_distance) == len(sample_data)
        assert len(composite_signal) == len(sample_data)
        assert len(trend) == len(sample_data)

        # Check value ranges
        assert cloud_thickness.min() >= 0  # Thickness should be non-negative
        assert trend.isin([-1, 0, 1]).all()  # Trend should be -1, 0, or 1

    def test_ichimoku_theory_features(self, sample_data):
        """Test theoretical Ichimoku features"""
        time_theory = compute_ichimoku_time_theory(sample_data)
        wave_theory = compute_ichimoku_wave_theory(sample_data)
        value_measurement = compute_ichimoku_value_measurement(sample_data)
        momentum_confirmation = compute_ichimoku_momentum_confirmation(sample_data)

        assert len(time_theory) == len(sample_data)
        assert len(wave_theory) == len(sample_data)
        assert len(value_measurement) == len(sample_data)
        assert len(momentum_confirmation) == len(sample_data)

        # Check value ranges (should be normalized)
        assert time_theory.min() >= -2
        assert time_theory.max() <= 2
        assert wave_theory.min() >= -2
        assert wave_theory.max() <= 2
        assert value_measurement.min() >= 0  # Value measurement should be positive
        assert momentum_confirmation.min() >= -2
        assert momentum_confirmation.max() <= 2

    def test_ichimoku_advanced_features(self, sample_data):
        """Test advanced Ichimoku features"""
        cloud_slope = compute_ichimoku_cloud_slope(sample_data)
        cloud_expansion = compute_ichimoku_cloud_expansion(sample_data)
        sanyaku_kouten = compute_ichimoku_sanyaku_kouten(sample_data)

        assert len(cloud_slope) == len(sample_data)
        assert len(cloud_expansion) == len(sample_data)
        assert len(sanyaku_kouten) == len(sample_data)

        # Check value ranges
        assert cloud_slope.min() >= -2
        assert cloud_slope.max() <= 2
        assert cloud_expansion.min() >= -1
        assert cloud_expansion.max() <= 1
        assert sanyaku_kouten.min() >= -2
        assert sanyaku_kouten.max() <= 2

    def test_calculate_ichimoku_extended(self, sample_data):
        """Test the main Ichimoku calculation function"""
        result = calculate_ichimoku_extended(sample_data)

        # Check that all expected columns are present
        expected_columns = [
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_senkou_a",
            "ichimoku_senkou_b",
            "ichimoku_chikou",
            "ichimoku_cloud_thickness",
            "ichimoku_price_cloud_distance",
            "ichimoku_price_cloud_normalized",
            "ichimoku_price_position",
            "ichimoku_tk_cross",
            "ichimoku_chikou_confirmation",
            "ichimoku_cloud_color",
            "ichimoku_tenkan_ratio",
            "ichimoku_kijun_ratio",
            "ichimoku_composite_signal",
            "ichimoku_trend",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        assert len(result) == len(sample_data)

    def test_ichimoku_edge_cases(self, sample_data):
        """Test Ichimoku features with edge cases"""
        # Test with very small dataset
        small_data = sample_data.head(10)
        result = calculate_ichimoku_extended(small_data)
        assert len(result) == len(small_data)

        # Test with constant prices
        constant_data = pd.DataFrame(
            {
                "high": [100] * 50,
                "low": [100] * 50,
                "close": [100] * 50,
                "volume": [1000] * 50,
            }
        )
        result = calculate_ichimoku_extended(constant_data)
        assert len(result) == len(constant_data)
        # With constant prices, Tenkan and Kijun should equal the price
        assert result["ichimoku_tenkan"].dropna().iloc[0] == 100.0

    def test_ichimoku_nan_handling(self, sample_data):
        """Test NaN handling in Ichimoku calculations"""
        # Add some NaN values
        data_with_nan = sample_data.copy()
        data_with_nan.loc[10:15, "close"] = np.nan

        result = calculate_ichimoku_extended(data_with_nan)

        # Should handle NaN gracefully (forward/backward fill)
        assert not result.isna().all().all()  # Not all values should be NaN
        assert len(result) == len(data_with_nan)

    def test_ichimoku_performance(self, sample_data):
        """Test performance of Ichimoku calculations"""
        import time

        # Test with larger dataset
        large_data = pd.concat([sample_data] * 10, ignore_index=True)

        start_time = time.time()
        result = calculate_ichimoku_extended(large_data)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second for 1000 rows)
        assert end_time - start_time < 1.0
        assert len(result) == len(large_data)
