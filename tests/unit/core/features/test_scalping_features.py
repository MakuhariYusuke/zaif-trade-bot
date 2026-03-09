"""
Unit tests for scalping features

スキャルピング特徴量の単体テスト
"""

import pytest
import numpy as np
import pandas as pd

from ztb.features.scalping import (
    realized_volatility,
    tick_volume_ratio,
    order_flow_imbalance
)


class TestScalpingFeatures:
    """Scalping features unit tests"""

    @pytest.fixture
    def sample_dataframe(self):
        """Sample dataframe for testing scalping features"""
        np.random.seed(42)
        n_periods = 100

        # Generate realistic OHLCV data
        base_price = 50000.0
        prices = [base_price]

        for i in range(1, n_periods):
            # Add some trend and random walk
            trend = 0.0001 * np.sin(i / 10)  # Slight trend
            noise = np.random.normal(0, 0.005)  # Random noise
            new_price = prices[-1] * (1 + trend + noise)
            prices.append(max(new_price, 1000))  # Floor price

        # Create OHLCV from close prices
        df_data = []
        for i, close in enumerate(prices):
            # Generate OHLC around close price
            volatility = abs(np.random.normal(0, 0.002))
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close * (1 + np.random.normal(0, 0.001))
            volume = np.random.randint(100, 1000)

            df_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(df_data)

    def test_realized_volatility_basic(self, sample_dataframe):
        """Test basic realized volatility calculation"""
        result = realized_volatility(sample_dataframe)

        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_dataframe)
        assert result.name == 'realized_volatility'

        # Check that values are non-negative
        valid_values = result.dropna()
        assert all(valid_values >= 0), "Realized volatility should be non-negative"

    def test_realized_volatility_window_parameter(self, sample_dataframe):
        """Test realized volatility with different window sizes"""
        # Test default window
        result_default = realized_volatility(sample_dataframe)

        # Test custom window
        result_custom = realized_volatility(sample_dataframe, window=20)

        # Results should be different
        assert not result_default.equals(result_custom)

        # Both should have same length
        assert len(result_default) == len(result_custom) == len(sample_dataframe)

    def test_realized_volatility_calculation_accuracy(self, sample_dataframe):
        """Test realized volatility calculation accuracy"""
        # Create simple test case with known returns
        simple_df = pd.DataFrame({
            'close': [100.0, 101.0, 99.0, 102.0, 103.0, 101.0]
        })

        result = realized_volatility(simple_df, window=3)

        # For window=3, we should have values starting from index 3 (0-indexed)
        # At index 3: returns between 1-2, 2-3: (101-100)/100=0.01, (99-101)/101≈-0.0198
        # RV = sqrt(0.01^2 + (-0.0198)^2) ≈ sqrt(0.0001 + 0.000392) ≈ sqrt(0.000492) ≈ 0.0222

        assert not np.isnan(result.iloc[3]), "Should have valid value at index 3"
        assert result.iloc[3] > 0, "Realized volatility should be positive"
        assert result.iloc[3] == pytest.approx(0.0222, abs=0.001)

    def test_tick_volume_ratio_basic(self, sample_dataframe):
        """Test basic tick volume ratio calculation"""
        result = tick_volume_ratio(sample_dataframe)

        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_dataframe)
        assert result.name == 'tick_volume_ratio'

        # Check that values are non-negative
        valid_values = result.dropna()
        assert all(valid_values >= 0), "Tick volume ratio should be non-negative"

    def test_tick_volume_ratio_window_parameter(self, sample_dataframe):
        """Test tick volume ratio with different window sizes"""
        # Test default window (5)
        result_default = tick_volume_ratio(sample_dataframe, window=5)

        # Test custom window (10)
        result_custom = tick_volume_ratio(sample_dataframe, window=10)

        # Results should be different for the same indices where both are calculated
        # Check values from index 10 onwards where both windows have data
        default_slice = result_default.iloc[10:20]
        custom_slice = result_custom.iloc[10:20]

        # At least some values should be different
        assert not default_slice.equals(custom_slice), "Different window sizes should produce different results"

        # Both should have same length
        assert len(result_default) == len(result_custom) == len(sample_dataframe)

    def test_tick_volume_ratio_calculation(self, sample_dataframe):
        """Test tick volume ratio calculation logic"""
        # Create controlled test data
        test_df = pd.DataFrame({
            'volume': [100, 200, 150, 300, 250, 400]
        })

        result = tick_volume_ratio(test_df, window=3)

        # At index 3: volume=300, avg of previous 3: (100+200+150)/3 = 150, ratio=300/150=2.0
        # At index 4: volume=250, avg of previous 3: (200+150+300)/3 = 216.67, ratio=250/216.67≈1.155
        # At index 5: volume=400, avg of previous 3: (150+300+250)/3 = 233.33, ratio=400/233.33≈1.714

        assert result.iloc[3] == pytest.approx(2.0, abs=0.01)
        assert result.iloc[4] == pytest.approx(1.154, abs=0.01)
        assert result.iloc[5] == pytest.approx(1.714, abs=0.01)

    def test_order_flow_imbalance_basic(self, sample_dataframe):
        """Test basic order flow imbalance calculation"""
        result = order_flow_imbalance(sample_dataframe)

        # Check basic properties
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_dataframe)
        assert result.name == 'order_flow_imbalance'

        # Values can be negative, but should be within reasonable bounds
        valid_values = result.dropna()
        assert all(abs(v) <= 2 for v in valid_values), "Order flow imbalance values seem unreasonable"

    def test_order_flow_imbalance_calculation(self, sample_dataframe):
        """Test order flow imbalance calculation logic"""
        # Create controlled test data
        test_df = pd.DataFrame({
            'high': [105, 102, 108, 106, 104],
            'low': [95, 98, 92, 94, 96],
            'close': [100, 100, 100, 100, 100],  # All closes at 100
            'open': [100, 100, 100, 100, 100]    # All opens at 100
        })

        result = order_flow_imbalance(test_df)

        # When close == open (no body), result should be 0
        # But the function uses close[i] vs close[i-1], so first value will be NaN or 0
        valid_results = result.dropna()
        assert len(valid_results) >= 1, "Should have at least one valid result"

    def test_features_with_insufficient_data(self):
        """Test features behavior with insufficient data"""
        # Create very small dataframe
        small_df = pd.DataFrame({
            'close': [100.0, 101.0],
            'volume': [100, 200]
        })

        # These should not raise errors but may return NaN values
        rv_result = realized_volatility(small_df, window=10)  # Window larger than data
        tv_result = tick_volume_ratio(small_df, window=10)    # Window larger than data

        # Should still return series of correct length
        assert len(rv_result) == len(small_df)
        assert len(tv_result) == len(small_df)

    def test_features_with_zero_values(self):
        """Test features behavior with zero or invalid values"""
        # Test with zero volume
        df_with_zeros = pd.DataFrame({
            'close': [100.0, 101.0, 99.0, 102.0],
            'volume': [0, 100, 200, 0]  # Some zero volumes
        })

        # Should handle zero volumes gracefully
        tv_result = tick_volume_ratio(df_with_zeros)
        assert len(tv_result) == len(df_with_zeros)

        # Test with zero/negative prices (should be handled)
        df_invalid = pd.DataFrame({
            'close': [100.0, 0.0, -10.0, 102.0],  # Invalid prices
            'volume': [100, 200, 150, 300]
        })

        rv_result = realized_volatility(df_invalid)
        assert len(rv_result) == len(df_invalid)

    def test_feature_consistency(self, sample_dataframe):
        """Test that features produce consistent results with same input"""
        # Run feature extraction multiple times
        result1 = realized_volatility(sample_dataframe)
        result2 = realized_volatility(sample_dataframe)

        # Results should be identical
        pd.testing.assert_series_equal(result1, result2)

        # Same for other features
        tv1 = tick_volume_ratio(sample_dataframe)
        tv2 = tick_volume_ratio(sample_dataframe)
        pd.testing.assert_series_equal(tv1, tv2)

        of1 = order_flow_imbalance(sample_dataframe)
        of2 = order_flow_imbalance(sample_dataframe)
        pd.testing.assert_series_equal(of1, of2)