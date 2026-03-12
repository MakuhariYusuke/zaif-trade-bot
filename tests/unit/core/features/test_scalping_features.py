"""
Unit tests for scalping features

スキャルピング特徴量の単体テスト
"""

import pytest
import numpy as np
import pandas as pd

from ztb.features.scalping import (
    liquidity_surge,
    micro_trend,
    micro_volatility,
    momentum_divergence,
    momentum_burst,
    price_acceleration,
    price_velocity,
    realized_volatility,
    order_flow_imbalance,
    tick_volume_ratio,
    volume_surge,
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
        assert result.iloc[0] == 0.0

    def test_price_velocity_handles_zero_previous_close(self):
        """previous close が 0 の点だけ 0.0 扱いを維持する."""
        df = pd.DataFrame({"close": [100.0, 0.0, 10.0, 15.0]})

        result = price_velocity(df)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == pytest.approx(-1.0)
        assert result.iloc[2] == 0.0
        assert result.iloc[3] == pytest.approx(0.5)

    def test_micro_trend_basic_window_behavior(self):
        """window 本前との比率差分を返す."""
        df = pd.DataFrame({"close": [100.0, 110.0, 121.0, 133.1]})

        result = micro_trend(df, window=2)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == pytest.approx(0.21)
        assert result.iloc[3] == pytest.approx(0.21)

    def test_price_acceleration_matches_velocity_delta_mean(self):
        """window 区間の velocity 差分平均と一致する."""
        df = pd.DataFrame({"close": [100.0, 110.0, 132.0, 171.6]})

        result = price_acceleration(df, window=3)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 0.0
        assert result.iloc[3] == pytest.approx(0.1)

    def test_volume_surge_keeps_zero_when_prior_std_is_zero(self):
        """直前 window の標準偏差が 0 なら誤検知しない."""
        df = pd.DataFrame({"volume": [10.0, 10.0, 10.0, 30.0]})

        result = volume_surge(df, window=3, threshold=2.0)

        assert result.iloc[3] == 0.0

    def test_momentum_burst_uses_previous_window_volume_average(self):
        """price change と直前 volume 平均の両方を使う."""
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 110.0, 121.0],
                "volume": [10.0, 10.0, 10.0, 20.0],
            }
        )

        result = momentum_burst(df, window=2)

        expected_first = 0.1 * np.log(2.0)
        expected = 0.21 * np.log(3.0)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == pytest.approx(expected_first)
        assert result.iloc[3] == pytest.approx(expected)

    def test_liquidity_surge_uses_previous_window_max(self):
        """直前 window の最大 volume に対する比率を返す."""
        df = pd.DataFrame({"volume": [1.0, 2.0, 4.0, 1.0]})

        result = liquidity_surge(df, window=2)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(0.25)

    def test_momentum_divergence_matches_fast_minus_slow_change(self):
        """fast / slow 変化率差分をそのまま返す."""
        df = pd.DataFrame({"close": [100.0, 110.0, 121.0, 133.1, 146.41]})

        result = momentum_divergence(df, fast_window=2, slow_window=4)

        fast_change = (146.41 - 121.0) / 121.0
        slow_change = (146.41 - 100.0) / 100.0
        assert result.iloc[:4].eq(0.0).all()
        assert result.iloc[4] == pytest.approx(fast_change - slow_change)

    def test_micro_volatility_basic(self, sample_dataframe):
        """Test basic micro volatility calculation"""
        result = micro_volatility(sample_dataframe)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_dataframe)
        assert result.name == "micro_volatility"
        assert (result.dropna() >= 0).all()

    def test_micro_volatility_handles_zero_previous_close(self):
        """previous close が 0 の return は 0.0 扱いを維持する."""
        df = pd.DataFrame({
            "close": [100.0, 0.0, 10.0, 10.0, 20.0, 20.0],
        })

        result = micro_volatility(df, window=3)

        assert len(result) == len(df)
        assert result.iloc[3] == pytest.approx(0.5, abs=1e-9)

    def test_micro_volatility_window_larger_than_data(self):
        """window > len(df) でもゼロ埋め series を返す."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = micro_volatility(df, window=10)

        assert len(result) == 3
        assert (result == 0.0).all()

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
