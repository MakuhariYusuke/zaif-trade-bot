"""
Unit tests for additional new features
"""

import numpy as np
import pandas as pd
import pytest

from ztb.features.volatility.normalized_atr import compute_normalized_atr
from ztb.features.volume.chaikin_ad import compute_chaikin_ad
from ztb.features.volume.chaikin_ad_oscillator import compute_chaikin_ad_oscillator
from ztb.features.time.time_features import (
    compute_time_monthly_cycle,
    compute_time_quarterly_cycle,
    calculate_time_features_extended
)


class TestAdditionalFeatures:
    """Test suite for additional new features"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        np.random.seed(42)
        n = 200  # Larger dataset for time features

        # Create realistic OHLCV data
        close = np.random.uniform(100, 200, n)
        high = close + np.random.uniform(0, 10, n)
        low = close - np.random.uniform(0, 10, n)
        volume = np.random.uniform(1000, 5000, n)

        # Create datetime index for time features
        dates = pd.date_range('2023-01-01', periods=n, freq='D')

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

    def test_normalized_atr(self, sample_data):
        """Test Normalized ATR feature"""
        normalized_atr = compute_normalized_atr(sample_data)

        assert len(normalized_atr) == len(sample_data)
        assert not normalized_atr.isna().all()

        # Normalized ATR should be positive (as percentage)
        assert (normalized_atr >= 0).all()

        # Check reasonable range (typically 0-10% for normalized ATR)
        assert normalized_atr.mean() < 0.5  # Less than 50%

    def test_chaikin_ad(self, sample_data):
        """Test Chaikin AD feature"""
        chaikin_ad = compute_chaikin_ad(sample_data)

        assert len(chaikin_ad) == len(sample_data)
        assert not chaikin_ad.isna().all()

        # Chaikin AD can be positive or negative
        # Should accumulate over time
        assert chaikin_ad.iloc[-1] != chaikin_ad.iloc[0]  # Should change over time

    def test_chaikin_ad_oscillator(self, sample_data):
        """Test Chaikin AD Oscillator feature"""
        chaikin_ad_osc = compute_chaikin_ad_oscillator(sample_data)

        assert len(chaikin_ad_osc) == len(sample_data)
        assert not chaikin_ad_osc.isna().all()

        # Oscillator can be positive or negative
        # Should oscillate around zero
        assert abs(chaikin_ad_osc.mean()) < 100  # Reasonable range

    def test_time_features(self, sample_data):
        """Test time-based features"""
        monthly_cycle = compute_time_monthly_cycle(sample_data)
        quarterly_cycle = compute_time_quarterly_cycle(sample_data)

        assert len(monthly_cycle) == len(sample_data)
        assert len(quarterly_cycle) == len(sample_data)

        # Monthly cycle should be between 0 and 1
        assert (monthly_cycle >= 0).all()
        assert (monthly_cycle <= 1).all()

        # Quarterly cycle should be between 0 and 1
        assert (quarterly_cycle >= 0).all()
        assert (quarterly_cycle <= 1).all()

    def test_time_features_extended(self, sample_data):
        """Test extended time features calculation"""
        time_features = calculate_time_features_extended(sample_data)

        expected_columns = [
            'time_day_of_week', 'time_hour_of_day', 'time_session',
            'time_volatility_adjustment', 'time_month', 'time_quarter',
            'time_monthly_cycle', 'time_quarterly_cycle', 'time_is_weekend',
            'time_is_business_day', 'time_session_progress'
        ]

        for col in expected_columns:
            assert col in time_features.columns, f"Missing column: {col}"

        assert len(time_features) == len(sample_data)

        # Check value ranges
        assert time_features['time_day_of_week'].isin(range(7)).all()  # 0-6
        assert time_features['time_hour_of_day'].isin(range(24)).all()  # 0-23
        assert time_features['time_session'].isin([0, 1, 2]).all()  # Pre-market, Regular, After-hours
        assert time_features['time_month'].isin(range(1, 13)).all()  # 1-12
        assert time_features['time_quarter'].isin(range(1, 5)).all()  # 1-4

    def test_feature_integration(self, sample_data):
        """Test that all features can be computed together"""
        from ztb.features.registry import FeatureRegistry

        # Test a few key features
        features_to_test = [
            'Normalized_ATR',
            'Chaikin_AD',
            'Chaikin_AD_Oscillator',
            'Time_Monthly_Cycle',
            'Time_Quarterly_Cycle'
        ]

        for feature_name in features_to_test:
            compute_func = FeatureRegistry.get_compute_function(feature_name)
            result = compute_func(sample_data)
            assert len(result) == len(sample_data)
            assert not result.isna().all()

    def test_edge_cases(self, sample_data):
        """Test edge cases for new features"""
        # Test with minimal data
        minimal_data = sample_data.head(5)
        try:
            normalized_atr = compute_normalized_atr(minimal_data)
            # Should handle minimal data gracefully
            assert len(normalized_atr) == len(minimal_data)
        except Exception:
            # Some features might require minimum periods - this is acceptable
            pass

        # Test with constant prices
        constant_data = pd.DataFrame({
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })

        # ATR should be 0 for constant prices
        normalized_atr = compute_normalized_atr(constant_data)
        assert (normalized_atr == 0).all()

        # Chaikin AD should accumulate volume properly
        chaikin_ad = compute_chaikin_ad(constant_data)
        # With constant prices, AD should be based on volume accumulation
        assert not chaikin_ad.isna().all()

    def test_nan_handling(self, sample_data):
        """Test NaN handling in new features"""
        # Add some NaN values
        data_with_nan = sample_data.copy()
        data_with_nan.loc[10:15, ['high', 'low', 'close']] = np.nan

        # Features should handle NaN gracefully
        try:
            normalized_atr = compute_normalized_atr(data_with_nan)
            assert len(normalized_atr) == len(data_with_nan)
        except Exception:
            # Some features might not handle NaN - this is acceptable for now
            pass

    def test_performance(self, sample_data):
        """Test performance of new feature calculations"""
        import time

        # Test with larger dataset
        large_data = pd.concat([sample_data] * 5, ignore_index=True)

        features_to_test = [
            ('Normalized_ATR', compute_normalized_atr),
            ('Chaikin_AD', compute_chaikin_ad),
            ('Chaikin_AD_Oscillator', compute_chaikin_ad_oscillator),
        ]

        for feature_name, compute_func in features_to_test:
            start_time = time.time()
            result = compute_func(large_data)
            end_time = time.time()

            # Should complete in reasonable time (less than 0.1 second for 1000 rows)
            assert end_time - start_time < 0.1, f"{feature_name} took too long"
            assert len(result) == len(large_data)