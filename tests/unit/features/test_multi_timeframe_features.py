"""
Unit tests for multi-timeframe features implementation.
多時間軸特徴量実装の単体テスト
"""

import numpy as np
import pandas as pd
import pytest

from ztb.features.curated_features import get_feature_set
from ztb.features.registry import FeatureRegistry
from ztb.features.timeframe import Timeframe, get_timeframe_params


class TestMultiTimeframeFeatures:
    """Test suite for multi-timeframe features"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        np.random.seed(42)
        n = 500  # Sufficient data for multi-timeframe calculations, especially ADX with long periods

        # Create realistic OHLCV data with trend
        base_price = 100
        trend = np.linspace(0, 20, n)
        noise = np.random.normal(0, 2, n)
        close = base_price + trend + noise

        # Generate OHLC from close prices
        high = close + np.abs(np.random.normal(0, 1, n))
        low = close - np.abs(np.random.normal(0, 1, n))
        open_prices = close + np.random.normal(0, 0.5, n)
        volume = np.random.uniform(1000, 10000, n)

        # Create datetime index
        dates = pd.date_range("2023-01-01", periods=n, freq="1min")

        df = pd.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

        return df

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Setup feature registry before each test"""
        # Import feature modules to register multi-timeframe features

        yield

    def test_registry_registration(self):
        """Test that multi-timeframe features are properly registered"""
        # Expected multi-timeframe features
        expected_features = [
            # RSI multi-timeframe
            "RSI_M1",
            "RSI_M5",
            "RSI_M15",
            "RSI_H1",
            "RSI_H4",
            "RSI_D1",
            # ATR multi-timeframe
            "ATR_M1",
            "ATR_M5",
            "ATR_M15",
            "ATR_H1",
            "ATR_H4",
            "ATR_D1",
            # ADX multi-timeframe
            "ADX_M1",
            "ADX_M5",
            "ADX_M15",
            "ADX_H1",
            "ADX_H4",
            "ADX_D1",
            # HeikinAshi multi-timeframe
            "HeikinAshi_Color_M1",
            "HeikinAshi_Color_M5",
            "HeikinAshi_Color_M15",
            "HeikinAshi_Color_H1",
            "HeikinAshi_Color_H4",
            "HeikinAshi_Color_D1",
            # EMACross multi-timeframe
            "EMACross_Diff_M1",
            "EMACross_Diff_M5",
            "EMACross_Diff_M15",
            "EMACross_Diff_H1",
            "EMACross_Diff_H4",
            "EMACross_Diff_D1",
            "EMACross_Signal_M1",
            "EMACross_Signal_M5",
            "EMACross_Signal_M15",
            "EMACross_Signal_H1",
            "EMACross_Signal_H4",
            "EMACross_Signal_D1",
        ]

        # Check each feature is registered
        for feature_name in expected_features:
            func = FeatureRegistry._registry.get(feature_name)
            assert func is not None, f"Feature {feature_name} is not registered"

    def test_feature_count_in_curated_set(self):
        """Test that curated feature set includes expected number of multi-timeframe features"""
        features = get_feature_set()

        # Count multi-timeframe features
        timeframe_features = [
            f
            for f in features
            if any(tf in f for tf in ["_M1", "_M5", "_M15", "_H1", "_H4", "_D1"])
        ]

        # Should have 78 multi-timeframe features based on implementation
        assert (
            len(timeframe_features) == 78
        ), f"Expected 78 multi-timeframe features, got {len(timeframe_features)}"

        # Total features should be 156
        assert len(features) == 156, f"Expected 156 total features, got {len(features)}"

    def test_rsi_multi_timeframe_calculation(self, sample_data):
        """Test RSI multi-timeframe calculations"""
        timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]

        for tf in timeframes:
            feature_name = f"RSI_{tf.name}"
            func = FeatureRegistry._registry.get(feature_name)
            assert func is not None, f"RSI feature for {tf.name} not found"

            result = func(sample_data)

            # Check result properties
            assert len(result) == len(sample_data), f"RSI_{tf.name} length mismatch"
            assert not result.isna().all(), f"RSI_{tf.name} is all NaN"

            # RSI should be between 0 and 100
            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert (
                    (valid_values >= 0).all() and (valid_values <= 100).all()
                ), f"RSI_{tf.name} values out of range: {valid_values.min()} - {valid_values.max()}"

    def test_atr_multi_timeframe_calculation(self, sample_data):
        """Test ATR multi-timeframe calculations"""
        timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]

        for tf in timeframes:
            feature_name = f"ATR_{tf.name}"
            func = FeatureRegistry._registry.get(feature_name)
            assert func is not None, f"ATR feature for {tf.name} not found"

            result = func(sample_data)

            # Check result properties
            assert len(result) == len(sample_data), f"ATR_{tf.name} length mismatch"
            assert not result.isna().all(), f"ATR_{tf.name} is all NaN"

            # ATR should be positive
            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all(), f"ATR_{tf.name} has negative values"

    def test_adx_multi_timeframe_calculation(self, sample_data):
        """Test ADX multi-timeframe calculations"""
        timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]

        for tf in timeframes:
            feature_name = f"ADX_{tf.name}"
            func = FeatureRegistry._registry.get(feature_name)
            assert func is not None, f"ADX feature for {tf.name} not found"

            result = func(sample_data)

            # Check result properties
            assert len(result) == len(sample_data), f"ADX_{tf.name} length mismatch"
            assert not result.isna().all(), f"ADX_{tf.name} is all NaN"

            # ADX should be between 0 and 100
            valid_values = result.dropna()
            if len(valid_values) > 0:
                assert (
                    (valid_values >= 0).all() and (valid_values <= 100).all()
                ), f"ADX_{tf.name} values out of range: {valid_values.min()} - {valid_values.max()}"

    def test_heikinashi_multi_timeframe_calculation(self, sample_data):
        """Test HeikinAshi multi-timeframe calculations"""
        timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]

        for tf in timeframes:
            feature_name = f"HeikinAshi_Color_{tf.name}"
            func = FeatureRegistry._registry.get(feature_name)
            assert func is not None, f"HeikinAshi feature for {tf.name} not found"

            result = func(sample_data)

            # Check result properties
            assert len(result) == len(
                sample_data
            ), f"HeikinAshi_Color_{tf.name} length mismatch"
            assert not result.isna().all(), f"HeikinAshi_Color_{tf.name} is all NaN"

            # HeikinAshi color should be discrete values (typically -1, 0, 1 or similar)
            valid_values = result.dropna()
            if len(valid_values) > 0:
                unique_values = sorted(valid_values.unique())
                assert (
                    len(unique_values) <= 5
                ), f"HeikinAshi_Color_{tf.name} has too many unique values: {unique_values}"

    def test_emacross_multi_timeframe_calculation(self, sample_data):
        """Test EMACross multi-timeframe calculations"""
        timeframes = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]

        for tf in timeframes:
            # Test both Diff and Signal variants
            for variant in ["Diff", "Signal"]:
                feature_name = f"EMACross_{variant}_{tf.name}"
                func = FeatureRegistry._registry.get(feature_name)
                assert (
                    func is not None
                ), f"EMACross_{variant} feature for {tf.name} not found"

                result = func(sample_data)

                # Check result properties
                assert len(result) == len(
                    sample_data
                ), f"EMACross_{variant}_{tf.name} length mismatch"
                assert (
                    not result.isna().all()
                ), f"EMACross_{variant}_{tf.name} is all NaN"

                # Additional checks based on variant
                valid_values = result.dropna()
                if len(valid_values) > 0:
                    if variant == "Signal":
                        # Signal should be binary (0 or 1)
                        unique_values = sorted(valid_values.unique())
                        assert all(
                            val in [0, 1] for val in unique_values
                        ), f"EMACross_Signal_{tf.name} should be binary, got: {unique_values}"
                    elif variant == "Diff":
                        # Diff can be any normalized value, but should be reasonable range
                        assert (
                            valid_values.abs().max() < 10
                        ), f"EMACross_Diff_{tf.name} values seem unreasonable: {valid_values.min()} - {valid_values.max()}"

    def test_timeframe_params(self):
        """Test timeframe parameter calculation"""
        # Test each timeframe has valid parameters
        for tf in [
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
        ]:
            params = get_timeframe_params(tf)

            # Should return a dict with required keys
            assert isinstance(
                params, dict
            ), f"Timeframe {tf.name} params should be dict"
            assert (
                "short_period" in params
            ), f"Timeframe {tf.name} missing 'short_period' param"
            assert (
                "medium_period" in params
            ), f"Timeframe {tf.name} missing 'medium_period' param"
            assert (
                "long_period" in params
            ), f"Timeframe {tf.name} missing 'long_period' param"

            # Values should be reasonable
            assert (
                params["short_period"] > 0
            ), f"Timeframe {tf.name} short_period should be positive"
            assert (
                params["medium_period"] > 0
            ), f"Timeframe {tf.name} medium_period should be positive"
            assert (
                params["long_period"] > 0
            ), f"Timeframe {tf.name} long_period should be positive"

    def test_timeframe_hierarchy(self):
        """Test that higher timeframes have appropriately scaled parameters"""
        m1_params = get_timeframe_params(Timeframe.M1)
        h1_params = get_timeframe_params(Timeframe.H1)

        # H1 should have larger periods than M1
        assert (
            h1_params["short_period"] > m1_params["short_period"]
        ), f"H1 short_period ({h1_params['short_period']}) should be larger than M1 short_period ({m1_params['short_period']})"
        assert (
            h1_params["medium_period"] > m1_params["medium_period"]
        ), f"H1 medium_period ({h1_params['medium_period']}) should be larger than M1 medium_period ({m1_params['medium_period']})"
        assert (
            h1_params["long_period"] > m1_params["long_period"]
        ), f"H1 long_period ({h1_params['long_period']}) should be larger than M1 long_period ({m1_params['long_period']})"

    def test_feature_calculation_consistency(self, sample_data):
        """Test that multi-timeframe features produce consistent results for same input"""
        # Calculate same feature multiple times - should be deterministic
        feature_name = "RSI_M5"
        func = FeatureRegistry._registry.get(feature_name)
        assert func is not None

        result1 = func(sample_data)
        result2 = func(sample_data)

        # Results should be identical
        pd.testing.assert_series_equal(result1, result2, check_names=False)
