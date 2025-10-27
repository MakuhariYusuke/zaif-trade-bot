"""
Unit tests for SAC v427 feature engineering, including bear and bull market features.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer


class TestSACv427FeatureEngineering:
    """Test suite for SAC v427 feature engineering"""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        np.random.seed(42)
        n = 200  # Sufficient data for rolling calculations

        # Create realistic OHLCV data with trends
        dates = pd.date_range("2023-01-01", periods=n, freq="D")

        # Generate trending data (bullish first half, bearish second half)
        trend = np.concatenate([
            np.linspace(100, 150, n//2),  # Bullish trend
            np.linspace(150, 120, n//2)   # Bearish trend
        ])

        close = trend + np.random.randn(n) * 2
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2
        open_price = close + np.random.randn(n) * 1
        volume = np.random.randint(1000, 10000, n)

        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return SACv427FeatureEngineer()

    def test_bear_market_features_generation(self, feature_engineer, sample_data):
        """Test bear market features are generated correctly"""
        bear_features = feature_engineer._generate_bear_market_features(sample_data)

        # Check that bear market features are present
        expected_bear_features = [
            "bear_momentum_5", "bear_momentum_10", "bear_momentum_20",
            "bear_acceleration_5", "bear_acceleration_10", "bear_acceleration_20",
            "bear_volume_confirmation", "bear_rsi_divergence",
            "bear_macd_crossover", "bear_macd_histogram",
            "bear_regime_strong", "bear_regime_moderate", "bear_regime_weak",
            "bear_persistence_3", "bear_persistence_5", "bear_persistence_10",
            "bear_large_candle", "bear_engulfing", "bear_volatility", "bear_drawdown"
        ]

        for feature in expected_bear_features:
            assert feature in bear_features.columns, f"Missing bear feature: {feature}"

        # Check data types and shapes
        assert len(bear_features) == len(sample_data)
        assert not bear_features.isna().all().all()

    def test_bull_market_features_generation(self, feature_engineer, sample_data):
        """Test bull market features are generated correctly"""
        bull_features = feature_engineer._generate_bull_market_features(sample_data)

        # Check that bull market features are present
        expected_bull_features = [
            "bull_momentum_5", "bull_momentum_10", "bull_momentum_20",
            "bull_acceleration_5", "bull_acceleration_10", "bull_acceleration_20",
            "bull_volume_confirmation", "bull_rsi_divergence",
            "bull_macd_crossover", "bull_macd_histogram",
            "bull_regime_strong", "bull_regime_moderate", "bull_regime_weak",
            "bull_persistence_3", "bull_persistence_5", "bull_persistence_10",
            "bull_large_candle", "bull_engulfing", "bull_volatility", "bull_rally"
        ]

        for feature in expected_bull_features:
            assert feature in bull_features.columns, f"Missing bull feature: {feature}"

        # Check data types and shapes
        assert len(bull_features) == len(sample_data)
        assert not bull_features.isna().all().all()

    def test_bear_momentum_features(self, feature_engineer, sample_data):
        """Test bear momentum features calculation"""
        bear_features = feature_engineer._generate_bear_market_features(sample_data)

        # Bear momentum should be <= 0 (clipped to negative values)
        for period in [5, 10, 20]:
            momentum_col = f"bear_momentum_{period}"
            # Check non-NaN values
            valid_values = bear_features[momentum_col].dropna()
            if len(valid_values) > 0:
                assert (valid_values <= 0).all(), f"{momentum_col} should be non-positive"

            accel_col = f"bear_acceleration_{period}"
            valid_accel = bear_features[accel_col].dropna()
            if len(valid_accel) > 0:
                assert (valid_accel <= 0).all(), f"{accel_col} should be non-positive"

    def test_bull_momentum_features(self, feature_engineer, sample_data):
        """Test bull momentum features calculation"""
        bull_features = feature_engineer._generate_bull_market_features(sample_data)

        # Bull momentum should be >= 0 (clipped to positive values)
        for period in [5, 10, 20]:
            momentum_col = f"bull_momentum_{period}"
            # Check non-NaN values
            valid_values = bull_features[momentum_col].dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all(), f"{momentum_col} should be non-negative"

            accel_col = f"bull_acceleration_{period}"
            valid_accel = bull_features[accel_col].dropna()
            if len(valid_accel) > 0:
                assert (valid_accel >= 0).all(), f"{accel_col} should be non-negative"

    def test_regime_detection_balance(self, feature_engineer, sample_data):
        """Test that bear and bull regime features are mutually exclusive where appropriate"""
        bear_features = feature_engineer._generate_bear_market_features(sample_data)
        bull_features = feature_engineer._generate_bull_market_features(sample_data)

        # Strong regimes should be mutually exclusive (can't be both strong bull and bear)
        strong_bear = bear_features["bear_regime_strong"]
        strong_bull = bull_features["bull_regime_strong"]

        # Should not have both strong bear and strong bull signals simultaneously
        overlap = (strong_bear == 1) & (strong_bull == 1)
        overlap_rate = overlap.sum() / len(overlap)
        assert overlap_rate < 0.1, f"Too much overlap between strong bear and bull regimes: {overlap_rate:.2%}"

    def test_volume_confirmation_features(self, feature_engineer, sample_data):
        """Test volume confirmation features"""
        bear_features = feature_engineer._generate_bear_market_features(sample_data)
        bull_features = feature_engineer._generate_bull_market_features(sample_data)

        # Volume confirmation should be >= 0 (check non-NaN values)
        bear_vol = bear_features["bear_volume_confirmation"].dropna()
        if len(bear_vol) > 0:
            assert (bear_vol >= 0).all()
        
        bull_vol = bull_features["bull_volume_confirmation"].dropna()
        if len(bull_vol) > 0:
            assert (bull_vol >= 0).all()

    def test_full_feature_generation_includes_bull_bear(self, feature_engineer, sample_data):
        """Test that full feature generation includes both bear and bull market features"""
        full_features = feature_engineer.generate_v427_features(sample_data)

        # Check for presence of key bear and bull features
        bear_features_present = any("bear_" in col for col in full_features.columns)
        bull_features_present = any("bull_" in col for col in full_features.columns)

        assert bear_features_present, "Bear market features not found in full feature set"
        assert bull_features_present, "Bull market features not found in full feature set"

        # Should have reasonable number of features
        assert len(full_features.columns) > len(sample_data.columns) + 30, "Insufficient features generated"