"""
Unit tests for SAC v427 feature engineering, including bear and bull market features.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer


class TestSACv427FeatureEngineering:
    """Test suite for SAC v427 feature engineering"""

    @pytest.fixture(scope="module")
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

    @pytest.fixture(scope="module")
    def feature_engineer(self):
        """Create feature engineer instance"""
        return SACv427FeatureEngineer()

    @pytest.fixture(scope="module")
    def full_features(self, feature_engineer, sample_data):
        """Generate the public v427 feature set once for the module."""
        return feature_engineer.generate_v427_features(sample_data)

    def test_bear_market_features_generation(self, full_features):
        """Current public output should expose bearish regime/ensemble signals."""
        expected_bear_features = [
            "regime_bearish",
            "ensemble_confidence_bear",
        ]

        for feature in expected_bear_features:
            assert feature in full_features.columns, f"Missing bear feature: {feature}"

        assert not full_features[expected_bear_features].isna().all().all()

    def test_bull_market_features_generation(self, full_features):
        """Current public output should expose bullish regime/ensemble signals."""
        expected_bull_features = [
            "regime_bullish",
            "ensemble_confidence_bull",
        ]

        for feature in expected_bull_features:
            assert feature in full_features.columns, f"Missing bull feature: {feature}"

        assert not full_features[expected_bull_features].isna().all().all()

    def test_confidence_features_range(self, full_features):
        """Ensemble confidence features should behave like probabilities."""
        bear_values = full_features["ensemble_confidence_bear"].dropna()
        bull_values = full_features["ensemble_confidence_bull"].dropna()

        assert ((bear_values >= 0.0) & (bear_values <= 1.0)).all()
        assert ((bull_values >= 0.0) & (bull_values <= 1.0)).all()

    def test_regime_detection_balance(self, full_features):
        """Test that bear and bull regime features are mutually exclusive where appropriate"""
        strong_bear = full_features["regime_bearish"]
        strong_bull = full_features["regime_bullish"]

        # Should not have both strong bear and strong bull signals simultaneously
        overlap = (strong_bear == 1) & (strong_bull == 1)
        overlap_rate = overlap.sum() / len(overlap)
        assert overlap_rate < 0.1, f"Too much overlap between strong bear and bull regimes: {overlap_rate:.2%}"

    def test_full_feature_generation_includes_bull_bear(self, full_features, sample_data):
        """Test that full feature generation includes both bear and bull market features"""
        # Check for presence of key bear and bull features
        bear_features_present = {
            "regime_bearish",
            "ensemble_confidence_bear",
        }.issubset(full_features.columns)
        bull_features_present = {
            "regime_bullish",
            "ensemble_confidence_bull",
        }.issubset(full_features.columns)

        assert bear_features_present, "Bear market features not found in full feature set"
        assert bull_features_present, "Bull market features not found in full feature set"

        # Should have reasonable number of features
        assert len(full_features.columns) > len(sample_data.columns) + 30, "Insufficient features generated"
