"""
379# Unit tests for market theory features

035#-306# pre-366# 市場理論特徴量の単体テスト
"""

import math

import numpy as np
import pandas as pd
import pytest

from ztb.features.market_theory import (
    amihud_illiq,
    ema_velocity_bps,
    kyle_lambda_proxy,
    parkinson_sigma,
    vpin_proxy,
)


class TestMarketTheoryFeatures:
    """Market theory features unit tests"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Sample OHLCV DataFrame for testing."""
        np.random.seed(42)
        n = 100
        base = 50000.0
        prices = [base]
        for i in range(1, n):
            change = np.random.normal(0, 0.003)
            prices.append(max(prices[-1] * (1 + change), 1000))

        rows = []
        for i, close in enumerate(prices):
            vol = abs(np.random.normal(0, 0.002))
            high = close * (1 + vol)
            low = close * (1 - vol)
            open_p = prices[i - 1] if i > 0 else close
            volume = float(np.random.randint(100, 1000))
            rows.append(
                {"open": open_p, "high": high, "low": low, "close": close, "volume": volume}
            )
        return pd.DataFrame(rows)

    @pytest.fixture
    def flat_df(self) -> pd.DataFrame:
        """Flat price DataFrame (no movement)."""
        n = 50
        return pd.DataFrame(
            {
                "open": [50000.0] * n,
                "high": [50000.0] * n,
                "low": [50000.0] * n,
                "close": [50000.0] * n,
                "volume": [500.0] * n,
            }
        )

    # ----------------------------------------------------------------
    # parkinson_sigma
    # ----------------------------------------------------------------

    def test_parkinson_sigma_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = parkinson_sigma(sample_df)
        assert len(result) == len(sample_df)
        assert result.name == "parkinson_sigma"

    def test_parkinson_sigma_non_negative(self, sample_df: pd.DataFrame) -> None:
        result = parkinson_sigma(sample_df)
        assert (result >= 0).all(), "Parkinson σ should be non-negative"

    def test_parkinson_sigma_flat_near_zero(self, flat_df: pd.DataFrame) -> None:
        result = parkinson_sigma(flat_df)
        assert result.max() < 1e-8, "Flat price should have σ ≈ 0"

    def test_parkinson_sigma_known_value(self) -> None:
        """H/L = 2 → σ_P = ln(2) / (2√ln2) ≈ 0.5887"""
        df = pd.DataFrame(
            {"open": [100], "high": [200], "low": [100], "close": [150], "volume": [1000]}
        )
        result = parkinson_sigma(df, window=1)
        expected = math.log(2) / (2 * math.sqrt(math.log(2)))
        assert abs(result.iloc[0] - expected) < 1e-4

    def test_parkinson_sigma_no_nan(self, sample_df: pd.DataFrame) -> None:
        result = parkinson_sigma(sample_df)
        assert not result.isna().any()

    # ----------------------------------------------------------------
    # vpin_proxy
    # ----------------------------------------------------------------

    def test_vpin_proxy_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = vpin_proxy(sample_df)
        assert len(result) == len(sample_df)
        assert result.name == "vpin_proxy"

    def test_vpin_proxy_range(self, sample_df: pd.DataFrame) -> None:
        result = vpin_proxy(sample_df)
        assert (result >= 0).all(), "VPIN should be non-negative"
        assert (result <= 1.0 + 1e-6).all(), "VPIN should be ≤ 1"

    def test_vpin_proxy_no_nan(self, sample_df: pd.DataFrame) -> None:
        result = vpin_proxy(sample_df)
        assert not result.isna().any()

    def test_vpin_proxy_empty_df(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = vpin_proxy(df)
        assert len(result) == 0

    # ----------------------------------------------------------------
    # kyle_lambda_proxy
    # ----------------------------------------------------------------

    def test_kyle_lambda_proxy_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = kyle_lambda_proxy(sample_df)
        assert len(result) == len(sample_df)
        assert result.name == "kyle_lambda_proxy"

    def test_kyle_lambda_proxy_no_nan(self, sample_df: pd.DataFrame) -> None:
        result = kyle_lambda_proxy(sample_df)
        assert not result.isna().any()

    def test_kyle_lambda_proxy_z_score_property(self, sample_df: pd.DataFrame) -> None:
        """Z-score normalized features should have near-zero mean over large windows."""
        result = kyle_lambda_proxy(sample_df, window=20)
        # Mean of z-scores over windowed data should be close to 0
        # (not exactly 0 due to rolling window warmup)
        tail_mean = result.iloc[30:].mean()
        assert abs(tail_mean) < 1.0, f"Z-score tail mean too large: {tail_mean}"

    # ----------------------------------------------------------------
    # amihud_illiq
    # ----------------------------------------------------------------

    def test_amihud_illiq_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = amihud_illiq(sample_df)
        assert len(result) == len(sample_df)
        assert result.name == "amihud_illiq"

    def test_amihud_illiq_no_nan(self, sample_df: pd.DataFrame) -> None:
        result = amihud_illiq(sample_df)
        assert not result.isna().any()

    def test_amihud_illiq_flat_near_zero(self, flat_df: pd.DataFrame) -> None:
        """No price change → |return| = 0 → ILLIQ = 0."""
        result = amihud_illiq(flat_df)
        assert result.abs().max() < 1e-6

    def test_amihud_illiq_single_row(self) -> None:
        df = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50000], "volume": [500]}
        )
        result = amihud_illiq(df)
        assert len(result) == 1
        assert not result.isna().any()

    # ----------------------------------------------------------------
    # ema_velocity_bps
    # ----------------------------------------------------------------

    def test_ema_velocity_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = ema_velocity_bps(sample_df)
        assert len(result) == len(sample_df)
        assert result.name == "ema_velocity_bps"

    def test_ema_velocity_no_nan(self, sample_df: pd.DataFrame) -> None:
        result = ema_velocity_bps(sample_df)
        assert not result.isna().any()

    def test_ema_velocity_flat_zero(self, flat_df: pd.DataFrame) -> None:
        result = ema_velocity_bps(flat_df)
        assert result.abs().max() < 1e-6

    def test_ema_velocity_single_row(self) -> None:
        df = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50000], "volume": [500]}
        )
        result = ema_velocity_bps(df)
        assert len(result) == 1
        # First bar velocity = 0 (no previous bar)
        assert abs(result.iloc[0]) < 1e-6

    def test_ema_velocity_uptrend_positive(self) -> None:
        """Consistent uptrend → EMA velocity should be positive."""
        n = 50
        prices = [50000 + i * 10 for i in range(n)]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 5 for p in prices],
                "low": [p - 5 for p in prices],
                "close": prices,
                "volume": [500] * n,
            }
        )
        result = ema_velocity_bps(df)
        # After warmup, velocity should be consistently positive
        assert (result.iloc[5:] > 0).all()

    # ----------------------------------------------------------------
    # FeatureRegistry integration
    # ----------------------------------------------------------------

    def test_features_registered(self) -> None:
        """Verify all 5 market theory features are registered."""
        from ztb.features.core.registry import FeatureRegistry

        expected = [
            "parkinson_sigma",
            "vpin_proxy",
            "kyle_lambda_proxy",
            "amihud_illiq",
            "ema_velocity_bps",
        ]
        registered = FeatureRegistry.list()
        for feat in expected:
            assert feat in registered, f"Feature '{feat}' not registered in FeatureRegistry"

    def test_registry_compute_roundtrip(self, sample_df: pd.DataFrame) -> None:
        """Test that FeatureRegistry.get() returns working compute functions."""
        from ztb.features.core.registry import FeatureRegistry

        for name in ["parkinson_sigma", "vpin_proxy", "kyle_lambda_proxy", "amihud_illiq", "ema_velocity_bps"]:
            func = FeatureRegistry.get(name)
            result = func(sample_df)
            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_df)
            assert not result.isna().any(), f"{name} has NaN values"
