"""
microstructure.py ユニットテスト — 10 特徴量の計算正確性を検証.

add_microstructure_features() が real 板/約定データから正しく特徴量を算出するか、
既知入力 → 既知出力のパターンで検証する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ztb.features.microstructure import (
    MICROSTRUCTURE_FEATURES,
    add_microstructure_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_1min_df(
    n: int = 30,
    *,
    spread_bps: float = 2.0,
    depth_bias: float = 0.0,
    trade_bias: float = 0.0,
) -> pd.DataFrame:
    """real aggregate_to_1min 出力を模したテスト DataFrame.

    Args:
        n: 行数 (1分足)
        spread_bps: bid-ask spread (bps of mid)
        depth_bias: depth_imbalance に加算 (-1~+1)
        trade_bias: trade_flow_imbalance に加算 (-1~+1)
    """
    np.random.seed(42)
    mid = 10_300_000 + np.cumsum(np.random.randn(n) * 500)
    half_spread = mid * spread_bps * 1e-4 / 2

    idx = pd.date_range("2026-02-13 10:00", periods=n, freq="1min", tz="UTC")

    df = pd.DataFrame(
        {
            "best_bid": mid - half_spread,
            "best_ask": mid + half_spread,
            "mid_price": mid,
            "spread": spread_bps * 1e-4 * np.ones(n),
            "bid_vol_5": np.random.uniform(0.05, 0.5, n) + depth_bias * 0.1,
            "ask_vol_5": np.random.uniform(0.05, 0.5, n) - depth_bias * 0.1,
            "depth_imbalance": np.clip(
                np.random.uniform(-0.5, 0.5, n) + depth_bias, -1, 1
            ),
            "buy_volume": np.random.uniform(0.1, 2.0, n) + max(trade_bias, 0) * 0.5,
            "sell_volume": np.random.uniform(0.1, 2.0, n) + max(-trade_bias, 0) * 0.5,
            "trade_count": np.random.randint(5, 50, n).astype(float),
            "vwap": mid + np.random.randn(n) * 100,
            "trade_flow_imbalance": np.clip(
                np.random.uniform(-0.5, 0.5, n) + trade_bias, -1, 1
            ),
            "close": mid,
        },
        index=idx,
    )
    return df


# =====================================================================
# Feature presence / shape
# =====================================================================

class TestFeaturePresence:
    """全 10 特徴量が生成されることを検証."""

    def test_all_10_features_present(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        for feat in MICROSTRUCTURE_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_output_rows_match_input(self) -> None:
        df = _make_1min_df(50)
        result = add_microstructure_features(df)
        assert len(result) == 50

    def test_no_nan_in_output(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        for feat in MICROSTRUCTURE_FEATURES:
            assert not result[feat].isna().any(), f"NaN found in {feat}"


# =====================================================================
# Individual feature correctness
# =====================================================================

class TestBidAskSpread:
    """bid_ask_spread の計算テスト."""

    def test_computed_from_best_bid_ask(self) -> None:
        df = _make_1min_df(10, spread_bps=3.0)
        # Remove any existing bid_ask_spread to force computation
        if "bid_ask_spread" in df.columns:
            df = df.drop(columns=["bid_ask_spread"])
        result = add_microstructure_features(df)
        # spread should be ~3 bps
        spread = result["bid_ask_spread"]
        assert spread.mean() == pytest.approx(3e-4, rel=0.5)

    def test_passthrough_if_present(self) -> None:
        df = _make_1min_df(10)
        df["bid_ask_spread"] = 0.0005  # fixed value
        result = add_microstructure_features(df)
        assert result["bid_ask_spread"].iloc[0] == pytest.approx(0.0005)

    def test_spread_non_negative(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        assert (result["bid_ask_spread"] >= 0).all()


class TestDepthImbalance:
    """depth_imbalance のテスト (pass-through)."""

    def test_passthrough(self) -> None:
        df = _make_1min_df(10, depth_bias=0.3)
        result = add_microstructure_features(df)
        # depth_imbalance is pass-through — should match input
        pd.testing.assert_series_equal(
            result["depth_imbalance"],
            df["depth_imbalance"],
            check_names=False,
        )

    def test_range(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        assert result["depth_imbalance"].between(-1, 1).all()


class TestTradeFlowImbalance:
    """trade_flow_imbalance のテスト (pass-through)."""

    def test_passthrough(self) -> None:
        df = _make_1min_df(10)
        result = add_microstructure_features(df)
        pd.testing.assert_series_equal(
            result["trade_flow_imbalance"],
            df["trade_flow_imbalance"],
            check_names=False,
        )


class TestVwapDeviation:
    """vwap_deviation のテスト."""

    def test_computed_correctly(self) -> None:
        df = _make_1min_df(10)
        result = add_microstructure_features(df)
        # vwap_deviation = (close - vwap) / (close + eps)
        expected = (df["close"] - df["vwap"]) / (df["close"] + 1e-10)
        pd.testing.assert_series_equal(
            result["vwap_deviation"],
            expected,
            check_names=False,
            rtol=1e-6,
        )

    def test_zero_when_vwap_equals_close(self) -> None:
        df = _make_1min_df(10)
        df["vwap"] = df["close"]
        result = add_microstructure_features(df)
        assert result["vwap_deviation"].abs().max() < 1e-8


class TestTradeIntensity:
    """trade_intensity のテスト."""

    def test_mean_is_approximately_one(self) -> None:
        """window 後の trade_intensity 平均は ~1.0."""
        df = _make_1min_df(100)
        result = add_microstructure_features(df, window=20)
        # After warmup, intensity should hover around 1.0
        tail = result["trade_intensity"].iloc[30:]
        assert tail.mean() == pytest.approx(1.0, abs=0.3)

    def test_spike_detection(self) -> None:
        """trade_count にスパイクがあると intensity > 1."""
        df = _make_1min_df(50)
        df.loc[df.index[40], "trade_count"] = 500.0  # spike
        result = add_microstructure_features(df, window=20)
        assert result["trade_intensity"].iloc[40] > 5.0


class TestOrderFlowToxicity:
    """order_flow_toxicity (VPIN approx) のテスト."""

    def test_range_0_to_1(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        assert result["order_flow_toxicity"].between(0, 1.0 + 1e-6).all()

    def test_high_imbalance_means_high_toxicity(self) -> None:
        """buy/sell の一方に偏る → toxicity 高."""
        df = _make_1min_df(30)
        df["buy_volume"] = 2.0
        df["sell_volume"] = 0.01
        result = add_microstructure_features(df, window=5)
        # Nearly all volume is buy → toxicity close to 1
        assert result["order_flow_toxicity"].iloc[-1] > 0.8


class TestPriceImpact:
    """price_impact のテスト."""

    def test_non_negative(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        assert (result["price_impact"] >= 0).all()

    def test_higher_with_low_volume(self) -> None:
        """出来高が少ないほど price impact は大きい."""
        df_low = _make_1min_df(30)
        df_low["buy_volume"] = 0.01
        df_low["sell_volume"] = 0.01
        df_high = _make_1min_df(30)
        df_high["buy_volume"] = 10.0
        df_high["sell_volume"] = 10.0
        r_low = add_microstructure_features(df_low)
        r_high = add_microstructure_features(df_high)
        assert r_low["price_impact"].mean() > r_high["price_impact"].mean()


class TestMicroReturnVol:
    """micro_return_vol のテスト."""

    def test_non_negative(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        assert (result["micro_return_vol"] >= 0).all()

    def test_flat_price_low_vol(self) -> None:
        """価格変動なし → vol ≈ 0."""
        df = _make_1min_df(30)
        df["close"] = 10_300_000.0
        result = add_microstructure_features(df)
        assert result["micro_return_vol"].iloc[-1] == pytest.approx(0, abs=1e-10)


class TestDepthSlope:
    """bid_depth_slope / ask_depth_slope のテスト."""

    def test_non_negative(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        assert (result["bid_depth_slope"] >= 0).all()
        assert (result["ask_depth_slope"] >= 0).all()

    def test_calculated_from_vol_and_range(self) -> None:
        """bid_depth_slope = bid_vol_5 / (mid - best_bid)."""
        df = _make_1min_df(10, spread_bps=5.0)
        result = add_microstructure_features(df)
        eps = 1e-10
        expected_bid = df["bid_vol_5"] / (df["mid_price"] - df["best_bid"]).clip(lower=eps)
        pd.testing.assert_series_equal(
            result["bid_depth_slope"],
            expected_bid,
            check_names=False,
            rtol=1e-4,
        )


# =====================================================================
# MICROSTRUCTURE_FEATURES canonical list
# =====================================================================

class TestCanonicalList:
    """MICROSTRUCTURE_FEATURES リストの一貫性テスト."""

    def test_list_contains_10_features(self) -> None:
        assert len(MICROSTRUCTURE_FEATURES) == 10

    def test_list_is_unique(self) -> None:
        assert len(set(MICROSTRUCTURE_FEATURES)) == len(MICROSTRUCTURE_FEATURES)

    def test_all_generated_by_function(self) -> None:
        df = _make_1min_df(30)
        result = add_microstructure_features(df)
        for feat in MICROSTRUCTURE_FEATURES:
            assert feat in result.columns


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    """エッジケースのテスト."""

    def test_single_row(self) -> None:
        """1行でもクラッシュしない."""
        df = _make_1min_df(1)
        result = add_microstructure_features(df)
        assert len(result) == 1
        for feat in MICROSTRUCTURE_FEATURES:
            assert feat in result.columns

    def test_missing_optional_columns(self) -> None:
        """一部カラムがなくても動作する (0やNaN fill)."""
        df = _make_1min_df(10)
        # Remove vwap → vwap_deviation should be computed differently or skip
        df_no_vwap = df.drop(columns=["vwap"])
        result = add_microstructure_features(df_no_vwap)
        assert "micro_return_vol" in result.columns

    def test_zero_volume(self) -> None:
        """出来高ゼロでもクラッシュしない."""
        df = _make_1min_df(10)
        df["buy_volume"] = 0.0
        df["sell_volume"] = 0.0
        result = add_microstructure_features(df)
        assert not result["order_flow_toxicity"].isna().any()

    def test_input_not_mutated(self) -> None:
        """元の DataFrame が変更されない."""
        df = _make_1min_df(10)
        orig_cols = set(df.columns)
        add_microstructure_features(df)
        assert set(df.columns) == orig_cols

    def test_window_parameter(self) -> None:
        """window パラメータが反映される."""
        df = _make_1min_df(50)
        r5 = add_microstructure_features(df, window=5)
        r50 = add_microstructure_features(df, window=50)
        # micro_return_vol with smaller window should have more variance
        # (reacts faster to changes)
        assert r5["micro_return_vol"].std() >= r50["micro_return_vol"].std() * 0.5
