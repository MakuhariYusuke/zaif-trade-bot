"""366# M5: Volume-Sync VPIN テスト.

compute_vpin_volume_sync() の単体テスト + feature_enricher 統合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.v460.lib.vpin_volume_sync import (
    NEUTRAL_VPIN,
    compute_vpin_volume_sync,
)


# =====================================================================
# Helpers
# =====================================================================


def _make_cumulative_arrays(
    amounts: list[float],
    buy_flags: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """amounts と buy_flags から cumulative 配列を生成."""
    amounts_arr = np.array(amounts, dtype=np.float64)
    cum_total = np.empty(len(amounts) + 1, dtype=np.float64)
    cum_total[0] = 0.0
    np.cumsum(amounts_arr, out=cum_total[1:])

    buy_amounts = np.where(buy_flags, amounts_arr, 0.0)
    cum_buy = np.empty(len(amounts) + 1, dtype=np.float64)
    cum_buy[0] = 0.0
    np.cumsum(buy_amounts, out=cum_buy[1:])

    return cum_total, cum_buy


# =====================================================================
# TestComputeVpinVolumeSync
# =====================================================================


class TestComputeVpinVolumeSync:
    """compute_vpin_volume_sync() の単体テスト."""

    def test_neutral_on_empty(self) -> None:
        """空の配列でニュートラル値を返す."""
        cum_total = np.array([0.0])
        cum_buy = np.array([0.0])
        result = compute_vpin_volume_sync(cum_total, cum_buy, end_index=0)
        assert result == NEUTRAL_VPIN

    def test_neutral_on_insufficient_volume(self) -> None:
        """バケット1つ分の出来高に満たない場合、ニュートラル値を返す."""
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.01, 0.01], [True, False]
        )
        # bucket_size=0.05 > total 0.02 → 不足
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=2, bucket_size=0.05, n_buckets=1
        )
        assert result == NEUTRAL_VPIN

    def test_single_bucket_all_buy(self) -> None:
        """1バケット、全 buy → VPIN=1.0."""
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.1, 0.1, 0.1], [True, True, True]
        )
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=3, bucket_size=0.3, n_buckets=1
        )
        assert result == pytest.approx(1.0)

    def test_single_bucket_balanced(self) -> None:
        """1バケット、buy=sell → VPIN=0.0."""
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.1, 0.1], [True, False]
        )
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=2, bucket_size=0.2, n_buckets=1
        )
        assert result == pytest.approx(0.0)

    def test_two_buckets_mixed(self) -> None:
        """2バケット: bucket1=balanced(0.0), bucket2=all_buy(1.0) → mean=0.5."""
        # Bucket 1: [buy 0.1, sell 0.1] → VPIN=0.0
        # Bucket 2: [buy 0.1, buy 0.1] → VPIN=1.0
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.1, 0.1, 0.1, 0.1], [True, False, True, True]
        )
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=4, bucket_size=0.2, n_buckets=2
        )
        assert result == pytest.approx(0.5)

    def test_n_buckets_caps_to_available(self) -> None:
        """n_buckets > 利用可能バケット数の場合、利用可能分のみ使用."""
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.1, 0.1], [True, True]
        )
        # 1 bucket available, requesting 50
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=2, bucket_size=0.2, n_buckets=50
        )
        assert result == pytest.approx(1.0)

    def test_end_index_clamped(self) -> None:
        """end_index が配列長を超えても安全."""
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.1, 0.1], [True, False]
        )
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=999, bucket_size=0.2, n_buckets=1
        )
        assert result == pytest.approx(0.0)

    def test_invalid_params_return_neutral(self) -> None:
        """不正パラメータでニュートラル値."""
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.1], [True]
        )
        assert compute_vpin_volume_sync(cum_total, cum_buy, 1, bucket_size=-1) == NEUTRAL_VPIN
        assert compute_vpin_volume_sync(cum_total, cum_buy, 1, n_buckets=0) == NEUTRAL_VPIN
        assert compute_vpin_volume_sync(cum_total, cum_buy, 0) == NEUTRAL_VPIN

    def test_large_dataset_accuracy(self) -> None:
        """1000件の約定で time-based VPIN と比較可能な値を返す."""
        rng = np.random.default_rng(42)
        n = 1000
        amounts = rng.uniform(0.001, 0.05, size=n)
        buy_flags = rng.random(n) > 0.5  # roughly balanced

        cum_total = np.empty(n + 1, dtype=np.float64)
        cum_total[0] = 0.0
        np.cumsum(amounts, out=cum_total[1:])

        buy_amounts = np.where(buy_flags, amounts, 0.0)
        cum_buy = np.empty(n + 1, dtype=np.float64)
        cum_buy[0] = 0.0
        np.cumsum(buy_amounts, out=cum_buy[1:])

        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=n, bucket_size=0.5, n_buckets=20
        )
        # balanced random → VPIN should be low (0.0-0.5 range)
        assert 0.0 <= result <= 1.0
        assert result < 0.5  # roughly balanced trades

    def test_toxic_flow_detected(self) -> None:
        """片方向フロー → VPIN 高値."""
        n = 100
        amounts = np.full(n, 0.01)
        buy_flags = np.ones(n, dtype=bool)  # all buy

        cum_total = np.empty(n + 1, dtype=np.float64)
        cum_total[0] = 0.0
        np.cumsum(amounts, out=cum_total[1:])

        cum_buy = cum_total.copy()  # all buy volume

        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=n, bucket_size=0.1, n_buckets=10
        )
        assert result == pytest.approx(1.0)

    def test_partial_bucket_not_counted(self) -> None:
        """不完全バケットはカウントしない (先頭の余り分を除外)."""
        # 7 trades × 0.05 = 0.35 total, bucket_size=0.15
        # n_full_buckets = int(0.35/0.15) = 2
        # start_vol = 0.35 - 2*0.15 = 0.05 → 先頭 trade0 は除外
        # bucket1: trades 1,2,3 (all buy) → VPIN=1.0
        # bucket2: trades 4,5,6 (all sell) → VPIN=1.0
        cum_total, cum_buy = _make_cumulative_arrays(
            [0.05] * 7,
            [False, True, True, True, False, False, False],
        )
        result = compute_vpin_volume_sync(
            cum_total, cum_buy, end_index=7, bucket_size=0.15, n_buckets=10
        )
        assert result == pytest.approx(1.0)


# =====================================================================
# TestFeatureEnricherVpinVolSync
# =====================================================================


class TestFeatureEnricherVpinVolSync:
    """feature_enricher 統合テスト: vpin_vol_sync がデフォルト dict に含まれる."""

    def test_default_features_include_vpin_vol_sync(self) -> None:
        """_default_multi_timeframe_trade_features に vpin_vol_sync が含まれる."""
        from scripts.v460.ml.feature_enricher import (
            _default_multi_timeframe_trade_features,
        )
        defaults = _default_multi_timeframe_trade_features()
        assert "vpin_vol_sync" in defaults
        assert defaults["vpin_vol_sync"] == NEUTRAL_VPIN

    def test_bundle_includes_vpin_vol_sync_when_enabled(self) -> None:
        """vpin_vol_sync_bucket > 0 で multi dict に vpin_vol_sync が出力される."""
        from scripts.v460.ml.feature_enricher import (
            _TradeFeatureContext,
            _compute_trade_feature_bundle,
        )
        # All-buy traffic
        n = 20
        amounts = np.full(n, 0.05)
        timestamps = np.arange(n, dtype=np.float64)
        cum_total = np.empty(n + 1, dtype=np.float64)
        cum_total[0] = 0.0
        np.cumsum(amounts, out=cum_total[1:])
        cum_buy = cum_total.copy()

        context = _TradeFeatureContext(
            timestamps=timestamps,
            prices=np.full(n, 100.0),
            cumulative_total_volume=cum_total,
            cumulative_buy_volume=cum_buy,
        )
        _, multi = _compute_trade_feature_bundle(
            ts=19.5,
            context=context,
            multi_windows=(),
            vpin_vol_sync_bucket=0.1,
            vpin_vol_sync_n_buckets=5,
        )
        assert "vpin_vol_sync" in multi
        assert multi["vpin_vol_sync"] == pytest.approx(1.0)

    def test_bundle_vpin_vol_sync_disabled_by_default(self) -> None:
        """vpin_vol_sync_bucket=0.0 (default) では volume-sync 計算されない."""
        from scripts.v460.ml.feature_enricher import (
            _TradeFeatureContext,
            _compute_trade_feature_bundle,
        )
        n = 10
        timestamps = np.arange(n, dtype=np.float64)
        cum_total = np.arange(n + 1, dtype=np.float64) * 0.01
        cum_buy = cum_total * 0.5

        context = _TradeFeatureContext(
            timestamps=timestamps,
            prices=np.full(n, 100.0),
            cumulative_total_volume=cum_total,
            cumulative_buy_volume=cum_buy,
        )
        _, multi = _compute_trade_feature_bundle(
            ts=9.5,
            context=context,
            multi_windows=(),
        )
        # Default (not computed) should be 0.5
        assert multi["vpin_vol_sync"] == NEUTRAL_VPIN


# =====================================================================
# セルフレビュー TG8: NaN/Inf テスト
# =====================================================================


class TestReviewGapsVpin:
    """セルフレビューで特定されたテストギャップの補完."""

    def test_tg8_nan_in_cumulative_total(self) -> None:
        """TG8: NaN を含む累積配列で NEUTRAL_VPIN を返す."""
        cum_total = np.array([0.0, 0.1, float("nan"), 0.3])
        cum_buy = np.array([0.0, 0.05, 0.1, 0.15])
        result = compute_vpin_volume_sync(cum_total, cum_buy, end_index=3, bucket_size=0.1, n_buckets=2)
        assert result == NEUTRAL_VPIN

    def test_tg8_inf_in_cumulative_buy(self) -> None:
        """TG8: Inf を含む累積配列で NEUTRAL_VPIN を返す."""
        cum_total = np.array([0.0, 0.1, 0.2, 0.3])
        cum_buy = np.array([0.0, 0.05, float("inf"), 0.15])
        result = compute_vpin_volume_sync(cum_total, cum_buy, end_index=3, bucket_size=0.1, n_buckets=2)
        assert result == NEUTRAL_VPIN

    def test_tg8_negative_inf_in_total(self) -> None:
        """TG8: -Inf で NEUTRAL_VPIN."""
        cum_total = np.array([0.0, float("-inf"), 0.2])
        cum_buy = np.array([0.0, 0.1, 0.15])
        result = compute_vpin_volume_sync(cum_total, cum_buy, end_index=2, bucket_size=0.1, n_buckets=1)
        assert result == NEUTRAL_VPIN
