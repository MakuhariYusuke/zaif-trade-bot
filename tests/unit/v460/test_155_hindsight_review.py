"""155# §9 レビュー対応テスト — price=0 補間, 待機時間帯, regime×side, trending sell 抑制."""

from __future__ import annotations

import pytest


# ======================================================================
# hindsight_filter: price=0 補間 (§9.4 #1)
# ======================================================================


class TestPriceZeroInterpolation:
    """order_price=0 のレコードが補間参照価格で分析されること."""

    def _make_timeline(self):
        from scripts.v460.analysis.hindsight_filter import PricePoint
        return [
            PricePoint(1000.0, 10_000_000.0),
            PricePoint(1060.0, 10_001_000.0),
            PricePoint(1120.0, 10_002_000.0),
            PricePoint(1180.0, 10_003_000.0),
        ]

    def test_price_zero_record_included(self) -> None:
        """order_price=0 のレコードが結果に含まれること."""
        from scripts.v460.analysis.hindsight_filter import _analyze_records

        records = [
            {
                "timestamp": 1030,
                "order_price": 0,
                "side": "buy",
                "filled": False,
                "cancel_reason": "balance_forced_skip",
                "cycle_id": "test_1",
            },
            {
                "timestamp": 1000,
                "order_price": 10_000_000,
                "side": "buy",
                "filled": True,
                "cancel_reason": "",
                "cycle_id": "test_2",
                "post_fill_30s_pnl": -0.5,
            },
        ]
        timeline = self._make_timeline()
        results = _analyze_records(records, timeline)

        # price=0 は補間で復元されるので 2 件とも含まれる
        assert len(results) == 2
        interp_rec = [r for r in results if r.interpolated_ref]
        assert len(interp_rec) == 1
        assert interp_rec[0].order_price > 0
        assert interp_rec[0].cancel_reason == "balance_forced_skip"

    def test_price_zero_no_timeline_coverage(self) -> None:
        """タイムライン外の price=0 は分析不能として除外."""
        from scripts.v460.analysis.hindsight_filter import _analyze_records, PricePoint

        records = [
            {
                "timestamp": 5000,  # timeline 外 (300s+ distant)
                "order_price": 0,
                "side": "sell",
                "filled": False,
                "cancel_reason": "orderbook_error",
                "cycle_id": "test_far",
            },
        ]
        # timeline は 1000-1180 — 5000 は 3820s 離れている
        timeline = self._make_timeline()
        results = _analyze_records(records, timeline)
        assert len(results) == 0

    def test_interpolated_hindsight_pnl(self) -> None:
        """補間参照価格から hindsight PnL が計算されること."""
        from scripts.v460.analysis.hindsight_filter import _analyze_records

        records = [
            {
                "timestamp": 1030,  # → interp ≈ 10_000_500
                "order_price": 0,
                "side": "buy",
                "filled": False,
                "cancel_reason": "balance_forced_skip",
                "cycle_id": "test_pnl",
            },
        ]
        timeline = self._make_timeline()
        results = _analyze_records(records, timeline)
        assert len(results) == 1
        r = results[0]
        assert r.interpolated_ref is True
        # 30s 後 = ts 1060 → price 10_001_000
        # buy PnL = (10_001_000 - ~10_000_500) / ~10_000_500 * 10000 ≈ +0.5 bps
        if r.hindsight_pnl_30s is not None:
            assert r.hindsight_pnl_30s > 0


# ======================================================================
# hindsight_filter: 待機時間帯分析 (§9.2 #3)
# ======================================================================


class TestWaitBandAnalysis:
    """_analyze_wait_bands のバンド分割が正しいこと."""

    def _make_results(self):
        from scripts.v460.analysis.hindsight_filter import HindsightResult
        return [
            HindsightResult(
                cycle_id="w1", timestamp=1000, side="buy", order_price=10_000_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.5,
                hindsight_pnl_30s=0.5, hindsight_pnl_60s=1.0, hindsight_pnl_120s=1.5,
                reverse_pnl_30s=-0.5, skip_gate_score=None, skip_gate_as_prob=None,
                regime="ranging", queue_wait_sec=3.0,
            ),
            HindsightResult(
                cycle_id="w2", timestamp=1120, side="sell", order_price=10_001_000,
                cancel_reason="", filled=True, actual_pnl_30s=-0.8,
                hindsight_pnl_30s=-0.8, hindsight_pnl_60s=-0.5, hindsight_pnl_120s=0.2,
                reverse_pnl_30s=0.8, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending", queue_wait_sec=20.0,
            ),
            HindsightResult(
                cycle_id="w3", timestamp=1240, side="buy", order_price=10_002_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.3,
                hindsight_pnl_30s=0.3, hindsight_pnl_60s=0.6, hindsight_pnl_120s=0.9,
                reverse_pnl_30s=-0.3, skip_gate_score=None, skip_gate_as_prob=None,
                regime="ranging", queue_wait_sec=8.0,
            ),
        ]

    def test_wait_bands_populated(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_wait_bands

        results = self._make_results()
        bands = _analyze_wait_bands(results)

        assert "0-5s" in bands
        assert bands["0-5s"]["count"] == 1  # w1 (3s)
        assert bands["5-15s"]["count"] == 1  # w3 (8s)
        assert bands["15-30s"]["count"] == 1  # w2 (20s)

    def test_wait_bands_avg_pnl(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_wait_bands

        results = self._make_results()
        bands = _analyze_wait_bands(results)
        assert bands["15-30s"]["avg_pnl_30s"] == pytest.approx(-0.8, abs=0.01)


# ======================================================================
# hindsight_filter: regime×side クロス分析 (§9.2 #4)
# ======================================================================


class TestRegimeSideAnalysis:
    """_analyze_regime_side のクロス集計が正しいこと."""

    def test_regime_side_cross(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_regime_side, HindsightResult

        results = [
            HindsightResult(
                cycle_id="rs1", timestamp=1000, side="buy", order_price=10_000_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.5,
                hindsight_pnl_30s=0.5, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.5, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending",
            ),
            HindsightResult(
                cycle_id="rs2", timestamp=1120, side="sell", order_price=10_001_000,
                cancel_reason="", filled=True, actual_pnl_30s=-0.7,
                hindsight_pnl_30s=-0.7, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=0.7, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending",
            ),
            HindsightResult(
                cycle_id="rs3", timestamp=1240, side="buy", order_price=10_002_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.2,
                hindsight_pnl_30s=0.2, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.2, skip_gate_score=None, skip_gate_as_prob=None,
                regime="ranging",
            ),
        ]
        rs = _analyze_regime_side(results)

        assert "trending_buy" in rs
        assert "trending_sell" in rs
        assert "ranging_buy" in rs
        assert rs["trending_buy"]["count"] == 1
        assert rs["trending_sell"]["avg_pnl_30s"] == pytest.approx(-0.7, abs=0.01)


# ======================================================================
# hindsight_filter: 補間統計 (§9.4 #1)
# ======================================================================


class TestInterpolatedStats:
    """_analyze_interpolated_stats の統計が正しいこと."""

    def test_interpolated_split(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _analyze_interpolated_stats, HindsightResult

        results = [
            HindsightResult(
                cycle_id="i1", timestamp=1000, side="buy", order_price=10_000_000,
                cancel_reason="", filled=True, actual_pnl_30s=0.5,
                hindsight_pnl_30s=0.5, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.5, skip_gate_score=None, skip_gate_as_prob=None,
                regime=None, interpolated_ref=False,
            ),
            HindsightResult(
                cycle_id="i2", timestamp=1060, side="sell", order_price=10_000_500,
                cancel_reason="balance_forced_skip", filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=0.3, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=-0.3, skip_gate_score=None, skip_gate_as_prob=None,
                regime=None, interpolated_ref=True,
            ),
        ]
        stats = _analyze_interpolated_stats(results)
        assert stats["interpolated"]["count"] == 1
        assert stats["original_price"]["count"] == 1
        assert stats["interpolated"]["avg_hindsight_30s"] == pytest.approx(0.3, abs=0.01)


# ======================================================================
# hindsight_filter: H8 regime_guard カテゴリ (trending_sell_skip)
# ======================================================================


class TestCategorization:
    """_categorize でレジームガード系が H8 に分類されること."""

    def test_trending_sell_skip_in_h8(self) -> None:
        from scripts.v460.analysis.hindsight_filter import _categorize, HindsightResult

        results = [
            HindsightResult(
                cycle_id="c1", timestamp=1000, side="sell", order_price=10_000_000,
                cancel_reason="trending_sell_skip", filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=None, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=None, skip_gate_score=None, skip_gate_as_prob=None,
                regime="trending",
            ),
            HindsightResult(
                cycle_id="c2", timestamp=1060, side="buy", order_price=10_000_000,
                cancel_reason="unknown_regime_buy_skip", filled=False, actual_pnl_30s=None,
                hindsight_pnl_30s=None, hindsight_pnl_60s=None, hindsight_pnl_120s=None,
                reverse_pnl_30s=None, skip_gate_score=None, skip_gate_as_prob=None,
                regime="unknown",
            ),
        ]
        cats = _categorize(results)
        assert "H8_regime_guard" in cats
        assert len(cats["H8_regime_guard"]) == 2


# ======================================================================
# cancel_reasons: TRENDING_SELL_SKIP 定数
# ======================================================================


class TestTrendingSellSkipConstant:
    """TRENDING_SELL_SKIP が cancel_reasons に存在し AUDIT set に含まれること."""

    def test_constant_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert hasattr(CR, "TRENDING_SELL_SKIP")
        assert CR.TRENDING_SELL_SKIP == "trending_sell_skip"

    def test_in_audit_set(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.TRENDING_SELL_SKIP in CR.AUDIT_CANCEL_REASONS


# ======================================================================
# fill_config: skip_sell_trending フィールド
# ======================================================================


class TestSkipSellTrendingConfig:
    """FillTestConfig に skip_sell_trending が存在すること."""

    def test_default_false(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.skip_sell_trending is False

    def test_set_true(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(skip_sell_trending=True)
        assert cfg.skip_sell_trending is True


# ======================================================================
# 155# §10 残課題: balance_forced_consecutive フィールド (§9.4 #2)
# ======================================================================


class TestBalanceForcedConsecutiveField:
    """FillRecord に balance_forced_consecutive が存在すること."""

    _BASE = {"cycle_id": "t1", "timestamp": 1.0, "side": "buy", "order_price": 100.0, "order_quantity": 0.001}

    def test_field_default_none(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(**self._BASE)
        assert rec.balance_forced_consecutive is None

    def test_field_set(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(**self._BASE, balance_forced_consecutive=5)
        assert rec.balance_forced_consecutive == 5

    def test_to_dict_contains_field(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        rec = FillRecord(**self._BASE, balance_forced_consecutive=3)
        d = rec.to_dict()
        assert "balance_forced_consecutive" in d
        assert d["balance_forced_consecutive"] == 3


# ======================================================================
# 155# S-3: order_timeout_sec_sell 設定
# ======================================================================


class TestSellTimeoutConfig:
    """FillTestConfig に order_timeout_sec_sell が存在すること."""

    def test_default_none(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.order_timeout_sec_sell is None

    def test_set_value(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig(order_timeout_sec_sell=75.0)
        assert cfg.order_timeout_sec_sell == 75.0

    def test_from_yaml_loads(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_cfg = {"order_timeout_sec_sell": 72.0}
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.order_timeout_sec_sell == 72.0
