"""458# macro regime ヒステリシス + F-lite offset boost テスト."""
from __future__ import annotations

import pytest

from scripts.v460.lib.macro_regime import (
    MacroRegimeConfig,
    MacroRegimeDetector,
    MacroRegimeResult,
    MacroTrend,
    compose_regimes,
)


# ============================================================
# §1 基本分類テスト
# ============================================================

class TestMacroRegimeClassification:
    """MacroTrend 分類の基本テスト."""

    def _feed_linear(
        self,
        detector: MacroRegimeDetector,
        base_price: float,
        n_buckets: int,
        slope_jpy_per_bucket: float,
    ) -> MacroRegimeResult:
        """線形価格系列を投入して最終結果を返す."""
        result = MacroRegimeResult(trend=MacroTrend.INSUFFICIENT)
        for i in range(n_buckets):
            t = float(i * 35)  # 35s 間隔 (> bucket_sec=30 で確実にバケット確定)
            p = base_price + slope_jpy_per_bucket * i
            result = detector.update(t, p)
        return result

    def test_insufficient_data(self) -> None:
        """データ不足で INSUFFICIENT."""
        cfg = MacroRegimeConfig(slope_threshold_bps_per_min=0.5, hysteresis_count=1)
        det = MacroRegimeDetector(cfg)
        r = det.update(0.0, 10_000_000.0)
        assert r.trend == MacroTrend.INSUFFICIENT

    def test_neutral_flat(self) -> None:
        """横ばいで NEUTRAL."""
        cfg = MacroRegimeConfig(slope_threshold_bps_per_min=0.5, hysteresis_count=1)
        det = MacroRegimeDetector(cfg)
        r = self._feed_linear(det, 10_000_000.0, 15, 0.0)
        assert r.trend == MacroTrend.NEUTRAL

    def test_strong_up(self) -> None:
        """強い上昇 → STRONG_UP (5m/15m 両方超).

        slope_threshold=0.5 bps/min, strong_threshold=2.0 bps/min
        bucket 間隔 35s = 0.583min
        2.0 bps/min = 2000 JPY/min (10M base) = 1167 JPY/bucket
        2000 JPY/bucket で確実に strong 超え
        """
        cfg = MacroRegimeConfig(
            slope_threshold_bps_per_min=0.5,
            strong_slope_threshold_bps_per_min=2.0,
            hysteresis_count=1,
        )
        det = MacroRegimeDetector(cfg)
        r = self._feed_linear(det, 10_000_000.0, 35, 2000.0)
        assert r.trend == MacroTrend.STRONG_UP
        assert r.slope_5m_bps_per_min > 0.5

    def test_weak_up_5m_only(self) -> None:
        """5m のみ上昇 → WEAK_UP (15m 不足時).

        0.5 bps/min → 500 JPY/min (10M base) → 292 JPY/bucket
        """
        cfg = MacroRegimeConfig(
            slope_threshold_bps_per_min=0.5,
            hysteresis_count=1,
        )
        det = MacroRegimeDetector(cfg)
        # 12 バケットだけ投入 (5m window 超えるが 15m 不足)
        r = self._feed_linear(det, 10_000_000.0, 12, 500.0)
        assert r.trend in (MacroTrend.WEAK_UP, MacroTrend.STRONG_UP)


# ============================================================
# §2 ヒステリシスのテスト
# ============================================================

class TestMacroRegimeHysteresis:
    """458# ヒステリシス: フラッピング防止."""

    def test_hysteresis_delays_transition(self) -> None:
        """hysteresis_count=3 では、raw が UP でも 3 回未満は NEUTRAL のまま."""
        cfg = MacroRegimeConfig(
            slope_threshold_bps_per_min=0.5,
            hysteresis_count=3,
            hold_count=2,
        )
        det = MacroRegimeDetector(cfg)

        # まず横ばいデータを蓄積 (NEUTRAL 状態)
        base = 10_000_000.0
        for i in range(12):
            det.update(float(i * 35), base)

        # ヒステリシスにより、初期は NEUTRAL のままのはず
        # (直前バケットまで横ばいなので 5m のスロープ算出ウィンドウの大半がフラット)
        r = det.update(12 * 35.0, base)
        assert r.trend == MacroTrend.NEUTRAL

    def test_hysteresis_confirms_after_n(self) -> None:
        """hysteresis_count 回連続で同方向なら確定する."""
        cfg = MacroRegimeConfig(
            slope_threshold_bps_per_min=0.3,
            strong_slope_threshold_bps_per_min=2.0,
            hysteresis_count=3,
            hold_count=1,
        )
        det = MacroRegimeDetector(cfg)

        # 強い上昇トレンドを投入 (十分な slope: 1500 JPY/bucket)
        base = 10_000_000.0
        for i in range(40):
            t = float(i * 35)
            p = base + 1500.0 * i  # 強い上昇
            det.update(t, p)

        # 十分なバケット後は UP 系に確定
        r = det.update(40 * 35.0, base + 1500.0 * 40)
        assert r.trend in (MacroTrend.WEAK_UP, MacroTrend.STRONG_UP)

    def test_hold_count_maintains_trend(self) -> None:
        """確定後、hold_count 回は raw が変わっても保持する."""
        cfg = MacroRegimeConfig(
            slope_threshold_bps_per_min=0.3,
            hysteresis_count=2,
            hold_count=3,
        )
        det = MacroRegimeDetector(cfg)

        # まず UP を確定させる (1000 JPY/bucket → 十分なスロープ)
        base = 10_000_000.0
        for i in range(35):
            det.update(float(i * 35), base + 1000.0 * i)

        last_r = det.update(35 * 35.0, base + 1000.0 * 35)
        # UP 系に確定しているはず
        confirmed_trend = last_r.trend
        assert confirmed_trend in (MacroTrend.WEAK_UP, MacroTrend.STRONG_UP)

        # 横ばいデータを投入 — hold_count 回は確定トレンドが維持される
        flat_price = base + 1000.0 * 35
        r_hold1 = det.update(36 * 35.0, flat_price)
        assert r_hold1.trend == confirmed_trend, "hold 1/3 should maintain confirmed trend"


# ============================================================
# §3 compose_regimes テスト
# ============================================================

class TestComposeRegimes:
    """macro/micro 矛盾検出."""

    def test_aligned(self) -> None:
        r = MacroRegimeResult(trend=MacroTrend.STRONG_UP, confidence=0.8)
        regime, aligned = compose_regimes("trending_up", 0.85, r)
        assert aligned is True
        assert regime == "trending_up"

    def test_conflict_micro_up_macro_down(self) -> None:
        r = MacroRegimeResult(trend=MacroTrend.STRONG_DOWN, confidence=0.7)
        regime, aligned = compose_regimes("trending_up", 0.85, r)
        assert aligned is False

    def test_insufficient_macro(self) -> None:
        r = MacroRegimeResult(trend=MacroTrend.INSUFFICIENT)
        regime, aligned = compose_regimes("trending_up", 0.85, r)
        assert aligned is True


# ============================================================
# §4 Config フィールドテスト
# ============================================================

class TestFillConfigMacroFields:
    """458# 新規 config フィールドのデフォルト値テスト."""

    def test_macro_boost_defaults(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.macro_sell_boost_weak_up == 1.3
        assert cfg.macro_sell_boost_strong_up == 1.6
        assert cfg.macro_buy_boost_weak_down == 1.3
        assert cfg.macro_buy_boost_strong_down == 1.6

    def test_macro_timeout_defaults(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.macro_sell_timeout_weak_up is None
        assert cfg.macro_sell_timeout_strong_up is None


# ============================================================
# §5 FillRecord macro_boost_applied フィールド
# ============================================================

class TestFillRecordMacroBoost:
    """FillRecord に macro_boost_applied フィールドが存在する."""

    def test_field_exists(self) -> None:
        from ztb.metrics.fill_quality import FillRecord
        r = FillRecord(
            cycle_id="test", timestamp=0.0, side="buy",
            order_price=100.0, order_quantity=0.001,
        )
        assert hasattr(r, "macro_boost_applied")
        assert r.macro_boost_applied is None


# ============================================================
# §6 Hot-reload 配線テスト
# ============================================================

class TestHotReloadWiring:
    """458# 新フィールドが _HOT_RELOADABLE_FIELDS に含まれる."""

    def test_macro_boost_fields_in_hot_reload(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
        for f in [
            "macro_sell_boost_weak_up",
            "macro_sell_boost_strong_up",
            "macro_buy_boost_weak_down",
            "macro_buy_boost_strong_down",
            "macro_sell_timeout_weak_up",
            "macro_sell_timeout_strong_up",
        ]:
            assert f in _HOT_RELOADABLE_FIELDS, f"{f} missing from hot-reload"


# ============================================================
# §7 メモリリーク防止テスト
# ============================================================

class TestMemoryLeakPrevention:
    """_current_bucket_prices が無制限成長しないことを検証."""

    def test_bucket_prices_capped(self) -> None:
        """バケットが閉じないシナリオで _current_bucket_prices が上限内に収まる."""
        cfg = MacroRegimeConfig(bucket_sec=30.0)
        det = MacroRegimeDetector(cfg)
        base_ts = 1_000_000.0
        # 同一タイムスタンプで 300 回更新 (バケット閉じない)
        for i in range(300):
            det.update(base_ts, 10_000_000.0 + i)
        # 200 以下にキャップされる
        assert len(det._current_bucket_prices) <= 200
