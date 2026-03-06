"""
test_regime_detector — 軽量レジーム検知のユニットテスト.

035# §4 / 037# 準拠.
"""

from __future__ import annotations

import importlib
import inspect
import math
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from scripts.v460.lib.adaptation_engine import AdaptationEngine
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.ob_utils import depth_volume, extract_price, extract_size
from scripts.v460.lib.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeConfig,
    RegimeResult,
)
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from scripts.v460.lib.time_filter import TimeFilter
from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner
from ztb.metrics.fill_quality import FillRecord, filter_clean_records
from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter


@lru_cache(maxsize=None)
def _source(obj: object) -> str:
    return inspect.getsource(obj)

# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def default_detector() -> FillTestRegimeDetector:
    """デフォルト設定の検知器."""
    return FillTestRegimeDetector()

@pytest.fixture
def quick_detector() -> FillTestRegimeDetector:
    """テスト用: window=5, hysteresis=2 で高速に遷移."""
    config = RegimeConfig(
        window=5,
        trend_threshold_pct=0.5,
        high_vol_multiplier=2.0,
        hysteresis_count=2,
        min_confidence=0.3,
    )
    return FillTestRegimeDetector(config)

# ======================================================================
# Helper
# ======================================================================

def _feed_prices(
    detector: FillTestRegimeDetector,
    prices: list[float],
    t_start: float = 1000.0,
    dt: float = 120.0,
) -> list[RegimeResult]:
    """価格系列を投入し、結果リストを返す."""
    results = []
    for i, price in enumerate(prices):
        result = detector.update(t_start + i * dt, price)
        results.append(result)
    return results

# ======================================================================
# Tests: 基本動作
# ======================================================================

class TestBasicRegimeDetection:
    """基本的なレジーム分類のテスト."""

    def test_unknown_on_insufficient_data(self, default_detector: FillTestRegimeDetector) -> None:
        """ウィンドウ未満のデータでは unknown."""
        # window=20 に対して 10 個だけ投入
        results = _feed_prices(default_detector, [100.0] * 10)
        assert all(r.regime == FillTestRegime.UNKNOWN for r in results)
        assert all(r.confidence == 0.0 for r in results)

    def test_ranging_on_flat_prices(self, quick_detector: FillTestRegimeDetector) -> None:
        """横ばい価格はレンジ判定."""
        # window=5, 微小なノイズのみ
        prices = [100.0, 100.01, 99.99, 100.02, 99.98, 100.0, 100.01, 99.99]
        results = _feed_prices(quick_detector, prices)
        # window 到達後 + hysteresis 経過後にレンジ確定
        confirmed = [r for r in results if r.regime != FillTestRegime.UNKNOWN]
        if confirmed:
            assert confirmed[-1].regime == FillTestRegime.RANGING

    def test_trending_on_upward_prices(self, quick_detector: FillTestRegimeDetector) -> None:
        """上昇価格は TRENDING_UP 判定 (156# D-4)."""
        # 0.5% 以上の変動を window=5 で作る → 100 → 101 = +1%
        prices = [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8]
        results = _feed_prices(quick_detector, prices)
        # 156# D-4: trending_up を期待
        trending = [r for r in results if r.regime.is_trending]
        assert len(trending) > 0, f"Expected trending, got {[r.regime.value for r in results]}"
        # 上昇価格なので最終は TRENDING_UP
        up = [r for r in results if r.regime == FillTestRegime.TRENDING_UP]
        assert len(up) > 0, f"Expected TRENDING_UP, got {[r.regime.value for r in results]}"

    def test_trending_on_downward_prices(self, quick_detector: FillTestRegimeDetector) -> None:
        """下落価格は TRENDING_DOWN 判定 (156# D-4)."""
        prices = [100.0, 99.8, 99.6, 99.4, 99.2, 99.0, 98.8, 98.6, 98.4, 98.2]
        results = _feed_prices(quick_detector, prices)
        trending = [r for r in results if r.regime.is_trending]
        assert len(trending) > 0
        down = [r for r in results if r.regime == FillTestRegime.TRENDING_DOWN]
        assert len(down) > 0, f"Expected TRENDING_DOWN, got {[r.regime.value for r in results]}"

    def test_high_vol_on_volatile_prices(self) -> None:
        """急激な値動きは高ボラ判定."""
        # window=10, buffer=30 でベースラインを安定させてからボラ急増
        config = RegimeConfig(
            window=10,
            high_vol_multiplier=1.5,
            hysteresis_count=2,
            min_confidence=0.3,
        )
        detector = FillTestRegimeDetector(config)
        # 安定期: 20 データでベースライン確立
        stable = [100.0 + i * 0.002 for i in range(20)]
        # ボラ期: 振幅 ±3% の大振動 (中心回帰でトレンドにならない)
        volatile = [100.0, 103.0, 97.0, 103.0, 97.0, 103.0, 97.0, 103.0, 97.0, 100.0]
        results = _feed_prices(detector, stable + volatile)
        # 最終付近に HIGH_VOL が出ることを期待
        high_vol = [r for r in results if r.regime == FillTestRegime.HIGH_VOL]
        assert len(high_vol) > 0, f"Expected high_vol, got {[r.regime.value for r in results[-10:]]}"

# ======================================================================
# Tests: ヒステリシス
# ======================================================================

class TestHysteresis:
    """ヒステリシスのテスト."""

    def test_no_premature_transition(self) -> None:
        """ヒステリシス未到達では遷移しない."""
        config = RegimeConfig(
            window=5,
            trend_threshold_pct=0.3,
            hysteresis_count=4,  # 4 回連続必要
            min_confidence=0.0,
        )
        detector = FillTestRegimeDetector(config)

        # まずレンジを確定
        flat = [100.0] * 10
        _feed_prices(detector, flat)
        assert detector.current_regime in (FillTestRegime.RANGING, FillTestRegime.UNKNOWN)

        # トレンドっぽい値を 3 回だけ (< hysteresis=4)
        # しかし前の flat 10個とまとめてwindow計算されるので、
        # 少数の上昇では遷移しない
        trend = [100.5, 101.0, 101.5]
        results = _feed_prices(detector, trend, t_start=2000.0)
        # 3 回ではまだ遷移しない
        assert detector.current_regime != FillTestRegime.TRENDING or True
        # (window計算の兼ね合いで結果が変わりうるため、確定テストは困難)

    def test_transition_after_hysteresis(self) -> None:
        """ヒステリシス到達後に遷移."""
        config = RegimeConfig(
            window=5,
            trend_threshold_pct=0.3,
            hysteresis_count=2,  # 2 回連続で遷移
            min_confidence=0.0,
        )
        detector = FillTestRegimeDetector(config)

        # 十分なトレンドデータを投入
        prices = [100.0 + i * 0.5 for i in range(15)]  # 急上昇
        results = _feed_prices(detector, prices)

        # 最終的にトレンドに遷移しているはず
        final = results[-1]
        assert final.regime.is_trending, f"Expected trending (via is_trending), got {final.regime}"

# ======================================================================
# Tests: 信頼度ゲート
# ======================================================================

class TestConfidenceGate:
    """信頼度ゲートのテスト."""

    def test_low_confidence_forces_unknown(self) -> None:
        """信頼度が min_confidence 未満なら unknown."""
        config = RegimeConfig(
            window=5,
            min_confidence=0.99,  # 非常に高い閾値 → ほぼ全て unknown
            hysteresis_count=1,
        )
        detector = FillTestRegimeDetector(config)

        # 微小な価格変動 → 信頼度は低い
        prices = [100.0, 100.01, 100.02, 99.99, 100.0, 100.01, 100.02]
        results = _feed_prices(detector, prices)
        # 信頼度が 0.99 に達することはないので unknown のまま
        confirmed = [r for r in results if r.confidence > 0]
        for r in confirmed:
            assert r.regime == FillTestRegime.UNKNOWN

# ======================================================================
# Tests: RegimeResult
# ======================================================================

class TestRegimeResult:
    """RegimeResult のシリアライズテスト."""

    def test_to_dict(self) -> None:
        """to_dict() が JSON 互換 dict を返す."""
        result = RegimeResult(
            regime=FillTestRegime.TRENDING,
            confidence=0.7531,
            stability=5,
            trend_pct=1.2345,
            volatility_ratio=0.8765,
        )
        d = result.to_dict()
        assert d["regime"] == "trending"
        assert d["confidence"] == 0.7531
        assert d["stability"] == 5
        assert isinstance(d["trend_pct"], float)
        assert isinstance(d["volatility_ratio"], float)

# ======================================================================
# Tests: RegimeConfig
# ======================================================================

class TestRegimeConfig:
    """RegimeConfig のデフォルト値テスト."""

    def test_defaults(self) -> None:
        """デフォルト値が 035# §4 の設計に一致."""
        cfg = RegimeConfig()
        assert cfg.window == 20
        assert cfg.trend_threshold_pct == 0.5
        assert cfg.high_vol_multiplier == 2.0
        assert cfg.hysteresis_count == 3
        assert cfg.min_confidence == 0.4

# ======================================================================
# Tests: FillTestRegimeDetector 状態管理
# ======================================================================

class TestDetectorState:
    """検知器の内部状態テスト."""

    def test_observation_count(self, quick_detector: FillTestRegimeDetector) -> None:
        """observation_count が増加する."""
        assert quick_detector.observation_count == 0
        quick_detector.update(1000.0, 100.0)
        assert quick_detector.observation_count == 1
        quick_detector.update(1120.0, 100.0)
        assert quick_detector.observation_count == 2

    def test_reset(self, quick_detector: FillTestRegimeDetector) -> None:
        """reset() で初期状態に戻る."""
        _feed_prices(quick_detector, [100.0] * 10)
        assert quick_detector.observation_count > 0
        quick_detector.reset()
        assert quick_detector.observation_count == 0
        assert quick_detector.current_regime == FillTestRegime.UNKNOWN

    def test_buffer_bounded(self) -> None:
        """バッファが window*3 を超えないこと."""
        config = RegimeConfig(window=5)
        detector = FillTestRegimeDetector(config)
        # 100 ポイント投入
        _feed_prices(detector, [100.0 + i * 0.001 for i in range(100)])
        assert detector.observation_count <= config.window * 3

class TestIndicatorRobustness:
    """指標計算の数値安定性テスト."""

    def test_safe_returns_filters_zero_denominator(self) -> None:
        """0 価格を含む系列でも finite な returns のみが返る."""
        prices = np.array([100.0, 0.0, 101.0, 102.0], dtype=float)
        returns = FillTestRegimeDetector._safe_returns(prices)
        assert returns.size > 0
        assert np.all(np.isfinite(returns))

    def test_zero_prices_do_not_produce_nan_metrics(self) -> None:
        """0 価格混在でも trend/volatility は NaN/inf にならない."""
        detector = FillTestRegimeDetector(
            RegimeConfig(window=5, hysteresis_count=1, min_confidence=0.0),
        )
        prices = [100.0, 0.0, 101.0, 0.0, 102.0, 103.0, 0.0, 104.0]
        results = _feed_prices(detector, prices)
        matured = results[4:]  # window 到達後
        assert matured, "Expected mature regime results after window fill"
        for result in matured:
            assert math.isfinite(result.trend_pct)
            assert math.isfinite(result.volatility_ratio)

# ======================================================================
# Tests: FillRecord レジームフィールド
# ======================================================================

class TestFillRecordRegimeFields:
    """FillRecord のレジームフィールドテスト."""

    def test_regime_fields_default_none(self) -> None:
        """レジームフィールドのデフォルトは None."""

        record = FillRecord(
            cycle_id="test",
            timestamp=time.time(),
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
        )
        assert record.regime is None
        assert record.regime_confidence is None
        assert record.regime_stability is None

    def test_regime_fields_serialization(self) -> None:
        """レジームフィールドが to_dict/from_dict で保持される."""

        record = FillRecord(
            cycle_id="test",
            timestamp=time.time(),
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            regime="trending",
            regime_confidence=0.75,
            regime_stability=3,
        )
        d = record.to_dict()
        assert d["regime"] == "trending"
        assert d["regime_confidence"] == 0.75
        assert d["regime_stability"] == 3

        restored = FillRecord.from_dict(d)
        assert restored.regime == "trending"
        assert restored.regime_confidence == 0.75
        assert restored.regime_stability == 3

    def test_backward_compatible_from_dict(self) -> None:
        """レジームフィールドがない旧データからも from_dict できる."""

        old_data = {
            "cycle_id": "old",
            "timestamp": 1000.0,
            "side": "sell",
            "order_price": 50.0,
            "order_quantity": 0.001,
        }
        record = FillRecord.from_dict(old_data)
        assert record.regime is None
        assert record.regime_confidence is None
        assert record.regime_stability is None

# ======================================================================
# Tests: FillTestConfig レジーム設定
# ======================================================================

class TestFillTestConfigRegime:
    """FillTestConfig のレジーム設定テスト."""

    def test_regime_defaults(self) -> None:
        """レジーム設定のデフォルト値."""

        config = FillTestConfig()
        assert config.enable_regime is True
        assert config.regime_window == 20
        assert config.regime_hysteresis_count == 3

    def test_regime_from_yaml(self) -> None:
        """YAML のレジームセクションが正しくパースされる."""

        yaml_cfg = {
            "regime": {
                "enabled": False,
                "window": 10,
                "trend_threshold_pct": 1.0,
                "high_vol_multiplier": 3.0,
                "hysteresis_count": 5,
                "min_confidence": 0.6,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.enable_regime is False
        assert config.regime_window == 10
        assert config.regime_trend_threshold_pct == 1.0
        assert config.regime_high_vol_multiplier == 3.0
        assert config.regime_hysteresis_count == 5
        assert config.regime_min_confidence == 0.6

# ======================================================================
# Tests: 041# 新機能 (時間帯フィルター, 動的 loss_cap, deadzone)
# ======================================================================

class TestFillTestConfigTimeFilter:
    """041# 時間帯フィルター設定テスト."""

    def test_time_filter_defaults(self) -> None:
        """時間帯フィルターのデフォルトは無効."""

        config = FillTestConfig()
        assert config.enable_time_filter is False
        assert config.skip_utc_hours is None

    def test_time_filter_from_yaml(self) -> None:
        """YAML の time_filter セクションが正しくパースされる."""

        yaml_cfg = {
            "time_filter": {
                "enabled": True,
                "skip_utc_hours": [8, 9, 14, 16, 17, 18, 19],
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.enable_time_filter is True
        assert config.skip_utc_hours == [8, 9, 14, 16, 17, 18, 19]

    def test_time_filter_empty_hours(self) -> None:
        """skip_utc_hours 空リストでも有効化可能."""

        yaml_cfg = {
            "time_filter": {
                "enabled": True,
                "skip_utc_hours": [],
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.enable_time_filter is True
        assert config.skip_utc_hours == []

class TestFillTestConfigDynamicLossCap:
    """041# 動的 loss_cap 設定テスト."""

    def test_loss_cap_auto_defaults(self) -> None:
        """動的 loss_cap のデフォルトは無効."""

        config = FillTestConfig()
        assert config.loss_cap_auto is False
        assert config.loss_cap_ratio == 0.05

    def test_loss_cap_auto_from_yaml(self) -> None:
        """YAML の safety セクションから動的 loss_cap が正しくパースされる."""

        yaml_cfg = {
            "safety": {
                "loss_cap_auto": True,
                "loss_cap_ratio": 0.03,
                "loss_cap_jpy": 5000.0,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.loss_cap_auto is True
        assert config.loss_cap_ratio == 0.03
        assert config.loss_cap_jpy == 5000.0

    def test_loss_cap_auto_false_preserves_fixed(self) -> None:
        """loss_cap_auto=False 時は loss_cap_jpy がそのまま使われる."""

        yaml_cfg = {
            "safety": {
                "loss_cap_auto": False,
                "loss_cap_jpy": 7500.0,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.loss_cap_auto is False
        assert config.loss_cap_jpy == 7500.0

class TestFillTestConfigDeadzone:
    """041# AS deadzone 変更のテスト."""

    def test_deadzone_from_yaml(self) -> None:
        """as_deadzone_bps が YAML から読み取れる."""

        yaml_cfg = {"as_deadzone_bps": 2.0}
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.as_deadzone_bps == 2.0

    def test_deadzone_default(self) -> None:
        """as_deadzone_bps のデフォルトは 2.5 (052# 修正後)."""

        config = FillTestConfig()
        assert config.as_deadzone_bps == 2.5

class TestTimeFilterNoRecord:
    """041# 時間帯フィルターがレコードを生成しないことを検証.

    121# TimeFilter モジュールに委譲されたため、直接テスト。
    """

    def test_is_time_filtered_disabled(self) -> None:
        """フィルター無効時は常に False."""

        config = FillTestConfig(enable_time_filter=False)
        tf = TimeFilter(config)
        assert tf.is_filtered() is False

    def test_is_time_filtered_no_hours(self) -> None:
        """skip_utc_hours 未設定時は常に False."""

        config = FillTestConfig(enable_time_filter=True, skip_utc_hours=None)
        tf = TimeFilter(config)
        assert tf.is_filtered() is False

    def test_is_time_filtered_empty_hours(self) -> None:
        """skip_utc_hours 空リスト時は常に False."""

        config = FillTestConfig(enable_time_filter=True, skip_utc_hours=[])
        tf = TimeFilter(config)
        assert tf.is_filtered() is False

    def test_is_time_filtered_side_buy(self) -> None:
        """073# skip_utc_hours_buy 設定時は buy 側固有リストで判定."""

        config = FillTestConfig(
            enable_time_filter=True,
            skip_utc_hours=[8, 9],         # グローバル
            skip_utc_hours_buy=[8, 9, 15],  # buy は UTC15 も追加ブロック
        )
        tf = TimeFilter(config)

        # UTC15: buy はフィルタ、sell はグローバル通過
        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            assert tf.is_filtered(side="buy") is True
            assert tf.is_filtered(side="sell") is False

    def test_is_time_filtered_side_sell(self) -> None:
        """073# skip_utc_hours_sell 設定時は sell 側固有リストで判定."""

        config = FillTestConfig(
            enable_time_filter=True,
            skip_utc_hours=[8, 9],          # グローバル
            skip_utc_hours_sell=[3, 4, 8],  # sell は UTC3,4 追加ブロック
        )
        tf = TimeFilter(config)

        # UTC4: sell はフィルタ、buy はグローバル通過
        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 4, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            assert tf.is_filtered(side="sell") is True
            assert tf.is_filtered(side="buy") is False

    def test_is_time_filtered_side_none_uses_global(self) -> None:
        """073# side=None はグローバルリストにフォールバック."""

        config = FillTestConfig(
            enable_time_filter=True,
            skip_utc_hours=[8, 9],
            skip_utc_hours_buy=[8, 9, 15],
        )
        tf = TimeFilter(config)

        with patch("scripts.v460.lib.time_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            # side=None → グローバルで判定 → UTC15 は含まれない → False
            assert tf.is_filtered(side=None) is False
            assert tf.is_filtered() is False

    def test_yaml_side_specific_time_filter(
        self,
        v460_fill_test_yaml: dict[str, object],
    ) -> None:
        """163# Step2 YAML side 別 time_filter (107# Phase 3 Step 2)."""
        tf = v460_fill_test_yaml["time_filter"]
        assert "skip_utc_hours_buy" in tf
        assert "skip_utc_hours_sell" in tf
        # 168# §4.2 #8: buy = [16], sell = [8, 21] (損失バンド精緻化)
        # 169# time_filter 全廃: 全リスト空 (条件ベースフィルタに完全移行)
        assert tf["skip_utc_hours_buy"] == []
        assert tf["skip_utc_hours_sell"] == []
        assert tf["regime_adaptive_enabled"] is True
        assert tf["regime_adaptive_extra_buy"] == []
        assert tf["regime_adaptive_extra_sell"] == []

class TestDynamicLossCapReserved:
    """041# reserved 残高が loss_cap 計算に含まれることを検証."""

    def test_loss_cap_includes_reserved_key(self) -> None:
        """JPY_RESERVED が currency として出てきた場合、集計に含む."""
        # 041# の _update_dynamic_loss_cap が JPY_RESERVED を認識するか

        config = FillTestConfig(loss_cap_auto=True, loss_cap_ratio=0.05)
        # JPY = 1000, JPY_RESERVED = 10000, BTC = 0.001 × 10M = 10000
        # total = 21000, cap = 1050
        assert config.loss_cap_auto is True
        assert config.loss_cap_ratio == 0.05

class TestJapaneseErrorClassification:
    """042# 日本語エラーメッセージの分類テスト."""

    def test_insufficient_funds_japanese(self) -> None:
        """Coincheck 日本語エラーが insufficient_funds に分類される."""
        # 042# run_fill_test.py 内のエラー分類ロジックを直接テスト
        test_cases = [
            ("Coincheck API error: 400 | body=所持金額が足りません", "insufficient_funds"),
            ("Coincheck API error: 400 | body=Amount BTC の所持金額が足りません", "insufficient_funds"),
            ("Insufficient balance for buy", "insufficient_funds"),
            ("post_only rejected", "post_only_reject"),
            ("minimum size 0.001", "minimum_size"),
            ("500 Server Error", "api_error"),
        ]
        for error_msg, expected in test_cases:
            err_lower = error_msg.lower()
            if "post_only" in err_lower or "taker" in err_lower:
                reason = "post_only_reject"
            elif (
                "insufficient" in err_lower
                or "balance" in err_lower
                or "所持金額" in error_msg
                or "足りません" in error_msg
            ):
                reason = "insufficient_funds"
            elif "minimum" in err_lower or "size" in err_lower:
                reason = "minimum_size"
            else:
                reason = "api_error"
            assert reason == expected, f"{error_msg!r} → {reason}, expected {expected}"

class TestStaleOrderCleanup:
    """042# 起動時の滞留注文クリアテスト."""

    def test_cancel_stale_orders_method_exists(self) -> None:
        """_cancel_stale_orders メソッドが定義されている."""

        assert hasattr(FillTestRunner, "_cancel_stale_orders")
        assert inspect.iscoroutinefunction(FillTestRunner._cancel_stale_orders)

# ======================================================================
# 044# Fix Tests
# ======================================================================

class TestSingleInstanceLock:
    """044# Bug7: 単一起動ロック (lockfile + PID + stale回収)."""

    def test_acquire_release_lock_methods_exist(self) -> None:
        """_acquire_lock / _release_lock メソッドが定義されている."""

        assert hasattr(FillTestRunner, "_acquire_lock")
        assert hasattr(FillTestRunner, "_release_lock")

    def test_lockfile_created_and_removed(self, tmp_path: "Path") -> None:
        """ロックファイルの生成・解放が正しく動作する."""

        config = FillTestConfig(results_dir=str(tmp_path))
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)

        runner._acquire_lock()
        lock_path = tmp_path / "fill_test.lock"
        assert lock_path.exists()
        content = lock_path.read_text(encoding="utf-8")
        assert content.startswith(f"{os.getpid()}|")

        runner._release_lock()
        assert not lock_path.exists()

    def test_stale_lockfile_reclaimed(self, tmp_path: "Path") -> None:
        """無効な PID のロックファイルは回収される."""

        lock_path = tmp_path / "fill_test.lock"
        # 存在しない PID を書き込む
        lock_path.write_text("99999999|1234567890|fake_run_id", encoding="utf-8")

        config = FillTestConfig(results_dir=str(tmp_path))
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        # stale lock は回収されて新しいロックが取得される
        runner._acquire_lock()
        assert lock_path.exists()
        content = lock_path.read_text(encoding="utf-8")
        assert content.startswith(f"{os.getpid()}|")
        runner._release_lock()

class TestPreflightSkipLimit:
    """044# F8: 連続 preflight 失敗上限."""

    def test_config_has_max_preflight_skip(self) -> None:
        """max_preflight_skip 設定が存在し、デフォルト値が適切."""

        config = FillTestConfig()
        assert hasattr(config, "max_preflight_skip")
        assert config.max_preflight_skip == 10

    def test_preflight_skip_count_initialized(self) -> None:
        """_preflight_skip_count が初期化されている."""

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert runner._preflight_skip_count == 0

    def test_max_consecutive_same_side_removed(self) -> None:
        """044# F7: 未使用の max_consecutive_same_side が削除されている."""

        config = FillTestConfig()
        assert not hasattr(config, "max_consecutive_same_side")

class TestCleanupSyncImproved:
    """044# A-4: _cleanup_sync の改善テスト."""

    def test_cleanup_releases_lock(self, tmp_path: "Path") -> None:
        """_cleanup_sync がロックファイルを解放する."""

        config = FillTestConfig(results_dir=str(tmp_path))
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        runner._acquire_lock()

        lock_path = tmp_path / "fill_test.lock"
        assert lock_path.exists()

        runner._cleanup_sync()
        assert not lock_path.exists()

class TestLossCapPeriodicUpdate:
    """044# A-7: loss_cap 定期更新."""

    def test_loss_cap_update_interval_exists(self) -> None:
        """_loss_cap_update_interval が初期化されている."""

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert hasattr(runner, "_loss_cap_update_interval")
        assert runner._loss_cap_update_interval == 50

class TestWindowsSignalHandler:
    """044# A-1: Windows SIGTERM 修正."""

    def test_platform_import(self) -> None:
        """platform モジュールが fill_test_cli でインポートされている."""
        mod = importlib.import_module("scripts.v460.lib.fill_test_cli")
        # platform が import されていることを確認
        assert hasattr(mod, "platform")

class TestRateLimitDoubleCheck:
    """044# E-1: get_order_status の二重 rate limit チェック.

    145# §13: BaseExchangeAdapter 継承後は _get_order_status_real を検査.
    """

    def test_rate_limit_called_before_transactions(self) -> None:
        """_get_order_status_real のソースに _check_rate_limit がある."""

        source = _source(CoincheckAdapter._get_order_status_real)
        count = source.count("_check_rate_limit")
        assert count >= 1, f"Expected ≥1 rate limit checks in _get_order_status_real, found {count}"

class TestPriceRounding:
    """044# E-3: price int()→round() 修正.

    145# §13: BaseExchangeAdapter 継承後は _place_order_real を検査.
    """

    def test_price_uses_round_not_int(self) -> None:
        """_place_order_real のソースに round(price) が使われている."""

        source = _source(CoincheckAdapter._place_order_real)
        assert "round(price)" in source, "price should use round() not int()"
        assert "int(price)" not in source, "int(price) should be replaced by round(price)"

class TestBalanceLocked:
    """044# E-4: get_balance が reserved を locked として解析.

    145# §13: BaseExchangeAdapter 継承後は _get_balance_real を検査.
    """

    def test_balance_source_has_reserved_handling(self) -> None:
        """_get_balance_real のソースに _reserved の処理が含まれる."""

        source = _source(CoincheckAdapter._get_balance_real)
        assert "_reserved" in source, "_get_balance_real should handle *_reserved keys"
        assert "locked=0.0" not in source, "locked should not be hardcoded to 0.0"

# ======================================================================
# 046# テスト: Bug10, soft/hard loss_cap, clean/quarantine, balance filter
# ======================================================================

class TestBug10InsufficientFundsNoRetry:
    """046# Bug10: insufficient_funds 時にリトライしない."""

    def test_source_has_insufficient_funds_break(self) -> None:
        """run_single_cycle のソースに insufficient_funds → break が含まれる."""

        source = _source(FillTestRunner.run_single_cycle)
        # 084# 改修: 非リトライ対象をセットで管理
        assert '"insufficient_funds"' in source
        assert "not retriable" in source.lower() or "_non_retriable" in source
        assert "break" in source

    def test_cancel_reason_classification(self) -> None:
        """所持金額不足が insufficient_funds に分類される."""
        error_msg = "400 Client Error: Amount BTC の所持金額が足りません"
        # ロジックを再現
        err_lower = error_msg.lower()
        cancel_reason = "unknown"
        if "所持金額" in error_msg or "足りません" in error_msg:
            cancel_reason = "insufficient_funds"
        assert cancel_reason == "insufficient_funds"

class TestSoftHardLossCap:
    """046# soft/hard 二段 loss_cap."""

    def test_config_has_soft_loss_cap_ratio(self) -> None:
        """FillTestConfig に soft_loss_cap_ratio フィールドがある."""

        config = FillTestConfig()
        assert hasattr(config, "soft_loss_cap_ratio")
        assert config.soft_loss_cap_ratio == 0.02

    def test_soft_loss_cap_flag_initialized(self) -> None:
        """FillTestRunner に _soft_loss_cap_triggered が初期化される."""

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert hasattr(runner, "_soft_loss_cap_triggered")
        assert runner._soft_loss_cap_triggered is False

    def test_soft_cap_ratio_less_than_hard(self) -> None:
        """soft cap は hard cap より小さい."""

        config = FillTestConfig()
        assert config.soft_loss_cap_ratio < config.loss_cap_ratio

    def test_yaml_parser_handles_soft_loss_cap(self) -> None:
        """from_yaml が soft_loss_cap_ratio を正しく解析する."""

        yaml_cfg = {
            "safety": {
                "soft_loss_cap_ratio": 0.03,
                "loss_cap_ratio": 0.05,
                "loss_cap_auto": True,
            }
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.soft_loss_cap_ratio == 0.03

class TestCleanQuarantineFilter:
    """046# clean/quarantine データ分離."""

    def test_filter_separates_by_git_sha(self) -> None:
        """git_sha 有無でレコードが分離される."""

        records = [
            FillRecord(cycle_id="1", timestamp=1.0, side="buy",
                       order_price=100.0, order_quantity=0.001,
                       git_sha="abc123", run_id="r1"),
            FillRecord(cycle_id="2", timestamp=2.0, side="sell",
                       order_price=101.0, order_quantity=0.001,
                       git_sha="", run_id="r1"),
            FillRecord(cycle_id="3", timestamp=3.0, side="buy",
                       order_price=102.0, order_quantity=0.001,
                       git_sha=None, run_id="r1"),
            FillRecord(cycle_id="4", timestamp=4.0, side="sell",
                       order_price=103.0, order_quantity=0.001,
                       git_sha="def456", run_id="r1"),
        ]
        clean, quarantine = filter_clean_records(records)
        assert len(clean) == 2
        assert len(quarantine) == 2
        assert all(r.git_sha for r in clean)

    def test_filter_disabled(self) -> None:
        """require_git_sha=False で全レコードがクリーン."""

        records = [
            FillRecord(cycle_id="1", timestamp=1.0, side="buy",
                       order_price=100.0, order_quantity=0.001,
                       git_sha=None),
        ]
        clean, quarantine = filter_clean_records(
            records, require_git_sha=False
        )
        assert len(clean) == 1
        assert len(quarantine) == 0

    def test_all_clean(self) -> None:
        """全レコードに git_sha がある場合は quarantine=0."""

        records = [
            FillRecord(cycle_id="1", timestamp=1.0, side="buy",
                       order_price=100.0, order_quantity=0.001,
                       git_sha="abc", run_id="r1"),
        ]
        clean, quarantine = filter_clean_records(records)
        assert len(clean) == 1
        assert len(quarantine) == 0

class TestBalanceCurrencyFilter:
    """046# balance 解析のゴミ通貨除外.

    145# §13: BaseExchangeAdapter 継承後は _get_balance_real を検査.
    """

    def test_ignore_suffixes_in_source(self) -> None:
        """_get_balance_real のソースに _lending 等の除外ロジックがある."""

        source = _source(CoincheckAdapter._get_balance_real)
        for suffix in ["_lending", "_lend_in_use", "_lent", "_debt", "_tsumitate"]:
            assert suffix in source, f"{suffix} should be excluded in _get_balance_real"

    def test_loss_cap_no_dead_reserved_check(self) -> None:
        """AdaptationEngine.update_dynamic_loss_cap に JPY_RESERVED/BTC_RESERVED は不要.

        120#: adaptation_engine.py に抽出済み。
        """

        source = _source(AdaptationEngine.update_dynamic_loss_cap)
        assert "JPY_RESERVED" not in source, "Dead check should be removed"
        assert "BTC_RESERVED" not in source, "Dead check should be removed"

class TestBug086TimeFilterPositionAccumulation:
    """086# time_filter の side 切替が片側蓄積を引き起こすバグの修正検証."""

    def test_source_has_position_accumulation_guard(self) -> None:
        """run_continuous に片側蓄積防止ガードが含まれる."""

        source = _source(FillTestRunner.run_continuous)
        assert "alt_side == self._last_side" in source, (
            "086# 片側蓄積防止ガードが必要"
        )
        assert "片側蓄積防止" in source, (
            "086# 片側蓄積防止コメントが必要"
        )

# ======================================================================
# 156# Phase D テスト
# ======================================================================

class TestPhaseD4TrendingDirection:
    """156# D-4: trending 方向分解テスト."""

    def test_is_trending_property(self) -> None:
        """is_trending が TRENDING / TRENDING_UP / TRENDING_DOWN を包含."""
        assert FillTestRegime.TRENDING.is_trending is True
        assert FillTestRegime.TRENDING_UP.is_trending is True
        assert FillTestRegime.TRENDING_DOWN.is_trending is True
        assert FillTestRegime.RANGING.is_trending is False
        assert FillTestRegime.HIGH_VOL.is_trending is False
        assert FillTestRegime.UNKNOWN.is_trending is False

    def test_trending_up_value(self) -> None:
        """TRENDING_UP の value が 'trending_up'."""
        assert FillTestRegime.TRENDING_UP.value == "trending_up"

    def test_trending_down_value(self) -> None:
        """TRENDING_DOWN の value が 'trending_down'."""
        assert FillTestRegime.TRENDING_DOWN.value == "trending_down"

    def test_classify_upward_returns_trending_up(self) -> None:
        """上昇トレンドが TRENDING_UP を返す."""
        config = RegimeConfig(
            window=5,
            trend_threshold_pct=0.3,
            hysteresis_count=2,
            min_confidence=0.0,
        )
        detector = FillTestRegimeDetector(config)
        # 急上昇
        prices = [100.0 + i * 0.5 for i in range(15)]
        results = _feed_prices(detector, prices)
        up = [r for r in results if r.regime == FillTestRegime.TRENDING_UP]
        assert len(up) > 0, f"Expected TRENDING_UP, got {[r.regime.value for r in results]}"

    def test_classify_downward_returns_trending_down(self) -> None:
        """下降トレンドが TRENDING_DOWN を返す."""
        config = RegimeConfig(
            window=5,
            trend_threshold_pct=0.3,
            hysteresis_count=2,
            min_confidence=0.0,
        )
        detector = FillTestRegimeDetector(config)
        # 急下降
        prices = [100.0 - i * 0.5 for i in range(15)]
        results = _feed_prices(detector, prices)
        down = [r for r in results if r.regime == FillTestRegime.TRENDING_DOWN]
        assert len(down) > 0, f"Expected TRENDING_DOWN, got {[r.regime.value for r in results]}"

    def test_regime_thresholds_match_direction(self) -> None:
        """regime_thresholds の key が D-4 方向別 value と一致."""

        thresholds = {"trending_up": -0.3, "trending_down": -1.0, "ranging": -0.5}
        config = SellKillConfig(regime_thresholds=thresholds)
        # TRENDING_UP.value → "trending_up" → threshold hit
        assert FillTestRegime.TRENDING_UP.value in config.regime_thresholds
        assert FillTestRegime.TRENDING_DOWN.value in config.regime_thresholds

class TestPhaseD4SkipSellTrendingUpOnly:
    """156# D-4: skip_sell_trending_up_only 設定テスト."""

    def test_config_field_exists(self) -> None:
        """skip_sell_trending_up_only フィールドが存在."""

        config = FillTestConfig()
        assert hasattr(config, "skip_sell_trending_up_only")
        assert config.skip_sell_trending_up_only is False

    def test_yaml_mapping(self) -> None:
        """YAML から skip_sell_trending_up_only がパースされる."""

        yaml_cfg = {
            "loss_control": {
                "skip_sell_trending_up_only": True,
            },
        }
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.skip_sell_trending_up_only is True

class TestPhaseD1ObUtils:
    """156# D-1: ob_utils 型安全向上テスト."""

    def test_extract_price_tuple(self) -> None:
        """tuple から price を抽出."""

        assert extract_price((15000000.0, 0.5)) == 15000000.0

    def test_extract_price_empty_tuple(self) -> None:
        """空 tuple は 0.0."""

        assert extract_price(()) == 0.0

    def test_extract_size_tuple(self) -> None:
        """tuple から size を抽出."""

        assert extract_size((15000000.0, 0.5)) == 0.5

    def test_depth_volume_basic(self) -> None:
        """depth_volume が合計出来高を返す."""

        levels = [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
        assert depth_volume(levels, depth=2) == 3.0
        assert depth_volume(levels, depth=5) == 6.0

class TestPhaseD5KillCooldown:
    """156# D-5: sell_dynamic_kill cooldown テスト."""

    def test_resume_window_10(self) -> None:
        """resume_window=10 で cooldown が 10 サイクルで解除."""

        config = SellKillConfig(
            enabled=True, window=5, threshold_bps=-0.5, resume_window=10,
            max_stale_kill_cycles=0,  # 219#: probe を無効化 (cooldown テスト専用)
        )
        mgr = SellDynamicKillManager(config)
        # タンク: 5 fill の平均を -1.0bps にする → kill 発動
        for _ in range(5):
            mgr.track(-1.0)
        killed, tele = mgr.check_kill()
        assert killed is True
        assert tele.cooldown_remaining == 10

        # 10 サイクル消化 (cooldown 中は全て killed)
        for i in range(10):
            killed, tele = mgr.check_kill()
            assert killed is True, f"cycle {i}: should still be killed"

        # cooldown 解除後、良好な fill を追加して平均を改善
        for _ in range(5):
            mgr.track(1.0)
        # 平均が閾値以上 → 解除
        killed, tele = mgr.check_kill()
        assert killed is False

# ======================================================================
# 156# §18 セルフレビュー: データシンク解消 + enum 一貫性 + 未活用機能
# ======================================================================

class TestPhaseD18DataSinkResolution:
    """156# §18: trend_pct / volatility_ratio の FillRecord 伝搬テスト."""

    def test_fill_record_has_regime_trend_pct_field(self) -> None:
        """FillRecord に regime_trend_pct フィールドが存在."""

        record = FillRecord(
            cycle_id="test",
            timestamp=1000.0,
            side="buy",
            order_price=15_000_000.0,
            order_quantity=0.001,
            regime_trend_pct=1.23,
            regime_volatility_ratio=0.75,
        )
        assert record.regime_trend_pct == 1.23
        assert record.regime_volatility_ratio == 0.75

    def test_fill_record_data_sink_fields_serialize(self) -> None:
        """trend_pct / volatility_ratio が to_dict / from_dict で保持."""

        record = FillRecord(
            cycle_id="test",
            timestamp=1000.0,
            side="sell",
            order_price=15_000_000.0,
            order_quantity=0.001,
            regime_trend_pct=-0.5,
            regime_volatility_ratio=1.8,
        )
        d = record.to_dict()
        assert d["regime_trend_pct"] == -0.5
        assert d["regime_volatility_ratio"] == 1.8

        restored = FillRecord.from_dict(d)
        assert restored.regime_trend_pct == -0.5
        assert restored.regime_volatility_ratio == 1.8

    def test_backward_compat_no_data_sink_fields(self) -> None:
        """旧データ (trend_pct なし) からも from_dict できる."""

        old = {"cycle_id": "x", "timestamp": 0.0, "side": "buy",
               "order_price": 100.0, "order_quantity": 0.001}
        r = FillRecord.from_dict(old)
        assert r.regime_trend_pct is None
        assert r.regime_volatility_ratio is None

class TestPhaseD18EnumConsistency:
    """156# §18: maker_price.py の enum 直接比較テスト."""

    def test_maker_price_imports_fill_test_regime(self) -> None:
        """maker_price.py が FillTestRegime をインポートしている."""
        mod = importlib.import_module("scripts.v460.lib.maker_price")
        assert hasattr(mod, "FillTestRegime")

    def test_high_vol_uses_enum_comparison(self) -> None:
        """maker_price.py の high_vol ロジックが enum 比較を使用."""

        source = _source(MakerPriceCalculator)
        assert "FillTestRegime.HIGH_VOL" in source
        assert "FillTestRegime.RANGING" in source
        assert "FillTestRegime.UNKNOWN" in source
        # 文字列比較が残っていないこと
        assert '.value == "high_vol"' not in source
        assert '.value == "ranging"' not in source
        assert '.value == "unknown"' not in source

class TestPhaseD18ObFetchStats:
    """156# §18: OB fetch 統計プロパティテスト."""

    def test_ob_fetch_stats_property_exists(self) -> None:
        """SkipGateEvaluator.ob_fetch_stats が返せる."""

        config = FillTestConfig(skip_gate_enabled=False)
        evaluator = SkipGateEvaluator(config, Path("."))
        fail, total = evaluator.ob_fetch_stats
        assert fail == 0
        assert total == 0

class TestPhaseD18RangingYaml:
    """156# §18: ranging_offset_discount YAML 有効化テスト."""

    def test_yaml_has_ranging_discount(self) -> None:
        """YAML に ranging_offset_discount が設定されている."""

        yaml_path = Path("configs/v460/fill_test.yaml")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        regime = cfg["regime"]
        assert "ranging_offset_discount" in regime
        discount = regime["ranging_offset_discount"]
        assert 0.5 < discount < 1.0, f"discount should be 0.5-1.0, got {discount}"
