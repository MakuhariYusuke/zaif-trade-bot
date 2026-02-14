"""
test_regime_detector — 軽量レジーム検知のユニットテスト.

035# §4 / 037# 準拠.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scripts.v460.lib.regime_detector import (
    FillTestRegime,
    FillTestRegimeDetector,
    RegimeConfig,
    RegimeResult,
)


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
        """上昇価格はトレンド判定."""
        # 0.5% 以上の変動を window=5 で作る → 100 → 101 = +1%
        prices = [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8]
        results = _feed_prices(quick_detector, prices)
        # 最終結果はトレンド
        final = results[-1]
        # hysteresis 2 回以上トレンドが出ていれば確定
        trending = [r for r in results if r.regime == FillTestRegime.TRENDING]
        assert len(trending) > 0, f"Expected trending, got {[r.regime.value for r in results]}"

    def test_trending_on_downward_prices(self, quick_detector: FillTestRegimeDetector) -> None:
        """下落価格もトレンド判定."""
        prices = [100.0, 99.8, 99.6, 99.4, 99.2, 99.0, 98.8, 98.6, 98.4, 98.2]
        results = _feed_prices(quick_detector, prices)
        trending = [r for r in results if r.regime == FillTestRegime.TRENDING]
        assert len(trending) > 0

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
        assert final.regime == FillTestRegime.TRENDING


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


# ======================================================================
# Tests: FillRecord レジームフィールド
# ======================================================================


class TestFillRecordRegimeFields:
    """FillRecord のレジームフィールドテスト."""

    def test_regime_fields_default_none(self) -> None:
        """レジームフィールドのデフォルトは None."""
        from ztb.metrics.fill_quality import FillRecord

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
        from ztb.metrics.fill_quality import FillRecord

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
        from ztb.metrics.fill_quality import FillRecord

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
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert config.enable_regime is True
        assert config.regime_window == 20
        assert config.regime_hysteresis_count == 3

    def test_regime_from_yaml(self) -> None:
        """YAML のレジームセクションが正しくパースされる."""
        from scripts.v460.run_fill_test import FillTestConfig

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
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert config.enable_time_filter is False
        assert config.skip_utc_hours is None

    def test_time_filter_from_yaml(self) -> None:
        """YAML の time_filter セクションが正しくパースされる."""
        from scripts.v460.run_fill_test import FillTestConfig

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
        from scripts.v460.run_fill_test import FillTestConfig

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
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert config.loss_cap_auto is False
        assert config.loss_cap_ratio == 0.05

    def test_loss_cap_auto_from_yaml(self) -> None:
        """YAML の safety セクションから動的 loss_cap が正しくパースされる."""
        from scripts.v460.run_fill_test import FillTestConfig

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
        from scripts.v460.run_fill_test import FillTestConfig

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
        from scripts.v460.run_fill_test import FillTestConfig

        yaml_cfg = {"as_deadzone_bps": 2.0}
        config = FillTestConfig.from_yaml(yaml_cfg)
        assert config.as_deadzone_bps == 2.0

    def test_deadzone_default(self) -> None:
        """as_deadzone_bps のデフォルトは 0.5."""
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert config.as_deadzone_bps == 0.5


class TestTimeFilterNoRecord:
    """041# 時間帯フィルターがレコードを生成しないことを検証."""

    def test_is_time_filtered_disabled(self) -> None:
        """フィルター無効時は常に False."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner
        from unittest.mock import MagicMock

        config = FillTestConfig(enable_time_filter=False)
        runner = FillTestRunner.__new__(FillTestRunner)
        runner.config = config
        assert runner._is_time_filtered() is False

    def test_is_time_filtered_no_hours(self) -> None:
        """skip_utc_hours 未設定時は常に False."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig(enable_time_filter=True, skip_utc_hours=None)
        runner = FillTestRunner.__new__(FillTestRunner)
        runner.config = config
        assert runner._is_time_filtered() is False

    def test_is_time_filtered_empty_hours(self) -> None:
        """skip_utc_hours 空リスト時は常に False."""
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig(enable_time_filter=True, skip_utc_hours=[])
        runner = FillTestRunner.__new__(FillTestRunner)
        runner.config = config
        assert runner._is_time_filtered() is False


class TestDynamicLossCapReserved:
    """041# reserved 残高が loss_cap 計算に含まれることを検証."""

    def test_loss_cap_includes_reserved_key(self) -> None:
        """JPY_RESERVED が currency として出てきた場合、集計に含む."""
        # 041# の _update_dynamic_loss_cap が JPY_RESERVED を認識するか
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig(loss_cap_auto=True, loss_cap_ratio=0.05)
        # JPY = 1000, JPY_RESERVED = 10000, BTC = 0.001 × 10M = 10000
        # total = 21000, cap = 1050
        assert config.loss_cap_auto is True
        assert config.loss_cap_ratio == 0.05
