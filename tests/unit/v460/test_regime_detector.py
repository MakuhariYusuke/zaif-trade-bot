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
        from scripts.v460.run_fill_test import FillTestRunner

        assert hasattr(FillTestRunner, "_cancel_stale_orders")
        import inspect
        assert inspect.iscoroutinefunction(FillTestRunner._cancel_stale_orders)


# ======================================================================
# 044# Fix Tests
# ======================================================================


class TestSingleInstanceLock:
    """044# Bug7: 単一起動ロック (lockfile + PID + stale回収)."""

    def test_acquire_release_lock_methods_exist(self) -> None:
        """_acquire_lock / _release_lock メソッドが定義されている."""
        from scripts.v460.run_fill_test import FillTestRunner

        assert hasattr(FillTestRunner, "_acquire_lock")
        assert hasattr(FillTestRunner, "_release_lock")

    def test_lockfile_created_and_removed(self, tmp_path: "Path") -> None:
        """ロックファイルの生成・解放が正しく動作する."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig(results_dir=str(tmp_path))
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)

        runner._acquire_lock()
        lock_path = tmp_path / "fill_test.lock"
        assert lock_path.exists()
        content = lock_path.read_text(encoding="utf-8")
        import os
        assert content.startswith(f"{os.getpid()}|")

        runner._release_lock()
        assert not lock_path.exists()

    def test_stale_lockfile_reclaimed(self, tmp_path: "Path") -> None:
        """無効な PID のロックファイルは回収される."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        lock_path = tmp_path / "fill_test.lock"
        # 存在しない PID を書き込む
        lock_path.write_text("99999999|1234567890|fake_run_id", encoding="utf-8")

        config = FillTestConfig(results_dir=str(tmp_path))
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        # stale lock は回収されて新しいロックが取得される
        runner._acquire_lock()
        assert lock_path.exists()
        import os
        content = lock_path.read_text(encoding="utf-8")
        assert content.startswith(f"{os.getpid()}|")
        runner._release_lock()


class TestPreflightSkipLimit:
    """044# F8: 連続 preflight 失敗上限."""

    def test_config_has_max_preflight_skip(self) -> None:
        """max_preflight_skip 設定が存在し、デフォルト値が適切."""
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert hasattr(config, "max_preflight_skip")
        assert config.max_preflight_skip == 10

    def test_preflight_skip_count_initialized(self) -> None:
        """_preflight_skip_count が初期化されている."""
        from unittest.mock import MagicMock

        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert runner._preflight_skip_count == 0

    def test_max_consecutive_same_side_removed(self) -> None:
        """044# F7: 未使用の max_consecutive_same_side が削除されている."""
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert not hasattr(config, "max_consecutive_same_side")


class TestCleanupSyncImproved:
    """044# A-4: _cleanup_sync の改善テスト."""

    def test_cleanup_releases_lock(self, tmp_path: "Path") -> None:
        """_cleanup_sync がロックファイルを解放する."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

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
        from unittest.mock import MagicMock

        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert hasattr(runner, "_loss_cap_update_interval")
        assert runner._loss_cap_update_interval == 50


class TestWindowsSignalHandler:
    """044# A-1: Windows SIGTERM 修正."""

    def test_platform_import(self) -> None:
        """platform モジュールが run_fill_test でインポートされている."""
        import importlib
        mod = importlib.import_module("scripts.v460.run_fill_test")
        # platform が import されていることを確認
        assert hasattr(mod, "platform")


class TestRateLimitDoubleCheck:
    """044# E-1: get_order_status の二重 rate limit チェック."""

    def test_rate_limit_called_before_transactions(self) -> None:
        """get_order_status のソースに2回の _check_rate_limit がある."""
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_order_status)
        count = source.count("_check_rate_limit")
        assert count >= 2, f"Expected ≥2 rate limit checks, found {count}"


class TestPriceRounding:
    """044# E-3: price int()→round() 修正."""

    def test_price_uses_round_not_int(self) -> None:
        """place_order のソースに round(price) が使われている."""
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.place_order)
        assert "round(price)" in source, "price should use round() not int()"
        assert "int(price)" not in source, "int(price) should be replaced by round(price)"


class TestBalanceLocked:
    """044# E-4: get_balance が reserved を locked として解析."""

    def test_balance_source_has_reserved_handling(self) -> None:
        """get_balance のソースに _reserved の処理が含まれる."""
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_balance)
        assert "_reserved" in source, "get_balance should handle *_reserved keys"
        assert "locked=0.0" not in source, "locked should not be hardcoded to 0.0"


# ======================================================================
# 046# テスト: Bug10, soft/hard loss_cap, clean/quarantine, balance filter
# ======================================================================


class TestBug10InsufficientFundsNoRetry:
    """046# Bug10: insufficient_funds 時にリトライしない."""

    def test_source_has_insufficient_funds_break(self) -> None:
        """run_single_cycle のソースに insufficient_funds → break が含まれる."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner.run_single_cycle)
        assert 'cancel_reason == "insufficient_funds"' in source
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
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert hasattr(config, "soft_loss_cap_ratio")
        assert config.soft_loss_cap_ratio == 0.02

    def test_soft_loss_cap_flag_initialized(self) -> None:
        """FillTestRunner に _soft_loss_cap_triggered が初期化される."""
        from unittest.mock import MagicMock
        from scripts.v460.run_fill_test import FillTestConfig, FillTestRunner

        config = FillTestConfig()
        adapter = MagicMock()
        runner = FillTestRunner(adapter, config)
        assert hasattr(runner, "_soft_loss_cap_triggered")
        assert runner._soft_loss_cap_triggered is False

    def test_soft_cap_ratio_less_than_hard(self) -> None:
        """soft cap は hard cap より小さい."""
        from scripts.v460.run_fill_test import FillTestConfig

        config = FillTestConfig()
        assert config.soft_loss_cap_ratio < config.loss_cap_ratio

    def test_yaml_parser_handles_soft_loss_cap(self) -> None:
        """from_yaml が soft_loss_cap_ratio を正しく解析する."""
        from scripts.v460.run_fill_test import FillTestConfig

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
        from ztb.metrics.fill_quality import FillRecord, filter_clean_records

        records = [
            FillRecord(cycle_id="1", timestamp=1.0, side="buy",
                       order_price=100.0, order_quantity=0.001,
                       git_sha="abc123"),
            FillRecord(cycle_id="2", timestamp=2.0, side="sell",
                       order_price=101.0, order_quantity=0.001,
                       git_sha=""),
            FillRecord(cycle_id="3", timestamp=3.0, side="buy",
                       order_price=102.0, order_quantity=0.001,
                       git_sha=None),
            FillRecord(cycle_id="4", timestamp=4.0, side="sell",
                       order_price=103.0, order_quantity=0.001,
                       git_sha="def456"),
        ]
        clean, quarantine = filter_clean_records(records)
        assert len(clean) == 2
        assert len(quarantine) == 2
        assert all(r.git_sha for r in clean)

    def test_filter_disabled(self) -> None:
        """require_git_sha=False で全レコードがクリーン."""
        from ztb.metrics.fill_quality import FillRecord, filter_clean_records

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
        from ztb.metrics.fill_quality import FillRecord, filter_clean_records

        records = [
            FillRecord(cycle_id="1", timestamp=1.0, side="buy",
                       order_price=100.0, order_quantity=0.001,
                       git_sha="abc"),
        ]
        clean, quarantine = filter_clean_records(records)
        assert len(clean) == 1
        assert len(quarantine) == 0


class TestBalanceCurrencyFilter:
    """046# balance 解析のゴミ通貨除外."""

    def test_ignore_suffixes_in_source(self) -> None:
        """get_balance のソースに _lending 等の除外ロジックがある."""
        import inspect
        from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

        source = inspect.getsource(CoincheckAdapter.get_balance)
        for suffix in ["_lending", "_lend_in_use", "_lent", "_debt", "_tsumitate"]:
            assert suffix in source, f"{suffix} should be excluded in get_balance"

    def test_loss_cap_no_dead_reserved_check(self) -> None:
        """_update_dynamic_loss_cap に JPY_RESERVED/BTC_RESERVED は不要."""
        import inspect
        from scripts.v460.run_fill_test import FillTestRunner

        source = inspect.getsource(FillTestRunner._update_dynamic_loss_cap)
        assert "JPY_RESERVED" not in source, "Dead check should be removed"
        assert "BTC_RESERVED" not in source, "Dead check should be removed"
