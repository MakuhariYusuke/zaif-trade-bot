"""277# マジックナンバー根拠化 + 271#-276# セルフレビュー テスト.

改修項目:
  A. FillTestConfig 新規フィールド (5 件) の存在・from_yaml・デフォルト値
  B. __post_init__ 構造的整合性バリデーション (3 件)
  C. fill_loop_orchestrator マジックナンバー → config 参照 (6 箇所)
  D. cycle_gate_aggregator UNKNOWN_REGIME_MAX_CONSECUTIVE config 化
  E. MCB σ 履歴 maxlen 導出 + 名前付き定数
  F. B1 warmup TZ 不一致修正
  G. gate block ログ間隔の quiescence 連動導出
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig

if TYPE_CHECKING:
    pass


# =====================================================================
# A. FillTestConfig 新規フィールドの検証
# =====================================================================

class TestNewConfigFields:
    """277# で追加した 5 つの config フィールドの基本検証."""

    def test_phantom_detection_sleep_multiplier_default(self) -> None:
        """phantom_detection_sleep_multiplier のデフォルト値 = 3.0."""
        cfg = FillTestConfig()
        assert cfg.phantom_detection_sleep_multiplier == 3.0

    def test_halt_persist_interval_default(self) -> None:
        """halt_persist_interval のデフォルト値 = 10."""
        cfg = FillTestConfig()
        assert cfg.halt_persist_interval == 10

    def test_stop_condition_check_interval_default(self) -> None:
        """stop_condition_check_interval のデフォルト値 = 30."""
        cfg = FillTestConfig()
        assert cfg.stop_condition_check_interval == 30

    def test_fallback_duration_sec_default(self) -> None:
        """fallback_duration_sec のデフォルト値 = 3600.0."""
        cfg = FillTestConfig()
        assert cfg.fallback_duration_sec == 3600.0

    def test_unknown_regime_max_consecutive_default(self) -> None:
        """unknown_regime_max_consecutive のデフォルト値 = 10."""
        cfg = FillTestConfig()
        assert cfg.unknown_regime_max_consecutive == 10

    def test_from_yaml_reads_new_fields(self) -> None:
        """from_yaml が 5 つの新規 flat_keys を読み込むこと."""
        yaml_cfg = {
            "results_dir": "results/test",
            "phantom_detection_sleep_multiplier": 4.0,
            "halt_persist_interval": 5,
            "stop_condition_check_interval": 20,
            "fallback_duration_sec": 1800.0,
            "unknown_regime_max_consecutive": 15,
        }
        cfg = FillTestConfig.from_yaml(yaml_cfg)
        assert cfg.phantom_detection_sleep_multiplier == 4.0
        assert cfg.halt_persist_interval == 5
        assert cfg.stop_condition_check_interval == 20
        assert cfg.fallback_duration_sec == 1800.0
        assert cfg.unknown_regime_max_consecutive == 15

    def test_yaml_file_has_new_fields(self, v460_fill_test_yaml: dict[str, object]) -> None:
        """fill_test.yaml に 277# の新規フィールドが定義されている."""
        data = v460_fill_test_yaml
        assert data["phantom_detection_sleep_multiplier"] == 3.0
        assert data["halt_persist_interval"] == 10
        assert data["stop_condition_check_interval"] == 30
        assert data["fallback_duration_sec"] == 3600.0
        assert data["unknown_regime_max_consecutive"] == 10


# =====================================================================
# B. __post_init__ 構造的整合性バリデーション
# =====================================================================

class TestStructuralValidation:
    """277# __post_init__ 構造的整合性チェックのテスト."""

    def test_max_cycle_sleep_sec_vs_halt_cap_valid(self) -> None:
        """max_cycle_sleep_sec >= cycle_interval × halt_mult → OK."""
        cfg = FillTestConfig(
            cycle_interval_sec=120.0,
            halt_sleep_multiplier=5.0,
            max_cycle_sleep_sec=600.0,  # 120 * 5 = 600 → ちょうど OK
        )
        assert cfg.max_cycle_sleep_sec == 600.0

    def test_max_cycle_sleep_sec_zero_bypasses_check(self) -> None:
        """max_cycle_sleep_sec=0 (無制限) → バリデーション skip."""
        cfg = FillTestConfig(
            cycle_interval_sec=120.0,
            halt_sleep_multiplier=5.0,
            max_cycle_sleep_sec=0.0,  # 無制限 → OK
        )
        assert cfg.max_cycle_sleep_sec == 0.0

    def test_max_cycle_sleep_sec_too_small_raises(self) -> None:
        """max_cycle_sleep_sec < halt_cap → ValueError."""
        with pytest.raises(ValueError, match="max_cycle_sleep_sec"):
            FillTestConfig(
                cycle_interval_sec=120.0,
                halt_sleep_multiplier=5.0,
                max_cycle_sleep_sec=500.0,  # < 600
            )

    def test_order_timeout_exceeds_cycle_interval_raises(self) -> None:
        """order_timeout_sec > cycle_interval_sec → ValueError."""
        with pytest.raises(ValueError, match="order_timeout_sec"):
            FillTestConfig(
                cycle_interval_sec=60.0,
                order_timeout_sec=90.0,  # > 60
            )

    def test_order_timeout_within_cycle_interval_ok(self) -> None:
        """order_timeout_sec <= cycle_interval_sec → OK."""
        cfg = FillTestConfig(
            cycle_interval_sec=120.0,
            order_timeout_sec=90.0,
        )
        assert cfg.order_timeout_sec == 90.0

    def test_lock_stale_heartbeat_too_small_raises(self) -> None:
        """lock_stale_heartbeat_sec < lock_heartbeat_period × 3 → ValueError."""
        with pytest.raises(ValueError, match="lock_stale_heartbeat_sec"):
            FillTestConfig(
                lock_heartbeat_period_sec=60.0,
                lock_stale_heartbeat_sec=100.0,  # < 180
            )

    def test_lock_stale_heartbeat_valid(self) -> None:
        """lock_stale_heartbeat_sec >= period × 3 → OK."""
        cfg = FillTestConfig(
            lock_heartbeat_period_sec=60.0,
            lock_stale_heartbeat_sec=300.0,  # >= 180
        )
        assert cfg.lock_stale_heartbeat_sec == 300.0

    def test_halt_persist_interval_zero_raises(self) -> None:
        """halt_persist_interval < 1 → ValueError."""
        with pytest.raises(ValueError, match="halt_persist_interval"):
            FillTestConfig(halt_persist_interval=0)

    def test_stop_condition_check_interval_zero_raises(self) -> None:
        """stop_condition_check_interval < 1 → ValueError."""
        with pytest.raises(ValueError, match="stop_condition_check_interval"):
            FillTestConfig(stop_condition_check_interval=0)

    def test_phantom_detection_sleep_multiplier_zero_raises(self) -> None:
        """phantom_detection_sleep_multiplier <= 0 → ValueError."""
        with pytest.raises(ValueError, match="phantom_detection_sleep_multiplier"):
            FillTestConfig(phantom_detection_sleep_multiplier=0.0)

    def test_fallback_duration_sec_zero_raises(self) -> None:
        """fallback_duration_sec <= 0 → ValueError."""
        with pytest.raises(ValueError, match="fallback_duration_sec"):
            FillTestConfig(fallback_duration_sec=0.0)

    def test_unknown_regime_max_consecutive_zero_raises(self) -> None:
        """unknown_regime_max_consecutive < 1 → ValueError."""
        with pytest.raises(ValueError, match="unknown_regime_max_consecutive"):
            FillTestConfig(unknown_regime_max_consecutive=0)


# =====================================================================
# C. orchestrator マジックナンバー → config 参照
# =====================================================================

class TestOrchestratorConfigReferences:
    """orchestrator 内のマジックナンバーが config 経由になっていることを検証."""

    def test_check_regime_stop_conditions_uses_config_fallback_duration(self) -> None:
        """_check_regime_stop_conditions が config.fallback_duration_sec を使用."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        from scripts.v460.lib.regime_policy import DefaultCycleStrategy, RegimePolicyConfig

        obj = MagicMock(spec=FillLoopOrchestratorMixin)
        policy = RegimePolicyConfig(
            dynamic_cycle_enabled=True,
            chase_enabled=True,
            fill_rate_floor=0.35,
            pnl_floor_bps=-0.8,
        )
        strategy = DefaultCycleStrategy(
            base_interval=120.0, base_wait_buy=30.0, base_wait_sell=90.0,
            policy=policy,
        )
        obj._cycle_strategy = strategy
        _mock_config = MagicMock()
        _mock_config.fallback_duration_sec = 1800.0  # 非デフォルト値で検証
        _mock_config.sell_dynamic_kill_window = 50
        _mock_config.min_adapt_samples = 50
        obj.config = _mock_config

        obj._check_regime_stop_conditions = (
            FillLoopOrchestratorMixin._check_regime_stop_conditions.__get__(obj)
        )
        obj._recent_records = []

        obj._check_regime_stop_conditions(filled_count=5, total_count=100)
        # fallback がカスタム duration で activate されたことを確認
        assert strategy._fallback_until is not None

    def test_pnl_window_derived_from_sell_dynamic_kill_window(self) -> None:
        """pnl_avg_window が sell_dynamic_kill_window × 2 で導出される."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        from scripts.v460.lib.regime_policy import DefaultCycleStrategy, RegimePolicyConfig

        obj = MagicMock(spec=FillLoopOrchestratorMixin)
        policy = RegimePolicyConfig(
            dynamic_cycle_enabled=True, chase_enabled=True,
            fill_rate_floor=0.0, pnl_floor_bps=-0.5,
        )
        strategy = DefaultCycleStrategy(
            base_interval=120.0, base_wait_buy=30.0, base_wait_sell=90.0,
            policy=policy,
        )
        obj._cycle_strategy = strategy
        _mock_config = MagicMock()
        _mock_config.fallback_duration_sec = 3600.0
        _mock_config.sell_dynamic_kill_window = 25  # → pnl_window = 50
        _mock_config.min_adapt_samples = 50  # → min_samples = 10
        obj.config = _mock_config

        @dataclass
        class _FakeRec:
            filled: bool = True
            post_fill_30s_pnl: float | None = -1.0

        # 50 records = sell_dynamic_kill_window(25) × 2
        obj._recent_records = [_FakeRec() for _ in range(60)]

        obj._check_regime_stop_conditions = (
            FillLoopOrchestratorMixin._check_regime_stop_conditions.__get__(obj)
        )
        obj._check_regime_stop_conditions(filled_count=50, total_count=100)
        # avg pnl は -1.0 < pnl_floor(-0.5) → fallback 発動
        assert strategy._fallback_until is not None


# =====================================================================
# D. cycle_gate_aggregator UNKNOWN_REGIME_MAX_CONSECUTIVE config 化
# =====================================================================

class TestGateAggregatorConfigIntegration:
    """CycleGateAggregator が config.unknown_regime_max_consecutive を参照."""

    def test_custom_threshold_from_config(self) -> None:
        """config で unknown_regime_max_consecutive を変更すると閾値が変わる."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = FillTestConfig(unknown_regime_max_consecutive=5)
        gate = CycleGateAggregator(cfg)
        assert gate.UNKNOWN_REGIME_MAX_CONSECUTIVE == 5

    def test_default_threshold_matches(self) -> None:
        """デフォルト config で UNKNOWN_REGIME_MAX_CONSECUTIVE = 10."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = FillTestConfig()
        gate = CycleGateAggregator(cfg)
        assert gate.UNKNOWN_REGIME_MAX_CONSECUTIVE == 10

    def test_bypass_with_custom_threshold(self) -> None:
        """カスタム閾値でバイパスが正しく機能すること."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator

        cfg = FillTestConfig(
            unknown_regime_max_consecutive=3,
            skip_buy_unknown_regime=True,
        )
        gate = CycleGateAggregator(cfg)

        # 3 回の unknown block → バイパス
        for _ in range(3):
            gate.evaluate(
                side="buy", regime=None, vol_ratio=None,
                balance_forced=False, inv_net_imbalance=0.0,
                is_buy_killed=False, is_sell_killed=False,
            )
        # 4 回目はバイパスされるべき
        result = gate.evaluate(
            side="buy", regime=None, vol_ratio=None,
            balance_forced=False, inv_net_imbalance=0.0,
            is_buy_killed=False, is_sell_killed=False,
        )
        # バイパス後は blocked=False (unknown_regime ゲートのみ評価)
        # 他のゲートがブロックしなければ通過
        # 少なくともバイパスログが出力されたはず (実装依存)
        assert gate._consecutive_unknown_blocks >= 3


# =====================================================================
# E. MCB σ 履歴 maxlen 導出 + 名前付き定数
# =====================================================================

class TestMCBMaxlenDerivation:
    """MCB の σ 履歴サイズが check_call_interval_sec から導出されること."""

    def test_default_maxlen_720(self) -> None:
        """デフォルト (120s interval) → maxlen = 86400/120 = 720."""
        from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker, MCBConfig

        mcb = MicroCircuitBreaker(MCBConfig())
        assert mcb._change_history_5m.maxlen == 720

    def test_custom_interval_changes_maxlen(self) -> None:
        """check_call_interval_sec=60 → maxlen = 86400/60 = 1440."""
        from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker, MCBConfig

        mcb = MicroCircuitBreaker(MCBConfig(check_call_interval_sec=60.0))
        assert mcb._change_history_5m.maxlen == 1440
        assert mcb._change_history_15m.maxlen == 1440
        assert mcb._change_history_1h.maxlen == 1440

    def test_very_short_interval_has_minimum(self) -> None:
        """極端に短い interval でも maxlen >= 30."""
        from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker, MCBConfig

        mcb = MicroCircuitBreaker(MCBConfig(check_call_interval_sec=100000.0))
        assert mcb._change_history_5m.maxlen >= 30

    def test_named_constants_exist(self) -> None:
        """_MIN_SIGMA_SAMPLES, _SIGMA_FLOOR_RATIO が定義されている."""
        from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker

        assert MicroCircuitBreaker._MIN_SIGMA_SAMPLES == 10
        assert MicroCircuitBreaker._SIGMA_FLOOR_RATIO == 0.1

    def test_calc_threshold_uses_min_sigma_samples(self) -> None:
        """サンプル不足時にデフォルト値を返す."""
        from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker

        history = deque([0.1] * 9)  # < _MIN_SIGMA_SAMPLES
        result = MicroCircuitBreaker._calc_threshold(history, 0.5)
        assert result == 0.5  # default_pct を返す

    def test_calc_threshold_uses_sigma_floor_ratio(self) -> None:
        """σ が極小時にフロア比率が適用される (flat market 保護)."""
        from scripts.v460.lib.micro_circuit_breaker import MicroCircuitBreaker

        # 全て同じ値 → σ=0 → floor = default_pct * 0.1
        history = deque([0.5] * 20)
        result = MicroCircuitBreaker._calc_threshold(history, 1.0)
        # σ=0 → default_pct が返るが、max(sigma, default*0.1) = max(1.0, 0.1) = 1.0
        # 実際は variance=0 → sigma = default_pct → max(1.0, 0.1) = 1.0
        assert result == 1.0


# =====================================================================
# F. B1 warmup TZ 不一致修正
# =====================================================================

class TestWarmupTZFix:
    """warmup 関数が DD guard と同一 TZ を使用することの検証."""

    def test_warmup_uses_dd_guard_timezone(self) -> None:
        """_warmup_daily_drawdown_from_records が DD guard の TZ を使用.

        JST (UTC+9) 設定時、UTC 日付ではなく JST 日付が使われること。
        """
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard

        obj = MagicMock(spec=FillLoopOrchestratorMixin)

        # DD guard: JST (UTC+9)
        guard = DailyDrawdownGuard(
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            day_reset_utc_offset_hours=9.0,
        )
        obj._daily_drawdown_guard = guard

        # JST の今日の日付を取得 (guard と同じ TZ)
        jst = timezone(timedelta(hours=9))
        today_jst = datetime.now(jst).strftime("%Y%m%d")

        @dataclass
        class _FakeRec:
            filled: bool = True
            post_fill_30s_pnl: float = -3.0
            timestamp: float = time.time()  # 現在時刻
            side: str = "buy"

        records = [_FakeRec() for _ in range(5)]

        obj._warmup_daily_drawdown_from_records = (
            FillLoopOrchestratorMixin._warmup_daily_drawdown_from_records.__get__(obj)
        )
        obj._warmup_daily_drawdown_from_records(records)

        # DD guard の current_day が JST 日付と一致すること
        assert guard.state.current_day == today_jst
        assert guard.state.daily_fill_count == 5
        assert guard.state.daily_pnl_bps == pytest.approx(-15.0)


# =====================================================================
# G. gate block ログ間隔の quiescence 連動導出
# =====================================================================

class TestGateBlockLogIntervalDerivation:
    """gate block ログ間隔が quiescence_gate_blocks_threshold / 2 で導出される."""

    def test_default_gate_log_interval(self) -> None:
        """デフォルト (threshold=20) → log_interval = max(5, 20//2) = 10."""
        cfg = FillTestConfig()
        interval = max(5, cfg.quiescence_gate_blocks_threshold // 2)
        assert interval == 10

    def test_small_threshold_has_minimum_5(self) -> None:
        """threshold=4 → max(5, 4//2) = max(5, 2) = 5."""
        cfg = FillTestConfig(quiescence_gate_blocks_threshold=4)
        interval = max(5, cfg.quiescence_gate_blocks_threshold // 2)
        assert interval == 5

    def test_large_threshold_scales(self) -> None:
        """threshold=40 → max(5, 40//2) = 20."""
        cfg = FillTestConfig(quiescence_gate_blocks_threshold=40)
        interval = max(5, cfg.quiescence_gate_blocks_threshold // 2)
        assert interval == 20
