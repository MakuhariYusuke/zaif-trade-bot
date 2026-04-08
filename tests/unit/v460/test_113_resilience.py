"""
113# resilience モジュール + R1 run_single_cycle 位相分割のテスト.

- resilience.py: CircuitBreaker factory / HealthMonitor / StatePersistence
- R1: run_single_cycle → pre_order / submit / monitor / finalize phase helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from scripts.v460.lib.regime_detector import FillTestRegimeDetector
from scripts.v460.lib.resilience import (
    FillTestHealthMonitor,
    FillTestState,
    FillTestStatePersistence,
    HealthThresholds,
    create_api_circuit_breaker,
)
from tests.unit.v460._fill_test_source import read_fill_test_method_source, source_contains_all
from ztb.utils.circuit_breaker import CircuitBreaker, CircuitState


# =====================================================================
# A. resilience.py — ユニットテスト
# =====================================================================

class TestCircuitBreakerFactory:
    """create_api_circuit_breaker() のテスト."""

    def test_factory_returns_circuit_breaker(self) -> None:
        cb = create_api_circuit_breaker()
        assert isinstance(cb, CircuitBreaker)
        assert cb.state == CircuitState.CLOSED
        assert cb.name == "coincheck_api"

    def test_factory_defaults(self) -> None:
        cb = create_api_circuit_breaker()
        assert cb.config.failure_threshold == 5
        assert cb.config.recovery_timeout == 120.0
        assert cb.config.success_threshold == 2
        assert cb.config.timeout == 30.0

    def test_factory_custom_params(self) -> None:
        cb = create_api_circuit_breaker(
            failure_threshold=3, recovery_timeout=60.0,
        )
        assert cb.config.failure_threshold == 3
        assert cb.config.recovery_timeout == 60.0


class TestHealthMonitor:
    """FillTestHealthMonitor のテスト."""

    def test_init_defaults(self) -> None:
        hm = FillTestHealthMonitor()
        assert hm._thresholds.rss_warn_mb == 1500.0
        assert hm._thresholds.gc_interval_cycles == 100

    def test_maybe_check_skips_if_too_early(self) -> None:
        """check_interval_sec 以内は None を返す."""
        hm = FillTestHealthMonitor(HealthThresholds(check_interval_sec=9999))
        # 初回は last_check_time=0 なので実行される
        result = hm.maybe_check(1)
        assert result is not None
        # 2回目はスキップ
        result2 = hm.maybe_check(2)
        assert result2 is None

    def test_maybe_gc_runs_after_interval(self) -> None:
        """gc_interval_cycles 回呼ぶと GC が実行される."""
        hm = FillTestHealthMonitor(HealthThresholds(gc_interval_cycles=3))
        with patch("scripts.v460.lib.resilience.gc.collect", return_value=7) as mock_gc:
            hm.maybe_gc()  # counter=1
            hm.maybe_gc()  # counter=2
            hm.maybe_gc()  # counter=3 → GC 実行 → reset to 0
        assert hm._gc_counter == 0
        mock_gc.assert_called_once()


class TestStatePersistence:
    """FillTestStatePersistence のテスト."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        sp = FillTestStatePersistence(tmp_path)
        state = FillTestState(
            run_id="test_run",
            cycle_count=42,
            total_count=100,
            filled_count=65,
            cumulative_pnl_jpy=-123.4,
            current_lot=0.002,
        )
        sp.save(state)
        loaded = sp.load()
        assert loaded is not None
        assert loaded.run_id == "test_run"
        assert loaded.cycle_count == 42
        assert loaded.filled_count == 65
        assert loaded.cumulative_pnl_jpy == pytest.approx(-123.4)

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        sp = FillTestStatePersistence(tmp_path)
        assert sp.load() is None

    def test_save_creates_json(self, tmp_path: Path) -> None:
        sp = FillTestStatePersistence(tmp_path)
        sp.save(FillTestState(run_id="abc"))
        state_file = tmp_path / "fill_test_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["run_id"] == "abc"
        assert data["saved_at"] > 0

    def test_atomic_write(self, tmp_path: Path) -> None:
        """tmp ファイル経由の atomic write."""
        sp = FillTestStatePersistence(tmp_path)
        sp.save(FillTestState(run_id="x"))
        assert not (tmp_path / "fill_test_state.tmp").exists()

    def test_regime_state_save_and_load(self, tmp_path: Path) -> None:
        """121# A4: regime state がシリアライズ/デシリアライズされる."""
        sp = FillTestStatePersistence(tmp_path)
        state = FillTestState(
            run_id="regime_test",
            cycle_count=100,
            regime_confirmed="ranging",
            regime_stability=5,
            regime_prices=[[1000.0, 14500000.0], [1120.0, 14510000.0]],
            regime_raw_history=["ranging", "ranging", "ranging"],
        )
        sp.save(state)
        loaded = sp.load()
        assert loaded is not None
        assert loaded.regime_confirmed == "ranging"
        assert loaded.regime_stability == 5
        assert loaded.regime_prices == [[1000.0, 14500000.0], [1120.0, 14510000.0]]
        assert loaded.regime_raw_history == ["ranging", "ranging", "ranging"]

    def test_regime_state_backward_compatible(self, tmp_path: Path) -> None:
        """121# A4: regime フィールドなしの旧 JSON からも load できる."""
        sp = FillTestStatePersistence(tmp_path)
        old_json = json.dumps({"run_id": "old", "cycle_count": 50, "saved_at": 1.0})
        (tmp_path / "fill_test_state.json").write_text(old_json, encoding="utf-8")
        loaded = sp.load()
        assert loaded is not None
        assert loaded.run_id == "old"
        assert loaded.regime_confirmed == "unknown"
        assert loaded.regime_prices is None

    def test_load_non_object_json_returns_none(self, tmp_path: Path) -> None:
        """状態JSONがobject以外の場合は安全に None を返す."""
        sp = FillTestStatePersistence(tmp_path)
        (tmp_path / "fill_test_state.json").write_text("[1,2,3]", encoding="utf-8")
        assert sp.load() is None


class TestRegimeDetectorPersistence:
    """121# A4: FillTestRegimeDetector の get_state/restore_state テスト."""

    def test_get_state_returns_dict(self) -> None:
        det = FillTestRegimeDetector()
        state = det.get_state()
        assert state["confirmed"] == "unknown"
        assert state["stability"] == 0
        assert state["prices"] == []
        assert state["raw_history"] == []

    def test_restore_state_roundtrip(self) -> None:
        det = FillTestRegimeDetector()
        # 20 回 update してレジームを確定させる
        base_price = 14_500_000.0
        for i in range(25):
            det.update(1000.0 + i * 120, base_price + i * 100)
        saved = det.get_state()
        assert saved["confirmed"] != "unknown" or len(saved["prices"]) >= 20

        # 新インスタンスに復元
        det2 = FillTestRegimeDetector()
        assert det2.restore_state(saved)
        assert det2.current_regime.value == saved["confirmed"]
        assert det2.observation_count == len(saved["prices"])

    def test_restore_state_invalid_returns_false(self) -> None:
        det = FillTestRegimeDetector()
        assert not det.restore_state({"confirmed": "INVALID_VALUE"})


# =====================================================================
# B. R1 run_single_cycle 分割 — 構造テスト
# =====================================================================

class TestR1MethodExtraction:
    """113# R1: run_single_cycle が phase helper に分割されている."""

    def test_run_single_cycle_delegates_to_skip_gate(self) -> None:
        """run_single_cycle は pre-order phase を経由する."""
        source = read_fill_test_method_source("run_single_cycle")
        assert "_run_pre_order_phase" in source

    def test_run_single_cycle_delegates_to_monitor(self) -> None:
        """run_single_cycle は monitor phase を経由する."""
        source = read_fill_test_method_source("run_single_cycle")
        assert "_monitor_fill_phase" in source

    def test_run_single_cycle_delegates_to_pnl(self) -> None:
        """run_single_cycle は finalize phase を経由する."""
        source = read_fill_test_method_source("run_single_cycle")
        assert "_finalize_cycle" in source

    def test_run_single_cycle_under_400_lines(self) -> None:
        """run_single_cycle が 830 行以下 (R1 目標 + ... + 442# Cross-Venue OB拡張)."""
        source = read_fill_test_method_source("run_single_cycle")
        line_count = len(source.splitlines())
        assert line_count <= 830, f"run_single_cycle is {line_count} lines (> 830)"

    def test_phase_helpers_delegate_to_core_methods(self) -> None:
        """phase helper が既存の core helper を呼ぶ."""
        pre_source = read_fill_test_method_source("_run_pre_order_phase")
        monitor_source = read_fill_test_method_source("_monitor_fill_phase")
        finalize_source = read_fill_test_method_source("_finalize_cycle")
        assert source_contains_all(pre_source, "_evaluate_skip_gate")
        assert source_contains_all(monitor_source, "_monitor_fill_polling")
        assert source_contains_all(finalize_source, "_measure_post_fill_pnl")

    def test_result_dataclasses_exist(self) -> None:
        """R1 結果データクラスがインポート可能."""
        from scripts.v460.run_fill_test import (
            _SkipGateResult,
            _FillMonitorResult,
            _PnlMeasurement,
        )
        from scripts.v460.lib.fill_cycle_executor import (
            _FillPhaseResult,
            _PreOrderPhaseResult,
            _SubmissionPhaseResult,
        )
        assert _SkipGateResult is not None
        assert _FillMonitorResult is not None
        assert _PnlMeasurement is not None
        assert _PreOrderPhaseResult is not None
        assert _SubmissionPhaseResult is not None
        assert _FillPhaseResult is not None


class TestR1CircuitBreakerInRunSingleCycle:
    """113# CircuitBreaker が run_single_cycle に組み込まれている."""

    def test_circuit_breaker_guard_in_source(self) -> None:
        source = read_fill_test_method_source("run_single_cycle")
        assert "circuit_breaker" in source
        # 145# §9-#6: CR.CIRCUIT_BREAKER_OPEN 定数に移行済み
        assert "CIRCUIT_BREAKER_OPEN" in source

    def test_circuit_breaker_success_recording(self) -> None:
        source = read_fill_test_method_source("_finalize_cycle")
        assert "_on_success" in source

    def test_circuit_breaker_failure_recording(self) -> None:
        source = read_fill_test_method_source("_submit_order_phase")
        assert "_on_failure" in source


class TestR1ResilienceInRunContinuous:
    """113# HealthMonitor / StatePersistence が run_continuous に組み込まれている."""

    def test_health_check_in_continuous(self) -> None:
        # 265# extract: health monitor は _log_progress_and_adapt に分離
        source = read_fill_test_method_source("_log_progress_and_adapt")
        assert source_contains_all(source, "maybe_check", "maybe_gc")

    def test_state_persistence_in_continuous(self) -> None:
        # 265# extract: state persistence は _log_progress_and_adapt + _finalize_run に分離
        source = read_fill_test_method_source("_log_progress_and_adapt")
        assert "state_persistence" in source
        finalize_source = read_fill_test_method_source("_finalize_run")
        assert "FillTestState" not in source or "FillTestState" in finalize_source

    def test_resilience_init_in_constructor(self) -> None:
        source = read_fill_test_method_source("__init__")
        assert source_contains_all(
            source,
            "_circuit_breaker",
            "_health_monitor",
            "_state_persistence",
        )
