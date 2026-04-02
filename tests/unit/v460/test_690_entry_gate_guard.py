from __future__ import annotations

import time

from scripts.v460.lib.entry_gate_guard import (
    EntryGateGuard,
    EntryGateGuardConfig,
)
from tests.unit.v460._fill_test_source import ORCHESTRATOR_MID_CYCLE, read_source_text


class TestEntryGateGuard:
    def test_consecutive_block_limit_auto_disables(self) -> None:
        guard = EntryGateGuard(
            EntryGateGuardConfig(
                max_consecutive_blocks=2,
                max_block_rate=0.95,
                min_eval_count_for_rate=20,
                staleness_threshold_sec=600.0,
            )
        )
        guard.notify_calibration_update()

        assert guard.should_suppress_block(ev=-0.1, regime="ranging", side="buy") is False
        guard.record_eval(blocked=True)
        assert guard.should_suppress_block(ev=-0.2, regime="ranging", side="buy") is False
        guard.record_eval(blocked=True)

        assert guard.should_suppress_block(ev=-0.3, regime="ranging", side="buy") is True
        assert guard.state.auto_disabled is True
        assert "consecutive" in guard.state.auto_disable_reason

    def test_block_rate_limit_auto_disables(self) -> None:
        guard = EntryGateGuard(
            EntryGateGuardConfig(
                max_consecutive_blocks=99,
                max_block_rate=0.5,
                min_eval_count_for_rate=4,
                staleness_threshold_sec=600.0,
            )
        )
        guard.notify_calibration_update()

        for blocked in (True, True, False, True):
            guard.record_eval(blocked=blocked)

        assert guard.should_suppress_block(ev=-0.2, regime="strong_up", side="sell") is True
        assert guard.state.auto_disabled is True
        assert "block_rate" in guard.state.auto_disable_reason

    def test_staleness_suppresses_block(self) -> None:
        guard = EntryGateGuard(
            EntryGateGuardConfig(
                max_consecutive_blocks=10,
                max_block_rate=0.9,
                min_eval_count_for_rate=20,
                staleness_threshold_sec=60.0,
            )
        )
        guard.notify_calibration_update()
        guard.state.last_calibration_update_ts = time.time() - 120.0

        assert guard.should_suppress_block(ev=-0.1, regime="weak_up", side="sell") is True
        assert "stale" in guard.state.auto_disable_reason

    def test_positive_eval_resets_consecutive_blocks(self) -> None:
        guard = EntryGateGuard(EntryGateGuardConfig(max_consecutive_blocks=3))
        guard.notify_calibration_update()

        guard.record_eval(blocked=True)
        guard.record_eval(blocked=True)
        assert guard.state.consecutive_blocks == 2

        guard.record_eval(blocked=False)
        assert guard.state.consecutive_blocks == 0

    def test_reset_auto_disable_clears_guard_state(self) -> None:
        guard = EntryGateGuard(EntryGateGuardConfig(max_consecutive_blocks=1))
        guard.notify_calibration_update()
        guard.record_eval(blocked=True)
        assert guard.should_suppress_block(ev=-0.1, regime="unknown", side="buy") is True

        last_update = guard.state.last_calibration_update_ts
        guard.reset_auto_disable()

        assert guard.state.auto_disabled is False
        assert guard.state.auto_disable_reason == ""
        assert guard.state.consecutive_blocks == 0
        assert guard.state.total_blocks == 0
        assert guard.state.total_evals == 0
        assert guard.state.last_calibration_update_ts == last_update

    def test_missing_calibration_update_is_treated_as_stale(self) -> None:
        guard = EntryGateGuard(EntryGateGuardConfig(staleness_threshold_sec=60.0))

        assert guard.should_suppress_block(ev=-0.1, regime="ranging", side="buy") is True
        assert "stale" in guard.state.auto_disable_reason


class TestEntryGateSourceAudit:
    def test_n_eff_guard_forces_neutral_probability(self) -> None:
        source = read_source_text(ORCHESTRATOR_MID_CYCLE)

        assert "_n_eff < self.config.entry_gate_n_min" in source
        assert "_p_win = 0.5" in source

