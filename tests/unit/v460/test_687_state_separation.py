from __future__ import annotations

import asyncio

from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin, RunSessionState
from scripts.v460.lib.side_selector import SideSelector
from scripts.v460.run_fill_test import FillTestConfig
from ztb.metrics.fill_quality import FillRecord, build_fill_record, build_skip_fill_record


class _BatchPersistenceStub:
    @staticmethod
    def maybe_flush(batch: list[FillRecord], _context: str) -> list[FillRecord]:
        return batch


class _SkipRunnerStub:
    def __init__(self) -> None:
        self._side_selector = SideSelector(FillTestConfig(enable_regime=False))
        self._batch_persistence = _BatchPersistenceStub()
        self.heartbeat_count = 0
        self.state_save_count = 0
        self.sleep_count = 0

    def _make_loop_skip_record(self, *, side: str, cancel_reason: str, order_quantity: float = 0.0, **_: object) -> FillRecord:
        return build_skip_fill_record(
            cycle_id="skip",
            timestamp=1.0,
            side=side,
            order_price=0.0,
            order_quantity=order_quantity,
            cancel_reason=cancel_reason,
            run_id=None,
            git_sha=None,
            last_executed_side=self._side_selector.last_executed_side,
            last_attempted_side=self._side_selector.last_attempted_side,
        )

    def _update_lock_heartbeat(self) -> None:
        self.heartbeat_count += 1

    def _maybe_skip_state_save(self, _st: RunSessionState, _context: str) -> None:
        self.state_save_count += 1

    async def _effective_sleep(self, *, multiplier: float = 1.0, max_override: float = 0.0) -> None:
        _ = (multiplier, max_override)
        self.sleep_count += 1


def _make_selector(*, start_side: str = "buy") -> SideSelector:
    return SideSelector(FillTestConfig(enable_regime=False, start_side=start_side))


class TestStateSeparation:
    def test_fill_success_updates_executed_and_attempted(self) -> None:
        selector = _make_selector()

        selector.update_after_decision("buy")

        assert selector.last_executed_side == "buy"
        assert selector.last_attempted_side == "buy"
        assert selector.next() == "sell"

    def test_attempt_only_keeps_last_executed_side_unchanged(self) -> None:
        selector = _make_selector()

        selector.update_after_attempt("buy")

        assert selector.last_executed_side is None
        assert selector.last_attempted_side == "buy"
        assert selector.next() == "buy"

    def test_preflight_like_attempt_does_not_change_executed_alternation(self) -> None:
        selector = _make_selector()
        selector.update_after_decision("buy")

        selector.update_after_attempt("sell")

        assert selector.last_executed_side == "buy"
        assert selector.last_attempted_side == "sell"
        assert selector.next() == "sell"

    def test_execute_skip_updates_attempted_only(self) -> None:
        runner = _SkipRunnerStub()
        runner._side_selector.update_after_decision("sell")
        state = RunSessionState()

        asyncio.run(
            FillLoopOrchestratorMixin._execute_skip(
                runner,
                state,
                side="buy",
                cancel_reason="preflight_insufficient",
                update_last_side=True,
                sleep=False,
            )
        )

        assert runner._side_selector.last_executed_side == "sell"
        assert runner._side_selector.last_attempted_side == "buy"

    def test_fill_record_roundtrip_preserves_side_states(self) -> None:
        record = build_fill_record(
            cycle_id="fill",
            timestamp=1.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.001,
            last_executed_side="buy",
            last_attempted_side="sell",
        )

        restored = FillRecord.from_dict(record.to_dict())

        assert restored.last_executed_side == "buy"
        assert restored.last_attempted_side == "sell"

    def test_skip_record_roundtrip_preserves_side_states(self) -> None:
        record = build_skip_fill_record(
            cycle_id="skip",
            timestamp=1.0,
            side="sell",
            order_price=100.0,
            order_quantity=0.001,
            cancel_reason="no_feasible_quote",
            run_id=None,
            git_sha=None,
            last_executed_side="buy",
            last_attempted_side="sell",
        )

        restored = FillRecord.from_dict(record.to_dict())

        assert restored.last_executed_side == "buy"
        assert restored.last_attempted_side == "sell"
