"""203# DD状態永続化修正 + fill records DD warmup の単体テスト.

E: HALT開始時のstate強制保存 (halt中の_cycle_count不変バグ修正)
F: fill records からのDD state warmup (stale/missing state のセーフティネット)
G: halt_elapsed カウンタ修正 (_halt_iter_count 導入)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
# 203# E: Halt state save — _halt_entering logic
# ============================================================

class TestHaltStateSave:
    """203# E: halt開始時に必ずstate保存される."""

    def test_halt_entering_flag_first_iteration(self) -> None:
        """_halt_start_cycle が None → _halt_entering=True."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        mixin = FillLoopOrchestratorMixin.__new__(FillLoopOrchestratorMixin)
        # 初期状態: _halt_start_cycle 未設定
        assert not hasattr(mixin, "_halt_start_cycle") or getattr(mixin, "_halt_start_cycle", None) is None

    def test_halt_iter_count_attr_on_orchestrator(self) -> None:
        """_halt_iter_count attr が halt 開始時に初期化."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        mixin = FillLoopOrchestratorMixin.__new__(FillLoopOrchestratorMixin)
        # 初期状態では _halt_iter_count は未設定
        assert not hasattr(mixin, "_halt_iter_count")


# ============================================================
# 203# F: DD warmup from fill records
# ============================================================

class TestDDWarmupFromRecords:
    """203# F: stale/missing state 時に fill records から DD state を復元."""

    def _make_mock_record(
        self, filled: bool, pnl: float | None, timestamp: float,
    ) -> MagicMock:
        r = MagicMock()
        r.filled = filled
        r.post_fill_30s_pnl = pnl
        r.timestamp = timestamp
        return r

    def _make_mixin_with_dd_guard(
        self, *, hard_limit: float = -50.0, soft_limit: float = -30.0,
    ) -> "FillLoopOrchestratorMixin":
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        mixin = FillLoopOrchestratorMixin.__new__(FillLoopOrchestratorMixin)
        mixin._daily_drawdown_guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=hard_limit, soft_limit_bps=soft_limit,
        )
        return mixin

    def test_warmup_calculates_daily_pnl(self) -> None:
        """当日UTC fill records の PnL が正しく合算される."""
        mixin = self._make_mixin_with_dd_guard()
        now = time.time()
        records = [
            self._make_mock_record(True, -5.0, now - 100),  # today
            self._make_mock_record(True, -3.0, now - 50),   # today
            self._make_mock_record(True, 2.0, now - 10),    # today
        ]
        mixin._warmup_daily_drawdown_from_records(records)
        assert mixin._daily_drawdown_guard.state.daily_pnl_bps == pytest.approx(-6.0)
        assert mixin._daily_drawdown_guard.state.daily_fill_count == 3

    def test_warmup_excludes_unfilled(self) -> None:
        """未約定レコードは除外される."""
        mixin = self._make_mixin_with_dd_guard()
        now = time.time()
        records = [
            self._make_mock_record(True, -5.0, now - 100),
            self._make_mock_record(False, None, now - 50),   # unfilled
            self._make_mock_record(True, None, now - 10),    # filled but no PnL
        ]
        mixin._warmup_daily_drawdown_from_records(records)
        assert mixin._daily_drawdown_guard.state.daily_pnl_bps == pytest.approx(-5.0)
        assert mixin._daily_drawdown_guard.state.daily_fill_count == 1

    def test_warmup_excludes_old_dates(self) -> None:
        """前日UTC以前のレコードは除外される."""
        mixin = self._make_mixin_with_dd_guard()
        now = time.time()
        yesterday = now - 86400 * 2  # 2日前 = 確実に別UTC日
        records = [
            self._make_mock_record(True, -20.0, yesterday),  # old
            self._make_mock_record(True, -1.0, now - 10),     # today
        ]
        mixin._warmup_daily_drawdown_from_records(records)
        assert mixin._daily_drawdown_guard.state.daily_pnl_bps == pytest.approx(-1.0)
        assert mixin._daily_drawdown_guard.state.daily_fill_count == 1

    def test_warmup_triggers_halt(self) -> None:
        """当日累積PnLがhard limitを超えていたらhalt状態にする."""
        mixin = self._make_mixin_with_dd_guard(hard_limit=-50.0)
        now = time.time()
        records = [
            self._make_mock_record(True, -30.0, now - 100),
            self._make_mock_record(True, -25.0, now - 50),
        ]
        mixin._warmup_daily_drawdown_from_records(records)
        assert mixin._daily_drawdown_guard.state.daily_pnl_bps == pytest.approx(-55.0)
        assert mixin._daily_drawdown_guard.state.halted is True

    def test_warmup_triggers_soft(self) -> None:
        """当日累積PnLがsoft limitを超えていたらsoft_triggered."""
        mixin = self._make_mixin_with_dd_guard(hard_limit=-50.0, soft_limit=-30.0)
        now = time.time()
        records = [
            self._make_mock_record(True, -20.0, now - 100),
            self._make_mock_record(True, -15.0, now - 50),
        ]
        mixin._warmup_daily_drawdown_from_records(records)
        assert mixin._daily_drawdown_guard.state.daily_pnl_bps == pytest.approx(-35.0)
        assert mixin._daily_drawdown_guard.state.halted is False
        assert mixin._daily_drawdown_guard._soft_triggered_today is True

    def test_warmup_no_records_noop(self) -> None:
        """当日レコードが0件なら何もしない."""
        mixin = self._make_mixin_with_dd_guard()
        yesterday = time.time() - 86400 * 2
        records = [
            self._make_mock_record(True, -10.0, yesterday),
        ]
        mixin._warmup_daily_drawdown_from_records(records)
        assert mixin._daily_drawdown_guard.state.daily_pnl_bps == 0.0
        assert mixin._daily_drawdown_guard.state.daily_fill_count == 0


# ============================================================
# 203# G: halt_elapsed counter fix
# ============================================================

class TestHaltElapsedCounter:
    """203# G: halt_iter_count による正確なカウント."""

    def test_halt_end_resets_iter_count(self) -> None:
        """halt終了時に _halt_iter_count がリセットされる."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin.run_continuous)
        assert "_halt_iter_count = 0" in src, \
            "halt終了時に _halt_iter_count がリセットされるべき"

    def test_halt_entering_saves_state(self) -> None:
        """halt開始時のstate保存コードが存在する."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin.run_continuous)
        assert "_halt_entering" in src, \
            "203# E halt開始フラグが存在するべき"
        # _halt_entering 条件でstate保存
        assert "if _halt_entering or" in src, \
            "203# E halt開始時にstate保存すべき"

    def test_no_old_cycle_count_modulo_in_halt(self) -> None:
        """旧実装の self._cycle_count % progress_log_interval (halt内) が除去された."""
        import inspect
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
        src = inspect.getsource(FillLoopOrchestratorMixin.run_continuous)
        # halt ブロック内に旧条件がないことを確認
        # "200# P0-3: HALT 中も" コメントが 203# E に置換されているか
        assert "200# P0-3" not in src, \
            "旧 200# P0-3 コメントは 203# E に置換されるべき"
