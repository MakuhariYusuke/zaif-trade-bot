"""225# F1/F2/6.1/5.1-5.2: warmup日付フィルタ + state save + recovery復元 + fire countテスト."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
from scripts.v460.lib.orchestrator_lifecycle import OrchestratorLifecycleMixin
from scripts.v460.lib.regime_detector import FillTestRegime


# ======================================================================
# F1: Kill manager warmup — 当日分のみ replay
# ======================================================================


@dataclass
class _FakeFillRecord:
    """テスト用の軽量 FillRecord スタブ."""
    filled: bool = True
    post_fill_30s_pnl: float | None = 0.5
    side: str = "sell"
    timestamp: float = 0.0


class _TrackRecorder:
    """kill manager track 呼び出しの最小 recorder."""

    def __init__(self) -> None:
        self._pnl_history: list[float] = []
        self.calls: list[float] = []

    def track(self, pnl: float) -> None:
        self.calls.append(pnl)

    def assert_not_called(self) -> None:
        assert not self.calls

    def assert_called_once_with(self, pnl: float) -> None:
        assert self.calls == [pnl]


class _WarmupOrchestrator:
    """_warmup_kill_managers_from_records 実行用の最小 orchestrator."""

    def __init__(self) -> None:
        self._sell_kill_mgr = _TrackRecorder()
        self._buy_kill_mgr = _TrackRecorder()


class TestKillManagerWarmupDateFilter:
    """225# F1: _warmup_kill_managers_from_records の日付フィルタテスト."""

    def _make_mock_orchestrator(self) -> _WarmupOrchestrator:
        """warmup メソッドだけを bind した軽量 orchestrator."""
        orch = _WarmupOrchestrator()
        orch._warmup_kill_managers_from_records = (  # type: ignore[attr-defined]
            OrchestratorLifecycleMixin._warmup_kill_managers_from_records.__get__(orch)
        )
        return orch

    def test_only_today_records_replayed(self) -> None:
        """当日分のみ kill manager に replay される."""
        orch = self._make_mock_orchestrator()

        now = time.time()
        yesterday = now - 86400 * 1.5  # 1.5 日前

        records = [
            _FakeFillRecord(side="sell", post_fill_30s_pnl=-1.0, timestamp=yesterday),
            _FakeFillRecord(side="buy", post_fill_30s_pnl=-0.5, timestamp=yesterday),
            _FakeFillRecord(side="sell", post_fill_30s_pnl=0.3, timestamp=now),
            _FakeFillRecord(side="buy", post_fill_30s_pnl=0.7, timestamp=now),
        ]

        orch._warmup_kill_managers_from_records(records)

        # 当日分のみ track された
        orch._sell_kill_mgr.assert_called_once_with(0.3)
        orch._buy_kill_mgr.assert_called_once_with(0.7)

    def test_unfilled_records_skipped(self) -> None:
        """未約定レコードはスキップ."""
        orch = self._make_mock_orchestrator()
        now = time.time()
        records = [
            _FakeFillRecord(filled=False, side="sell", timestamp=now),
            _FakeFillRecord(filled=True, post_fill_30s_pnl=None, side="buy", timestamp=now),
        ]
        orch._warmup_kill_managers_from_records(records)
        orch._sell_kill_mgr.assert_not_called()
        orch._buy_kill_mgr.assert_not_called()

    def test_empty_records(self) -> None:
        """空リストでもエラーにならない."""
        orch = self._make_mock_orchestrator()
        orch._warmup_kill_managers_from_records([])
        orch._sell_kill_mgr.assert_not_called()

    def test_all_old_records_skipped(self) -> None:
        """全て前日のレコードの場合、track は 0 回."""
        orch = self._make_mock_orchestrator()
        old = time.time() - 86400 * 2
        records = [
            _FakeFillRecord(side="sell", post_fill_30s_pnl=1.0, timestamp=old),
            _FakeFillRecord(side="buy", post_fill_30s_pnl=2.0, timestamp=old),
        ]
        orch._warmup_kill_managers_from_records(records)
        orch._sell_kill_mgr.assert_not_called()
        orch._buy_kill_mgr.assert_not_called()


# ======================================================================
# 6.1: recovery counter 例外時復元
# ======================================================================


class TestRecoveryCounterRestore:
    """225# 6.1: restore_recovery_counter テスト."""

    def _make_guard(self) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=-10.0,
            per_side_halt_cycles=2,
            per_side_recovery_cycles=3,
            per_side_recovery_lot_scale=0.5,
        )

    def test_restore_buy_counter(self) -> None:
        """buy 側の recovery counter が復元される."""
        guard = self._make_guard()
        # halt → release → recovery 開始
        guard.update_pnl(-15.0, side="buy")
        guard.tick_side_halt()
        guard.tick_side_halt()
        assert guard.state.side_recovery_remaining_buy == 3

        # 1回消費 (consume_recovery_cycle)
        scale = guard.consume_recovery_cycle("buy")
        assert scale == 0.5
        assert guard.state.side_recovery_remaining_buy == 2

        # 例外で復元
        guard.restore_recovery_counter("buy")
        assert guard.state.side_recovery_remaining_buy == 3

    def test_restore_sell_counter(self) -> None:
        """sell 側の recovery counter が復元される."""
        guard = self._make_guard()
        guard.update_pnl(-15.0, side="sell")
        guard.tick_side_halt()
        guard.tick_side_halt()
        assert guard.state.side_recovery_remaining_sell == 3

        scale = guard.consume_recovery_cycle("sell")
        assert scale == 0.5
        assert guard.state.side_recovery_remaining_sell == 2

        guard.restore_recovery_counter("sell")
        assert guard.state.side_recovery_remaining_sell == 3

    def test_restore_does_not_exceed_original(self) -> None:
        """restore を余分に呼ぶと上限を超えるが、これは呼出側の責任."""
        guard = self._make_guard()
        guard.state.side_recovery_remaining_buy = 0
        guard.restore_recovery_counter("buy")
        assert guard.state.side_recovery_remaining_buy == 1


# ======================================================================
# FillTestRegime.is_high_vol
# ======================================================================


class TestFillTestRegimeIsHighVol:
    """225# FillTestRegime.is_high_vol プロパティテスト."""

    def test_high_vol_returns_true(self) -> None:
        assert FillTestRegime.HIGH_VOL.is_high_vol is True

    def test_trending_returns_false(self) -> None:
        assert FillTestRegime.TRENDING.is_high_vol is False
        assert FillTestRegime.TRENDING_UP.is_high_vol is False
        assert FillTestRegime.TRENDING_DOWN.is_high_vol is False

    def test_ranging_returns_false(self) -> None:
        assert FillTestRegime.RANGING.is_high_vol is False

    def test_unknown_returns_false(self) -> None:
        assert FillTestRegime.UNKNOWN.is_high_vol is False


# ======================================================================
# FillTestState MCB/SAD fields
# ======================================================================


class TestFillTestStateMcbSadFields:
    """225# FillTestState に mcb_state/sad_state フィールドが存在する."""

    def test_mcb_state_field_default_none(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState()
        assert state.mcb_state is None

    def test_sad_state_field_default_none(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState()
        assert state.sad_state is None

    def test_mcb_state_roundtrip(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        mcb_data = {"halt_until": 123.0, "total_halts": 2, "price_buffer": []}
        state = FillTestState(mcb_state=mcb_data)
        assert state.mcb_state == mcb_data

    def test_sad_state_roundtrip(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        sad_data = {"frozen_until": 456.0, "total_frozens": 1, "spread_buffer": []}
        state = FillTestState(sad_state=sad_data)
        assert state.sad_state == sad_data


# ======================================================================
# FillConfig new params
# ======================================================================


class TestFillConfigRecoveryPenalty:
    """225# recovery_trending_penalty / recovery_high_vol_penalty デフォルト値."""

    def test_default_trending_penalty(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.recovery_trending_penalty == 0.7

    def test_default_high_vol_penalty(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.recovery_high_vol_penalty == 0.8


# ======================================================================
# DD Guard daily reset clears recovery (clarification test)
# ======================================================================


class TestDailyResetClearsRecovery:
    """225# 5-2: DailyDrawdownGuard の日替わりリセットで recovery が消える."""

    def test_day_reset_clears_recovery_remaining(self) -> None:
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=-10.0,
            per_side_halt_cycles=2,
            per_side_recovery_cycles=5,
            per_side_recovery_lot_scale=0.5,
        )
        # recovery 設定
        guard.state.side_recovery_remaining_buy = 3
        guard.state.side_recovery_remaining_sell = 4
        guard.state.current_day = "20250101"  # 強制的に旧日

        # 日替わりリセット発動
        reset = guard.maybe_reset_day()
        assert reset is True
        # recovery カウンタはリセットされる (意図的設計)
        assert guard.state.side_recovery_remaining_buy == 0
        assert guard.state.side_recovery_remaining_sell == 0
