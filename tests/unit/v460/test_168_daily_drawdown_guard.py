"""168# §4.1 #3: DailyDrawdownGuard ユニットテスト.

テスト対象:
- DailyDrawdownGuard クラス (日次 PnL 追跡、soft/hard 二段制御)
- cancel_reasons.DAILY_DRAWDOWN_HALT 定数
- FillTestConfig の daily_drawdown_* フィールド
- FillTestState の daily_drawdown_state フィールド
- State export/import (永続化)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from scripts.v460.lib.daily_drawdown_guard import (
    DailyDrawdownGuard,
    DailyDrawdownState,
)


# ======================================================================
# 1. DailyDrawdownGuard — 基本テスト
# ======================================================================


class TestDailyDrawdownGuardBasic:
    """基本的なインスタンス化と無効時の挙動."""

    def test_disabled_by_default(self) -> None:
        guard = DailyDrawdownGuard()
        assert not guard.enabled
        assert not guard.is_halted()

    def test_disabled_update_pnl_noop(self) -> None:
        guard = DailyDrawdownGuard(enabled=False)
        result = guard.update_pnl(-100.0)
        assert result["halted"] is False
        assert result["soft_triggered"] is False
        assert result["daily_pnl_bps"] == 0.0

    def test_disabled_maybe_reset_day(self) -> None:
        guard = DailyDrawdownGuard(enabled=False)
        assert guard.maybe_reset_day() is False

    def test_enabled_initial_state(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        assert guard.enabled
        assert not guard.is_halted()
        metrics = guard.get_metrics()
        assert metrics["hard_limit_bps"] == -50.0
        assert metrics["soft_limit_bps"] == -30.0


# ======================================================================
# 2. DailyDrawdownGuard — PnL 追跡テスト
# ======================================================================


class TestDailyDrawdownGuardPnL:
    """PnL 累積と soft/hard 二段制御."""

    def _make_guard(
        self,
        hard: float = -50.0,
        soft: float = -30.0,
    ) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(enabled=True, hard_limit_bps=hard, soft_limit_bps=soft)

    def test_pnl_accumulation(self) -> None:
        guard = self._make_guard()
        guard.update_pnl(-10.0)
        guard.update_pnl(-5.0)
        guard.update_pnl(3.0)
        assert guard.state.daily_pnl_bps == pytest.approx(-12.0)
        assert guard.state.daily_fill_count == 3

    def test_soft_triggered_once(self) -> None:
        guard = self._make_guard(hard=-50.0, soft=-20.0)
        r1 = guard.update_pnl(-10.0)
        assert r1["soft_triggered"] is False
        r2 = guard.update_pnl(-15.0)  # -25 bps <= -20 soft
        assert r2["soft_triggered"] is True
        # 2回目は trigger しない
        r3 = guard.update_pnl(-1.0)
        assert r3["soft_triggered"] is False

    def test_hard_halt(self) -> None:
        guard = self._make_guard(hard=-30.0, soft=-20.0)
        guard.update_pnl(-10.0)
        r = guard.update_pnl(-25.0)  # -35 bps <= -30 hard
        assert r["halted"] is True
        assert guard.is_halted()

    def test_hard_halts_repeatedly_true(self) -> None:
        guard = self._make_guard(hard=-30.0, soft=-20.0)
        guard.update_pnl(-35.0)
        assert guard.is_halted()
        # keep reporting halted
        r = guard.update_pnl(-5.0)
        assert r["halted"] is True
        assert guard.is_halted()

    def test_soft_not_triggered_above_threshold(self) -> None:
        guard = self._make_guard(hard=-50.0, soft=-30.0)
        r = guard.update_pnl(-10.0)
        assert r["soft_triggered"] is False
        assert r["halted"] is False

    def test_positive_pnl_no_trigger(self) -> None:
        guard = self._make_guard()
        r = guard.update_pnl(20.0)
        assert r["halted"] is False
        assert r["soft_triggered"] is False


# ======================================================================
# 3. DailyDrawdownGuard — 日替わりリセット
# ======================================================================


class TestDailyDrawdownDayReset:
    """UTC 日替わりリセットのテスト."""

    def test_day_reset_clears_halt(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-20.0, soft_limit_bps=-10.0)
        guard.update_pnl(-25.0)
        assert guard.is_halted()

        # 日付を進める
        tomorrow = _tomorrow_str()
        with patch.object(DailyDrawdownGuard, "_utc_today", return_value=tomorrow):
            assert not guard.is_halted()
            assert guard.state.daily_pnl_bps == 0.0
            assert guard.state.total_halt_days == 1  # halt された日がカウント

    def test_day_reset_clears_soft_trigger(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-10.0)
        r = guard.update_pnl(-15.0)
        assert r["soft_triggered"] is True

        tomorrow = _tomorrow_str()
        with patch.object(DailyDrawdownGuard, "_utc_today", return_value=tomorrow):
            guard.maybe_reset_day()
            # soft 再発動可能
            r2 = guard.update_pnl(-12.0)
            assert r2["soft_triggered"] is True

    def test_total_halt_days_increments(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-10.0, soft_limit_bps=-5.0)
        # 初日: 固定日付で開始 → update_pnl 内の maybe_reset_day が current day をセット
        with patch.object(DailyDrawdownGuard, "_utc_today", return_value="20500101"):
            guard.update_pnl(-15.0)  # halt day 1
            assert guard.state.total_halt_days == 0  # not counted until reset

        with patch.object(DailyDrawdownGuard, "_utc_today", return_value="20500102"):
            guard.maybe_reset_day()
            assert guard.state.total_halt_days == 1

            guard.update_pnl(-15.0)  # halt day 2

        with patch.object(DailyDrawdownGuard, "_utc_today", return_value="20500103"):
            guard.maybe_reset_day()
            assert guard.state.total_halt_days == 2


# ======================================================================
# 4. DailyDrawdownGuard — State export/import
# ======================================================================


class TestDailyDrawdownStatePersistence:
    """export_state / import_state テスト."""

    def test_export_state_structure(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        guard.update_pnl(-10.0)
        state = guard.export_state()
        assert "current_day" in state
        assert "daily_pnl_bps" in state
        assert "halted" in state
        assert "soft_triggered_today" in state
        assert state["daily_pnl_bps"] == pytest.approx(-10.0)

    def test_import_state_same_day(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        guard.update_pnl(-20.0)
        exported = guard.export_state()

        new_guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        new_guard.import_state(exported)
        assert new_guard.state.daily_pnl_bps == pytest.approx(-20.0)
        assert new_guard.state.daily_fill_count == 1

    def test_import_state_different_day_ignored(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        guard.update_pnl(-40.0)
        exported = guard.export_state()

        # 翌日に import → 無視される
        new_guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        tomorrow = _tomorrow_str()
        with patch.object(DailyDrawdownGuard, "_utc_today", return_value=tomorrow):
            new_guard.import_state(exported)
        assert new_guard.state.daily_pnl_bps == 0.0

    def test_import_state_disabled_noop(self) -> None:
        guard = DailyDrawdownGuard(enabled=False)
        guard.import_state({"current_day": "20260228", "daily_pnl_bps": -99.0})
        assert guard.state.daily_pnl_bps == 0.0

    def test_import_empty_dict_noop(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        guard.import_state({})
        assert guard.state.daily_pnl_bps == 0.0

    def test_roundtrip_halted_state(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-20.0, soft_limit_bps=-10.0)
        guard.update_pnl(-25.0)
        assert guard.is_halted()
        exported = guard.export_state()

        new_guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-20.0, soft_limit_bps=-10.0)
        new_guard.import_state(exported)
        assert new_guard.is_halted()
        assert new_guard.state.daily_pnl_bps == pytest.approx(-25.0)


# ======================================================================
# 5. cancel_reasons 定数テスト
# ======================================================================


class TestDailyDrawdownCancelReason:
    """DAILY_DRAWDOWN_HALT 定数の存在と AUDIT set 所属."""

    def test_constant_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert hasattr(CR, "DAILY_DRAWDOWN_HALT")
        assert CR.DAILY_DRAWDOWN_HALT == "daily_drawdown_halt"

    def test_in_audit_set(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.DAILY_DRAWDOWN_HALT in CR.AUDIT_CANCEL_REASONS


# ======================================================================
# 6. FillTestConfig — daily_drawdown_* フィールドテスト
# ======================================================================


class TestFillTestConfigDailyDrawdown:
    """FillTestConfig の新規フィールドのデフォルト値."""

    def test_default_values(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.daily_drawdown_enabled is False
        assert cfg.daily_drawdown_hard_limit_bps == -50.0
        assert cfg.daily_drawdown_soft_limit_bps == -30.0


# ======================================================================
# 7. FillTestState — daily_drawdown_state フィールドテスト
# ======================================================================


class TestFillTestStateDailyDrawdown:
    """FillTestState の daily_drawdown_state フィールド."""

    def test_default_none(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState()
        assert state.daily_drawdown_state is None

    def test_with_state_dict(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        dd = {"current_day": "20260228", "daily_pnl_bps": -15.0, "halted": False}
        state = FillTestState(daily_drawdown_state=dd)
        assert state.daily_drawdown_state == dd

    def test_backward_compat_load(self) -> None:
        """旧 state ファイル (daily_drawdown_state なし) から FillTestState を生成可能."""
        from dataclasses import fields
        from scripts.v460.lib.resilience import FillTestState
        old_data = {"run_id": "test", "cycle_count": 100, "total_count": 50}
        valid_fields = {f.name for f in fields(FillTestState)}
        filtered = {k: v for k, v in old_data.items() if k in valid_fields}
        state = FillTestState(**filtered)
        assert state.daily_drawdown_state is None
        assert state.cycle_count == 100


# ======================================================================
# 8. get_metrics テスト
# ======================================================================


class TestDailyDrawdownMetrics:
    """get_metrics() の出力内容検証."""

    def test_metrics_keys(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        m = guard.get_metrics()
        expected_keys = {
            "enabled", "current_day", "daily_pnl_bps", "daily_fill_count",
            "halted", "soft_triggered", "hard_limit_bps", "soft_limit_bps",
            "total_halt_days", "halt_blocked_cycles",  # 173# 追加
            # 205# §9.5: 片側 DD
            "per_side_enabled", "daily_pnl_bps_buy", "daily_pnl_bps_sell",
            "side_halted_buy", "side_halted_sell",
        }
        assert set(m.keys()) == expected_keys

    def test_metrics_after_update(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-20.0)
        guard.update_pnl(-25.0)
        m = guard.get_metrics()
        assert m["daily_pnl_bps"] == pytest.approx(-25.0)
        assert m["daily_fill_count"] == 1
        assert m["soft_triggered"] is True
        assert m["halted"] is False


# ======================================================================
# Helpers
# ======================================================================


def _tomorrow_str() -> str:
    """翌日の UTC 日付文字列 (テスト用)."""
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    tomorrow = now + timedelta(days=1)
    return tomorrow.strftime("%Y%m%d")


# ======================================================================
# 9. 205# §9.5: 片側 DD Halt テスト
# ======================================================================


class TestPerSideDDHalt:
    """205# §9.5: サイド別累積損失ガード."""

    def _make_guard(
        self,
        hard: float = -50.0,
        soft: float = -30.0,
        per_side_hard: float = -25.0,
        per_side_halt_cycles: int = 0,
    ) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=hard,
            soft_limit_bps=soft,
            per_side_enabled=True,
            per_side_hard_limit_bps=per_side_hard,
            per_side_halt_cycles=per_side_halt_cycles,
        )

    def test_per_side_disabled_by_default(self) -> None:
        guard = DailyDrawdownGuard(enabled=True)
        result = guard.update_pnl(-100.0, side="buy")
        assert result["side_halted"] == ""
        assert not guard.is_side_halted("buy")

    def test_per_side_buy_halt(self) -> None:
        guard = self._make_guard(per_side_hard=-20.0)
        guard.update_pnl(-10.0, side="buy")
        assert not guard.is_side_halted("buy")
        guard.update_pnl(-15.0, side="buy")  # buy累積 -25 <= -20
        assert guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

    def test_per_side_sell_halt(self) -> None:
        guard = self._make_guard(per_side_hard=-20.0)
        guard.update_pnl(-25.0, side="sell")
        assert guard.is_side_halted("sell")
        assert not guard.is_side_halted("buy")

    def test_per_side_pnl_tracked_independently(self) -> None:
        guard = self._make_guard(per_side_hard=-30.0)
        guard.update_pnl(-20.0, side="buy")
        guard.update_pnl(-20.0, side="sell")
        # 集約は -40、片側はそれぞれ -20 → まだ封鎖されない
        assert not guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

    def test_per_side_halt_cycles_expiry(self) -> None:
        guard = self._make_guard(per_side_hard=-10.0, per_side_halt_cycles=3)
        guard.update_pnl(-15.0, side="buy")
        assert guard.is_side_halted("buy")
        # 3サイクル tick で解除
        guard.tick_side_halt()  # remaining: 2
        assert guard.is_side_halted("buy")
        guard.tick_side_halt()  # remaining: 1
        assert guard.is_side_halted("buy")
        guard.tick_side_halt()  # remaining: 0 → 解除
        assert not guard.is_side_halted("buy")

    def test_per_side_day_reset_clears_halt(self) -> None:
        guard = self._make_guard(per_side_hard=-10.0)
        guard.update_pnl(-15.0, side="sell")
        assert guard.is_side_halted("sell")

        tomorrow = _tomorrow_str()
        with patch.object(DailyDrawdownGuard, "_utc_today", return_value=tomorrow):
            guard.maybe_reset_day()
            assert not guard.is_side_halted("sell")
            assert guard.state.daily_pnl_bps_sell == 0.0

    def test_per_side_halt_returns_side_halted(self) -> None:
        guard = self._make_guard(per_side_hard=-10.0)
        result = guard.update_pnl(-15.0, side="buy")
        assert result["side_halted"] == "buy"

    def test_per_side_export_import(self) -> None:
        guard = self._make_guard(per_side_hard=-10.0, per_side_halt_cycles=5)
        guard.update_pnl(-15.0, side="buy")
        guard.update_pnl(-5.0, side="sell")
        exported = guard.export_state()

        new_guard = self._make_guard(per_side_hard=-10.0, per_side_halt_cycles=5)
        new_guard.import_state(exported)
        assert new_guard.is_side_halted("buy")
        assert not new_guard.is_side_halted("sell")
        assert new_guard.state.daily_pnl_bps_buy == pytest.approx(-15.0)
        assert new_guard.state.daily_pnl_bps_sell == pytest.approx(-5.0)


# ======================================================================
# 10. 205# cancel_reasons 追加テスト
# ======================================================================


class TestCancelReasons205:
    """205# で追加した cancel_reason 定数。"""

    def test_hard_skip_utc_hour_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.HARD_SKIP_UTC_HOUR == "hard_skip_utc_hour"
        assert CR.HARD_SKIP_UTC_HOUR in CR.AUDIT_CANCEL_REASONS

    def test_toxic_fill_side_veto_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.TOXIC_FILL_SIDE_VETO == "toxic_fill_side_veto"
        assert CR.TOXIC_FILL_SIDE_VETO in CR.AUDIT_CANCEL_REASONS

    def test_per_side_dd_halt_exists(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.PER_SIDE_DD_HALT == "per_side_dd_halt"
        assert CR.PER_SIDE_DD_HALT in CR.AUDIT_CANCEL_REASONS


# ======================================================================
# 11. 205# FillTestConfig 新規フィールドテスト
# ======================================================================


class TestFillTestConfig205:
    """205# で追加した FillTestConfig フィールドのデフォルト値。"""

    def test_hard_skip_utc_hours_default(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.hard_skip_utc_hours == []

    def test_toxic_fill_veto_defaults(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.toxic_fill_veto_threshold_bps == -5.0
        assert cfg.toxic_fill_veto_cycles == 3

    def test_per_side_dd_defaults(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert cfg.per_side_dd_enabled is False
        assert cfg.per_side_dd_hard_limit_bps == -30.0
        assert cfg.per_side_dd_halt_cycles == 0
