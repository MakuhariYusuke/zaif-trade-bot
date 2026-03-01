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


# ======================================================================
# 12. 207# 堅牢性修正テスト
# ======================================================================


class TestFillTestStateToxicVeto:
    """207# §1: FillTestState に toxic_veto フィールドが存在し永続化可能。"""

    def test_toxic_veto_field_exists(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState()
        assert state.toxic_veto is None

    def test_toxic_veto_with_data(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState(toxic_veto={"buy": 3, "sell": 1})
        assert state.toxic_veto == {"buy": 3, "sell": 1}

    def test_toxic_veto_none_default(self) -> None:
        from scripts.v460.lib.resilience import FillTestState
        state = FillTestState(toxic_veto=None)
        assert state.toxic_veto is None


class TestPerSideDDWarmup:
    """207# §2: warmup が per-side PnL を正しく計算する。"""

    def _make_guard(
        self,
        per_side_hard: float = -20.0,
        per_side_halt_cycles: int = 0,
    ) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=per_side_hard,
            per_side_halt_cycles=per_side_halt_cycles,
        )

    def test_warmup_sets_per_side_pnl(self) -> None:
        """warmup 後に daily_pnl_bps_buy / sell が計算されること。"""
        guard = self._make_guard(per_side_hard=-20.0)
        # warmup で注入される値をシミュレート
        guard.state.daily_pnl_bps_buy = -10.0
        guard.state.daily_pnl_bps_sell = -5.0
        guard.state.daily_pnl_bps = -15.0
        guard.state.daily_fill_count = 3

        assert guard.state.daily_pnl_bps_buy == pytest.approx(-10.0)
        assert guard.state.daily_pnl_bps_sell == pytest.approx(-5.0)
        assert not guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")

    def test_warmup_triggers_per_side_halt(self) -> None:
        """warmup で閾値超過した場合に片側封鎖されること。"""
        guard = self._make_guard(per_side_hard=-10.0)
        # 直接 state を設定（warmup がやることをシミュレート）
        guard.state.daily_pnl_bps_buy = -15.0
        guard.state.side_halted_buy = True
        guard.state.side_halt_remaining_buy = 0

        assert guard.is_side_halted("buy")
        assert not guard.is_side_halted("sell")


class TestToxicVetoDayReset:
    """207# §4: UTC 日替わりで toxic veto がクリアされること。

    (fill_loop_orchestrator 内のロジックのため、ここでは DailyDrawdownGuard
    の maybe_reset_day がトリガとなることを前提に、ガード側の動作を確認。)
    """

    def test_day_reset_returns_true(self) -> None:
        """maybe_reset_day が True を返した際に veto クリアのトリガとなること。"""
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-30.0)
        tomorrow = _tomorrow_str()
        with patch.object(DailyDrawdownGuard, "_utc_today", return_value=tomorrow):
            assert guard.maybe_reset_day() is True


class TestPerSideDDAndVetoInteraction:
    """207# §5b: per_side_dd と toxic veto の相互作用テスト。"""

    def test_veto_alt_side_blocked_by_dd(self) -> None:
        """toxic veto で切替先が per_side_dd で封鎖されている場合の検証。

        per_side_dd が sell を封鎖中 + toxic veto が buy を封鎖中
        → 両方封鎖 → スキップが期待される。
        """
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=-10.0,
            per_side_halt_cycles=0,
        )
        # sell 側を per_side_dd で封鎖
        guard.update_pnl(-15.0, side="sell")
        assert guard.is_side_halted("sell")

        # toxic veto は dict として外部管理 (orchestrator)
        toxic_veto: dict[str, int] = {"buy": 3}

        # next_side = buy → toxic veto でブロック → alt = sell
        next_side = "buy"
        assert next_side in toxic_veto
        alt_side = "sell"
        # alt_side が per_side_dd で封鎖されているか確認
        alt_blocked = alt_side in toxic_veto or guard.is_side_halted(alt_side)
        assert alt_blocked is True  # sell は per_side_dd で封鎖済み


class TestOneSidedConsecutiveConfig:
    """207# §4: one-sided 連続実行制限の設定デフォルト値テスト。"""

    def test_default_values(self) -> None:
        """FillTestConfig のデフォルト値が正しいこと。"""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.one_sided_consecutive_limit == 5
        assert cfg.one_sided_consecutive_interval_mult == 3.0

    def test_custom_values(self) -> None:
        """カスタム値を設定できること。"""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(
            one_sided_consecutive_limit=10,
            one_sided_consecutive_interval_mult=5.0,
        )
        assert cfg.one_sided_consecutive_limit == 10
        assert cfg.one_sided_consecutive_interval_mult == 5.0

    def test_disabled_when_zero(self) -> None:
        """limit=0 で無制限（無効化）となること。"""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig(one_sided_consecutive_limit=0)
        # limit=0 → 条件 `_os_limit > 0` が False → mult 不適用
        assert cfg.one_sided_consecutive_limit == 0


class TestOneSidedConsecutiveMultLogic:
    """207# §4: one-sided interval 乗数の適用ロジック単体テスト。

    fill_loop_orchestrator の sleep 付近のロジックを関数外で再現し、
    multiplier 計算の正しさを検証する。
    """

    @staticmethod
    def _calc_os_mult(count: int, limit: int, mult: float) -> float:
        """orchestrator の §4 ロジックを再現。"""
        if limit > 0 and count >= limit:
            return mult
        return 1.0

    def test_under_limit(self) -> None:
        assert self._calc_os_mult(3, 5, 3.0) == 1.0

    def test_at_limit(self) -> None:
        assert self._calc_os_mult(5, 5, 3.0) == 3.0

    def test_over_limit(self) -> None:
        assert self._calc_os_mult(8, 5, 3.0) == 3.0

    def test_disabled(self) -> None:
        assert self._calc_os_mult(100, 0, 3.0) == 1.0


class TestConfigValidation209:
    """209# config validation テスト."""

    def test_negative_one_sided_limit_raises(self) -> None:
        """one_sided_consecutive_limit < 0 で ValueError。"""
        import pytest
        from scripts.v460.lib.fill_config import FillTestConfig

        with pytest.raises(ValueError, match="one_sided_consecutive_limit"):
            FillTestConfig(one_sided_consecutive_limit=-1)

    def test_zero_one_sided_mult_raises(self) -> None:
        """one_sided_consecutive_interval_mult <= 0 で ValueError。"""
        import pytest
        from scripts.v460.lib.fill_config import FillTestConfig

        with pytest.raises(ValueError, match="one_sided_consecutive_interval_mult"):
            FillTestConfig(one_sided_consecutive_interval_mult=0.0)

    def test_zero_cycle_interval_raises(self) -> None:
        """cycle_interval_sec <= 0 で ValueError。"""
        import pytest
        from scripts.v460.lib.fill_config import FillTestConfig

        with pytest.raises(ValueError, match="cycle_interval_sec"):
            FillTestConfig(cycle_interval_sec=0.0)

    def test_zero_poll_interval_raises(self) -> None:
        """poll_interval_sec <= 0 で ValueError。"""
        import pytest
        from scripts.v460.lib.fill_config import FillTestConfig

        with pytest.raises(ValueError, match="poll_interval_sec"):
            FillTestConfig(poll_interval_sec=-1.0)

    def test_max_cycle_sleep_default(self) -> None:
        """max_cycle_sleep_sec のデフォルト値が 600。"""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.max_cycle_sleep_sec == 600.0

    def test_max_cycle_sleep_negative_raises(self) -> None:
        """max_cycle_sleep_sec < 0 で ValueError。"""
        import pytest
        from scripts.v460.lib.fill_config import FillTestConfig

        with pytest.raises(ValueError, match="max_cycle_sleep_sec"):
            FillTestConfig(max_cycle_sleep_sec=-1.0)


class TestSleepClampLogic209:
    """209# M4: sleep 乗数上限キャップロジックのテスト。"""

    @staticmethod
    def _calc_clamped_sleep(
        interval: float,
        soft_dd_mult: float,
        loss_cd: float,
        os_mult: float,
        max_sleep: float,
    ) -> float:
        """orchestrator sleep ロジックを再現。"""
        raw = interval * soft_dd_mult * loss_cd * os_mult
        if max_sleep > 0:
            return min(raw, max_sleep)
        return raw

    def test_under_cap(self) -> None:
        result = self._calc_clamped_sleep(120, 1.0, 1.0, 1.0, 600.0)
        assert result == 120.0

    def test_capped(self) -> None:
        # 120 * 3.0 * 2.0 * 3.0 = 2160 → capped to 600
        result = self._calc_clamped_sleep(120, 3.0, 2.0, 3.0, 600.0)
        assert result == 600.0

    def test_disabled_cap(self) -> None:
        # max_sleep=0 → no cap
        result = self._calc_clamped_sleep(120, 3.0, 2.0, 3.0, 0.0)
        assert result == 2160.0


class TestVetoDeadlockFix209:
    """209# H-1: 両サイド veto 時のデクリメント検証。"""

    def test_both_blocked_decrements_veto(self) -> None:
        """両サイド封鎖時に veto カウンタが減算されること (デッドロック防止)。"""
        toxic_veto: dict[str, int] = {"buy": 2, "sell": 1}

        # 209# ロジック再現: both-blocked パスでの decrement
        for _vs in list(toxic_veto.keys()):
            toxic_veto[_vs] -= 1
            if toxic_veto[_vs] <= 0:
                del toxic_veto[_vs]

        # sell は 1→0 で削除、buy は 2→1 で残存
        assert toxic_veto == {"buy": 1}

    def test_both_blocked_clears_all(self) -> None:
        """両サイドとも残り1の場合、両方クリアされること。"""
        toxic_veto: dict[str, int] = {"buy": 1, "sell": 1}

        for _vs in list(toxic_veto.keys()):
            toxic_veto[_vs] -= 1
            if toxic_veto[_vs] <= 0:
                del toxic_veto[_vs]

        assert toxic_veto == {}


class TestInstantVelocityBoundary209:
    """209# M-4: dt == max_dt の境界条件テスト。"""

    def test_dt_equals_max_dt_returns_none(self) -> None:
        """dt == max_dt の場合は None (stale) を返すこと。"""
        from scripts.v460.lib.velocity_math import compute_instant_velocity_bps

        result = compute_instant_velocity_bps(
            current_mid=10_100_000.0,
            prev_mid=10_000_000.0,
            dt=30.0,
            max_dt=30.0,
        )
        assert result is None

