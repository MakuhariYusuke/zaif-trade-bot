"""168# §4.1 #3: DailyDrawdownGuard ユニットテスト.

テスト対象:
- DailyDrawdownGuard クラス (日次 PnL 追跡、soft/hard 二段制御)
- cancel_reasons.DAILY_DRAWDOWN_HALT 定数
- FillTestConfig の daily_drawdown_* フィールド
- FillTestState の daily_drawdown_state フィールド
- State export/import (永続化)
- 210#: FFD hot-reload, spread staleness, one-sided count persistence, velocity wiring, DRY snapshot
"""

from __future__ import annotations

import time
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib import cancel_reasons as CR
from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator
from scripts.v460.lib.daily_drawdown_guard import (
    DailyDrawdownGuard,
    DailyDrawdownState,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from scripts.v460.lib.resilience import FillTestState
from scripts.v460.lib.velocity_math import compute_instant_velocity_bps
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from ztb.utils.dataclass_utils import filter_known_dataclass_fields


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
        with patch.object(DailyDrawdownGuard, "_today", return_value=tomorrow):
            assert not guard.is_halted()
            assert guard.state.daily_pnl_bps == 0.0
            assert guard.state.total_halt_days == 1  # halt された日がカウント

    def test_day_reset_clears_soft_trigger(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0, soft_limit_bps=-10.0)
        r = guard.update_pnl(-15.0)
        assert r["soft_triggered"] is True

        tomorrow = _tomorrow_str()
        with patch.object(DailyDrawdownGuard, "_today", return_value=tomorrow):
            guard.maybe_reset_day()
            # soft 再発動可能
            r2 = guard.update_pnl(-12.0)
            assert r2["soft_triggered"] is True

    def test_total_halt_days_increments(self) -> None:
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-10.0, soft_limit_bps=-5.0)
        # 初日: 固定日付で開始 → update_pnl 内の maybe_reset_day が current day をセット
        with patch.object(DailyDrawdownGuard, "_today", return_value="20500101"):
            guard.update_pnl(-15.0)  # halt day 1
            assert guard.state.total_halt_days == 0  # not counted until reset

        with patch.object(DailyDrawdownGuard, "_today", return_value="20500102"):
            guard.maybe_reset_day()
            assert guard.state.total_halt_days == 1

            guard.update_pnl(-15.0)  # halt day 2

        with patch.object(DailyDrawdownGuard, "_today", return_value="20500103"):
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
        with patch.object(DailyDrawdownGuard, "_today", return_value=tomorrow):
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
        assert hasattr(CR, "DAILY_DRAWDOWN_HALT")
        assert CR.DAILY_DRAWDOWN_HALT == "daily_drawdown_halt"

    def test_in_audit_set(self) -> None:
        assert CR.DAILY_DRAWDOWN_HALT in CR.AUDIT_CANCEL_REASONS


# ======================================================================
# 6. FillTestConfig — daily_drawdown_* フィールドテスト
# ======================================================================


class TestFillTestConfigDailyDrawdown:
    """FillTestConfig の新規フィールドのデフォルト値."""

    def test_default_values(self) -> None:
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
        state = FillTestState()
        assert state.daily_drawdown_state is None

    def test_with_state_dict(self) -> None:
        dd = {"current_day": "20260228", "daily_pnl_bps": -15.0, "halted": False}
        state = FillTestState(daily_drawdown_state=dd)
        assert state.daily_drawdown_state == dd

    def test_backward_compat_load(self) -> None:
        """旧 state ファイル (daily_drawdown_state なし) から FillTestState を生成可能."""
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
            # 224# B1: リカバリ
            "side_recovery_remaining_buy", "side_recovery_remaining_sell",
            # 246# cooldown release
            "cooldown_released", "cooldown_release_lot_scale",
            # 249# cooldown re-arm
            "cooldown_rearmed", "cooldown_rearm_pnl_bps",
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
        with patch.object(DailyDrawdownGuard, "_today", return_value=tomorrow):
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
        assert CR.HARD_SKIP_UTC_HOUR == "hard_skip_utc_hour"
        assert CR.HARD_SKIP_UTC_HOUR in CR.AUDIT_CANCEL_REASONS

    def test_toxic_fill_side_veto_exists(self) -> None:
        assert CR.TOXIC_FILL_SIDE_VETO == "toxic_fill_side_veto"
        assert CR.TOXIC_FILL_SIDE_VETO in CR.AUDIT_CANCEL_REASONS

    def test_per_side_dd_halt_exists(self) -> None:
        assert CR.PER_SIDE_DD_HALT == "per_side_dd_halt"
        assert CR.PER_SIDE_DD_HALT in CR.AUDIT_CANCEL_REASONS


# ======================================================================
# 11. 205# FillTestConfig 新規フィールドテスト
# ======================================================================


class TestFillTestConfig205:
    """205# で追加した FillTestConfig フィールドのデフォルト値。"""

    def test_hard_skip_utc_hours_default(self) -> None:
        cfg = FillTestConfig()
        assert cfg.hard_skip_utc_hours == []

    def test_toxic_fill_veto_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.toxic_fill_veto_threshold_bps == -5.0
        assert cfg.toxic_fill_veto_cycles == 3

    def test_per_side_dd_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.per_side_dd_enabled is False
        assert cfg.per_side_dd_hard_limit_bps == -50.0
        assert cfg.per_side_dd_halt_cycles == 10
        assert cfg.per_side_dd_reanchor_budget_bps == -25.0


# ======================================================================
# 12. 207# 堅牢性修正テスト
# ======================================================================


class TestFillTestStateToxicVeto:
    """207# §1: FillTestState に toxic_veto フィールドが存在し永続化可能。"""

    def test_toxic_veto_field_exists(self) -> None:
        state = FillTestState()
        assert state.toxic_veto is None

    def test_toxic_veto_with_data(self) -> None:
        state = FillTestState(toxic_veto={"buy": 3, "sell": 1})
        assert state.toxic_veto == {"buy": 3, "sell": 1}

    def test_toxic_veto_none_default(self) -> None:
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
        with patch.object(DailyDrawdownGuard, "_today", return_value=tomorrow):
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

        cfg = FillTestConfig()
        assert cfg.one_sided_consecutive_limit == 5
        assert cfg.one_sided_consecutive_interval_mult == 3.0

    def test_custom_values(self) -> None:
        """カスタム値を設定できること。"""

        cfg = FillTestConfig(
            one_sided_consecutive_limit=10,
            one_sided_consecutive_interval_mult=5.0,
        )
        assert cfg.one_sided_consecutive_limit == 10
        assert cfg.one_sided_consecutive_interval_mult == 5.0

    def test_disabled_when_zero(self) -> None:
        """limit=0 で無制限（無効化）となること。"""

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

        with pytest.raises(ValueError, match="one_sided_consecutive_limit"):
            FillTestConfig(one_sided_consecutive_limit=-1)

    def test_zero_one_sided_mult_raises(self) -> None:
        """one_sided_consecutive_interval_mult <= 0 で ValueError。"""

        with pytest.raises(ValueError, match="one_sided_consecutive_interval_mult"):
            FillTestConfig(one_sided_consecutive_interval_mult=0.0)

    def test_zero_cycle_interval_raises(self) -> None:
        """cycle_interval_sec <= 0 で ValueError。"""

        with pytest.raises(ValueError, match="cycle_interval_sec"):
            FillTestConfig(cycle_interval_sec=0.0)

    def test_zero_poll_interval_raises(self) -> None:
        """poll_interval_sec <= 0 で ValueError。"""

        with pytest.raises(ValueError, match="poll_interval_sec"):
            FillTestConfig(poll_interval_sec=-1.0)

    def test_max_cycle_sleep_default(self) -> None:
        """max_cycle_sleep_sec のデフォルト値が 600。"""

        cfg = FillTestConfig()
        assert cfg.max_cycle_sleep_sec == 600.0

    def test_max_cycle_sleep_negative_raises(self) -> None:
        """max_cycle_sleep_sec < 0 で ValueError。"""

        with pytest.raises(ValueError, match="max_cycle_sleep_sec"):
            FillTestConfig(max_cycle_sleep_sec=-1.0)

    def test_loss_cap_ratio_zero_raises(self) -> None:
        """327# loss_cap_ratio <= 0 で ValueError (ZeroDivisionError 防止)。"""
        with pytest.raises(ValueError, match="loss_cap_ratio"):
            FillTestConfig(loss_cap_ratio=0.0)

    def test_loss_cap_ratio_negative_raises(self) -> None:
        """327# loss_cap_ratio < 0 で ValueError。"""
        with pytest.raises(ValueError, match="loss_cap_ratio"):
            FillTestConfig(loss_cap_ratio=-0.01)

    def test_soft_loss_cap_ratio_negative_raises(self) -> None:
        """327# soft_loss_cap_ratio < 0 で ValueError。"""
        with pytest.raises(ValueError, match="soft_loss_cap_ratio"):
            FillTestConfig(soft_loss_cap_ratio=-0.01)


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

        result = compute_instant_velocity_bps(
            current_mid=10_100_000.0,
            prev_mid=10_000_000.0,
            dt=30.0,
            max_dt=30.0,
        )
        assert result is None


# ======================================================================
# 210# テスト群
# ======================================================================


class TestFillTestStateOneSidedPersistence210:
    """210# L-2: one_sided_consecutive_count の FillTestState 永続化."""

    def test_field_default(self) -> None:
        """デフォルト値は 0."""
        state = FillTestState()
        assert state.one_sided_consecutive_count == 0

    def test_round_trip(self) -> None:
        """dataclass asdict → FillTestState の往復でカウンタが保存/復元される."""
        state = FillTestState(one_sided_consecutive_count=7, toxic_veto={"buy": 2})
        d = asdict(state)
        assert d["one_sided_consecutive_count"] == 7
        # filter で復元
        restored = FillTestState(**filter_known_dataclass_fields(FillTestState, d))
        assert restored.one_sided_consecutive_count == 7
        assert restored.toxic_veto == {"buy": 2}


class TestSpreadStaleness210:
    """210# M5: last_spread staleness guard."""

    @staticmethod
    def _make_calc():
        """テスト用 MakerPriceCalculator を構築."""
        cfg = FillTestConfig()
        ffd = MagicMock()
        return MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=cfg.spread_offset_ratio,
        )

    def test_fresh_spread_returned(self) -> None:
        """60秒以内の spread は正常に返る."""
        calc = self._make_calc()
        calc._last_spread = 500.0
        calc._last_spread_time = time.time()
        assert calc.last_spread == 500.0

    def test_stale_spread_returns_none(self) -> None:
        """60秒超の spread は None を返す."""
        calc = self._make_calc()
        calc._last_spread = 500.0
        calc._last_spread_time = time.time() - 61.0
        assert calc.last_spread is None

    def test_no_spread_returns_none(self) -> None:
        """compute() 未実行時は None."""
        calc = self._make_calc()
        assert calc.last_spread is None


class TestMidTrendBpsProperty210:
    """210# H3: last_mid_trend_bps property の動作確認."""

    @staticmethod
    def _make_calc():
        """テスト用 MakerPriceCalculator を構築."""
        cfg = FillTestConfig()
        ffd = MagicMock()
        return MakerPriceCalculator(
            config=cfg,
            fast_fill_defense=ffd,
            regime_detector=None,
            base_offset_ratio=cfg.spread_offset_ratio,
        )

    def test_initial_none(self) -> None:
        """初期値は None."""
        calc = self._make_calc()
        assert calc.last_mid_trend_bps is None

    def test_set_and_get(self) -> None:
        """_last_mid_trend_bps を設定し property で取得できる."""
        calc = self._make_calc()
        calc._last_mid_trend_bps = 5.0
        assert calc.last_mid_trend_bps == 5.0


class TestFFDHotReloadSync210:
    """210# H2: FFD hot-reload 後の MakerPriceCalculator 参照同期."""

    def test_rebuild_syncs_ffd(self) -> None:
        """_rebuild_fast_fill_defense 後に _maker_price._fast_fill_defense が同期される."""
        # シンプルなモック runner を構築
        class MockRunner:
            def __init__(self) -> None:
                self.config = FillTestConfig()
                self._maker_price = MakerPriceCalculator(
                    config=self.config,
                    fast_fill_defense=MagicMock(),
                    regime_detector=None,
                    base_offset_ratio=self.config.spread_offset_ratio,
                )
                self._fast_fill_defense = object()  # 初期 FFD
                self._maker_price._fast_fill_defense = self._fast_fill_defense
                self._git_sha = "test"

            def _rebuild_sell_kill_mgr(self) -> None: ...
            def _rebuild_buy_kill_mgr(self) -> None: ...
            def _rebuild_daily_drawdown_guard(self) -> None: ...
            def _rebuild_fast_fill_defense(self) -> None:
                # 本番の _rebuild は新しいインスタンスを代入
                self._fast_fill_defense = object()
            def _rebuild_cycle_strategy(self) -> None: ...

        runner = MockRunner()
        old_ffd = runner._fast_fill_defense

        # hot-reload コールバックのシミュレーション
        # 直接 _rebuild を呼び、その後 sync 処理を検証
        runner._rebuild_fast_fill_defense()
        new_ffd = runner._fast_fill_defense
        assert new_ffd is not old_ffd, "rebuild で新インスタンスが生成されているべき"

        # sync 処理: hot-reload コードと同じロジック (setter を使用)
        _ffd = getattr(runner, "_fast_fill_defense", None)
        if _ffd is not None:
            runner._maker_price.update_fast_fill_defense(_ffd)

        assert runner._maker_price._fast_fill_defense is new_ffd


class TestVelocityGateWiring210:
    """210# H3: CycleGateAggregator に velocity が渡されること確認."""

    def test_velocity_skip_with_value(self) -> None:
        """price_velocity_bps が渡された場合に velocity gate が評価される."""
        cfg = FillTestConfig(
            sell_velocity_skip_enabled=True,
            sell_velocity_skip_threshold_bps=5.0,
            velocity_skip_as_offset_enabled=False,  # hard mode で test
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            price_velocity_bps=10.0,  # > threshold (5.0)
        )
        assert result.blocked is True
        assert result.blocking_reason == "rule_velocity_sell_skip"

    def test_velocity_none_passes(self) -> None:
        """price_velocity_bps=None の場合は velocity gate を通過."""
        cfg = FillTestConfig(
            sell_velocity_skip_enabled=True,
            sell_velocity_skip_threshold_bps=5.0,
            velocity_skip_as_offset_enabled=False,
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            price_velocity_bps=None,
        )
        # velocity gate は通過、他の gate でブロックされない限り
        # velocity_skip はブロックしない
        velocity_check = next(
            (c for c in result.checks if c.gate_name == "velocity_skip"), None
        )
        assert velocity_check is not None
        assert velocity_check.blocked is False

    def test_velocity_soft_mode_passes(self) -> None:
        """velocity_skip_as_offset_enabled=True の場合は velocity gate を通過."""
        cfg = FillTestConfig(
            sell_velocity_skip_enabled=True,
            sell_velocity_skip_threshold_bps=5.0,
            velocity_skip_as_offset_enabled=True,  # soft mode
        )
        gate = CycleGateAggregator(cfg)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=1.0,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            price_velocity_bps=10.0,
        )
        velocity_check = next(
            (c for c in result.checks if c.gate_name == "velocity_skip"), None
        )
        assert velocity_check is not None
        assert velocity_check.blocked is False


# ========================================================================
# 211# 204# I: Per-fill loss offset boost テスト
# ========================================================================


class TestLossBoostOffset211:
    """211# 204# I: 大損後 offset 拡大の検証."""

    @staticmethod
    def _make_calc() -> "MakerPriceCalculator":
        cfg = FillTestConfig()
        ffd = MagicMock()
        ffd.get_boost_multiplier.return_value = 1.0
        return MakerPriceCalculator(
            cfg, ffd, regime_detector=None, base_offset_ratio=0.05,
        )

    def test_initial_loss_boost_is_noop(self) -> None:
        """初期状態では loss_boost_mult = 1.0 (noop)."""
        calc = self._make_calc()
        assert calc._loss_boost_mult == 1.0

    def test_set_loss_boost_applies_once(self) -> None:
        """set_loss_boost() 後、_loss_boost_mult が設定される."""
        calc = self._make_calc()
        calc.set_loss_boost(1.5)
        assert calc._loss_boost_mult == 1.5

    def test_loss_boost_config_default(self) -> None:
        """FillTestConfig.loss_boost_offset_mult のデフォルト値が 1.5."""
        cfg = FillTestConfig()
        assert cfg.loss_boost_offset_mult == 1.5


# ========================================================================
# 246# DD Halt Cooldown Release テスト
# ========================================================================


class TestCooldownRelease246:
    """246# cooldown release: 集約 halt からの時間ベース部分解除."""

    def test_cooldown_disabled_by_default(self) -> None:
        """cooldown_release_sec=0 (デフォルト) では cooldown release 無効."""
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0)
        guard.update_pnl(-60.0)
        assert guard.is_halted() is True
        assert guard.get_cooldown_lot_scale() == 1.0

    def test_cooldown_not_released_before_timeout(self) -> None:
        """halt 後、cooldown_release_sec 未経過では halted のまま."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=3600.0, cooldown_release_lot_scale=0.3,
        )
        guard.update_pnl(-60.0)
        assert guard.is_halted() is True
        assert guard.get_cooldown_lot_scale() == 1.0

    def test_cooldown_released_after_timeout(self) -> None:
        """halt 後、cooldown_release_sec 経過で partial release."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=3600.0, cooldown_release_lot_scale=0.3,
        )
        guard.update_pnl(-60.0)
        assert guard.is_halted() is True
        # Simulate time passage
        guard._state.halt_triggered_at = time.time() - 3601.0
        assert guard.is_halted() is False
        assert guard._state.cooldown_released is True
        assert guard.get_cooldown_lot_scale() == 0.3

    def test_cooldown_lot_scale_default_when_not_released(self) -> None:
        """cooldown 未解除時は lot_scale=1.0."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=3600.0, cooldown_release_lot_scale=0.3,
        )
        assert guard.get_cooldown_lot_scale() == 1.0

    def test_cooldown_released_persists_across_is_halted_calls(self) -> None:
        """cooldown_released=True は永続 (同日中)."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=100.0, cooldown_release_lot_scale=0.5,
        )
        guard.update_pnl(-60.0)
        guard._state.halt_triggered_at = time.time() - 200.0
        # First call: sets cooldown_released
        assert guard.is_halted() is False
        # Subsequent calls: still released
        assert guard.is_halted() is False
        assert guard.get_cooldown_lot_scale() == 0.5

    def test_cooldown_export_import_roundtrip(self) -> None:
        """cooldown_released が export/import で保持される."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=100.0, cooldown_release_lot_scale=0.3,
        )
        guard.update_pnl(-60.0)
        guard._state.halt_triggered_at = time.time() - 200.0
        guard.is_halted()  # trigger cooldown release
        assert guard._state.cooldown_released is True
        exported = guard.export_state()
        assert exported["cooldown_released"] is True

        # Import into new guard
        guard2 = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=100.0, cooldown_release_lot_scale=0.3,
        )
        guard2.import_state(exported)
        assert guard2._state.cooldown_released is True
        assert guard2.get_cooldown_lot_scale() == 0.3

    def test_cooldown_reset_on_day_change(self) -> None:
        """日替わりで cooldown_released がリセットされる."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=100.0, cooldown_release_lot_scale=0.3,
        )
        guard.update_pnl(-60.0)
        guard._state.halt_triggered_at = time.time() - 200.0
        guard.is_halted()
        assert guard._state.cooldown_released is True
        # Force day reset
        guard._state.current_day = "19700101"  # stale day
        guard.maybe_reset_day()
        assert guard._state.cooldown_released is False
        assert guard._state.halted is False
        assert guard.get_cooldown_lot_scale() == 1.0

    def test_cooldown_metrics_include_fields(self) -> None:
        """get_metrics() に cooldown_released, cooldown_release_lot_scale が含まれる."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=7200.0, cooldown_release_lot_scale=0.3,
        )
        metrics = guard.get_metrics()
        assert "cooldown_released" in metrics
        assert metrics["cooldown_released"] is False
        assert metrics["cooldown_release_lot_scale"] == 0.3

    def test_cooldown_halt_blocked_cycles_not_incremented_when_released(self) -> None:
        """cooldown release 後は halt_blocked_cycles がインクリメントされない."""
        guard = DailyDrawdownGuard(
            enabled=True, hard_limit_bps=-50.0,
            cooldown_release_sec=100.0, cooldown_release_lot_scale=0.3,
        )
        guard.update_pnl(-60.0)
        # Before cooldown: halt should increment blocked_cycles
        blocked_before = guard._state.halt_blocked_cycles
        guard.is_halted()
        assert guard._state.halt_blocked_cycles == blocked_before + 1
        # After cooldown release: no increment
        guard._state.halt_triggered_at = time.time() - 200.0
        guard.is_halted()
        blocked_after = guard._state.halt_blocked_cycles
        guard.is_halted()  # should still not increment
        assert guard._state.halt_blocked_cycles == blocked_after


class TestCooldownReleaseConfig246:
    """246# FillTestConfig cooldown release フィールドテスト."""

    def test_config_defaults(self) -> None:
        """dd_cooldown_release_sec/lot_scale のデフォルト値."""
        cfg = FillTestConfig()
        assert cfg.dd_cooldown_release_sec == 0.0
        assert cfg.dd_cooldown_release_lot_scale == 0.3

    def test_config_yaml_parsing(self) -> None:
        """YAML から cooldown_release 設定がパースされる."""
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "loss_control": {
                        "daily_drawdown": {
                            "enabled": True,
                            "hard_limit_bps": -50.0,
                            "soft_limit_bps": -30.0,
                            "cooldown_release_sec": 7200.0,
                            "cooldown_release_lot_scale": 0.5,
                        }
                    }
                }
            )
        )
        assert cfg.dd_cooldown_release_sec == 7200.0
        assert cfg.dd_cooldown_release_lot_scale == 0.5


# ======================================================================
# 268# DD 日付リセット JST 化テスト
# ======================================================================


class TestDayResetTimezone:
    """268# DD 日付リセットのタイムゾーン設定テスト."""

    def test_default_utc_offset_is_zero(self) -> None:
        """デフォルトは UTC (offset=0)."""
        guard = DailyDrawdownGuard(enabled=True)
        assert guard._day_reset_tz.utcoffset(None).total_seconds() == 0

    def test_jst_offset(self) -> None:
        """JST (offset=9) で構築すると +9h の timezone が設定される."""
        guard = DailyDrawdownGuard(enabled=True, day_reset_utc_offset_hours=9.0)
        assert guard._day_reset_tz.utcoffset(None).total_seconds() == 9 * 3600

    def test_jst_day_reset_at_midnight_jst(self) -> None:
        """JST モードの日付リセットが JST 00:00 で発生する.

        UTC 15:00 = JST 00:00 で日替わり → halt 解除。
        UTC ベースだと 22h 以上かかるケースが 15h 以下に短縮。
        """
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            day_reset_utc_offset_hours=9.0,
        )
        # JST 3/3 10:51 に halt 発火相当
        jst_halt_time = datetime(2026, 3, 3, 1, 51, tzinfo=timezone.utc)  # UTC 01:51 = JST 10:51
        jst_day_str = "20260303"  # JST ベースの日付
        guard._state.current_day = jst_day_str
        guard._state.halted = True

        # JST 3/3 23:59 (= UTC 14:59) → まだ同日
        with patch.object(
            DailyDrawdownGuard, '_today', return_value="20260303"
        ):
            assert guard.maybe_reset_day() is False
            assert guard._state.halted is True

        # JST 3/4 00:00 (= UTC 15:00) → 日替わり → リセット
        with patch.object(
            DailyDrawdownGuard, '_today', return_value="20260304"
        ):
            assert guard.maybe_reset_day() is True
            assert guard._state.halted is False  # 新しい日 → halt 解除

    def test_utc_mode_worst_case_is_22h(self) -> None:
        """UTC ベースだと JST 10:51 発火 → リセットまで ~22h.

        268# 根本原因: DD halt at UTC 01:51 → UTC day change at UTC 00:00+1d = 22h09m。
        JST モードなら JST 10:51 → JST 00:00+1d = 13h09m に短縮。
        """
        # UTC ベース
        halt_utc = datetime(2026, 3, 3, 1, 51, tzinfo=timezone.utc)  # = JST 10:51
        next_utc_day = datetime(2026, 3, 4, 0, 0, tzinfo=timezone.utc)
        utc_wait = (next_utc_day - halt_utc).total_seconds() / 3600
        assert utc_wait == pytest.approx(22.15, abs=0.01)

        # JST ベース
        jst = timezone(timedelta(hours=9))
        halt_jst = halt_utc.astimezone(jst)  # JST 10:51
        next_jst_day = datetime(2026, 3, 4, 0, 0, tzinfo=jst)
        jst_wait = (next_jst_day - halt_jst).total_seconds() / 3600
        assert jst_wait == pytest.approx(13.15, abs=0.01)
        assert jst_wait < utc_wait  # JST の方が短い

    def test_config_default_is_jst(self) -> None:
        """fill_config のデフォルトが JST (9.0) であること."""
        cfg = FillTestConfig()
        assert cfg.dd_day_reset_utc_offset_hours == 9.0

    def test_config_yaml_parsing_tz(self) -> None:
        """YAML から day_reset_utc_offset_hours がパースされる."""
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "loss_control": {
                        "daily_drawdown": {
                            "enabled": True,
                            "hard_limit_bps": -50.0,
                            "soft_limit_bps": -30.0,
                            "day_reset_utc_offset_hours": 9.0,
                        }
                    }
                }
            )
        )
        assert cfg.dd_day_reset_utc_offset_hours == 9.0

    def test_today_uses_configured_tz(self) -> None:
        """_today() がコンストラクタで設定した TZ を使用する."""
        utc_guard = DailyDrawdownGuard(enabled=True, day_reset_utc_offset_hours=0.0)
        jst_guard = DailyDrawdownGuard(enabled=True, day_reset_utc_offset_hours=9.0)

        utc_today = utc_guard._today()
        jst_today = jst_guard._today()

        # 両方とも YYYYMMDD 形式
        assert len(utc_today) == 8
        assert len(jst_today) == 8
        assert utc_today.isdigit()
        assert jst_today.isdigit()

    def test_import_state_respects_configured_tz(self) -> None:
        """import_state が設定 TZ の today で stale 判定する."""
        guard = DailyDrawdownGuard(
            enabled=True,
            day_reset_utc_offset_hours=9.0,
        )
        today_jst = guard._today()

        # 同日の state → 正常復元
        state = {"current_day": today_jst, "halted": True, "daily_pnl_bps": -50.0}
        guard.import_state(state)
        assert guard._state.halted is True

        # 異なる日の state → stale で無視
        guard2 = DailyDrawdownGuard(
            enabled=True,
            day_reset_utc_offset_hours=9.0,
        )
        state2 = {"current_day": "20200101", "halted": True, "daily_pnl_bps": -50.0}
        guard2.import_state(state2)
        assert guard2._state.halted is False  # stale → 未復元


# ======================================================================
# 326# warmup_from_records encapsulation テスト
# ======================================================================


class TestWarmupFromRecords326:
    """326# DailyDrawdownGuard.warmup_from_records() のユニットテスト."""

    def test_warmup_basic_pnl_aggregation(self) -> None:
        """当日分のみ集計する."""
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-50.0)
        today_str = guard._today()
        now = time.time()
        yesterday = now - 86400 * 2  # 2 days ago (definitely different day)

        records = [
            (now, -10.0, "buy"),
            (now, -5.0, "sell"),
            (yesterday, -100.0, "buy"),  # 前日 → 除外
        ]
        count = guard.warmup_from_records(records)
        assert count == 2
        assert guard.state.daily_pnl_bps == pytest.approx(-15.0)
        assert guard.state.daily_fill_count == 2
        assert guard.state.daily_pnl_bps_buy == pytest.approx(-10.0)
        assert guard.state.daily_pnl_bps_sell == pytest.approx(-5.0)

    def test_warmup_triggers_hard_halt(self) -> None:
        """warmup で hard limit を超過すると halted=True."""
        guard = DailyDrawdownGuard(enabled=True, hard_limit_bps=-20.0)
        now = time.time()
        records = [(now, -25.0, "buy")]
        guard.warmup_from_records(records)
        assert guard.state.halted is True

    def test_warmup_triggers_soft(self) -> None:
        """warmup で soft limit を超過すると _soft_triggered_today=True."""
        guard = DailyDrawdownGuard(
            enabled=True, soft_limit_bps=-10.0, hard_limit_bps=-50.0,
        )
        now = time.time()
        records = [(now, -15.0, "sell")]
        guard.warmup_from_records(records)
        assert guard._soft_triggered_today is True
        assert guard.state.halted is False

    def test_warmup_per_side_halt(self) -> None:
        """per-side halt が warmup で発動する."""
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-100.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=-20.0,
            per_side_halt_cycles=3,
        )
        now = time.time()
        records = [(now, -25.0, "sell")]
        guard.warmup_from_records(records)
        assert guard.state.side_halted_sell is True
        assert guard.state.side_halt_remaining_sell == 3
        assert guard.state.side_halted_buy is False

    def test_warmup_empty_records_returns_zero(self) -> None:
        """空リストでは 0 を返し state は変更しない."""
        guard = DailyDrawdownGuard(enabled=True)
        count = guard.warmup_from_records([])
        assert count == 0
        assert guard.state.daily_fill_count == 0

