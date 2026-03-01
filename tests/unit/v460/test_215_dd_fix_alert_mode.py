"""215# P0-A/C: DD state consistency repair & alert_mode unit tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from scripts.v460.lib.alert_mode import AlertModeOverride, load_alert_mode
from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard


# ======================================================================
# P0-A: DailyDrawdownGuard — needs_warmup_repair & soft trigger fix
# ======================================================================

class TestDDNeedsWarmupRepair:
    """215# P0-A: import_state 後の整合性検証."""

    def _make_guard(self, **kw: object) -> DailyDrawdownGuard:
        defaults = dict(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=-30.0,
        )
        defaults.update(kw)
        return DailyDrawdownGuard(**defaults)  # type: ignore[arg-type]

    def test_no_repair_when_disabled(self) -> None:
        guard = DailyDrawdownGuard(enabled=False)
        assert not guard.needs_warmup_repair()

    def test_no_repair_when_no_fills(self) -> None:
        guard = self._make_guard()
        guard.maybe_reset_day()
        assert not guard.needs_warmup_repair()

    def test_repair_needed_per_side_zero(self) -> None:
        """per-side PnL が 0.0 なのに total PnL が有意 → repair 必要."""
        guard = self._make_guard()
        today = guard._utc_today()
        guard.import_state({
            "current_day": today,
            "daily_pnl_bps": -110.94,
            "daily_fill_count": 29,
            "halted": True,
            "soft_triggered_today": False,
            "daily_pnl_bps_buy": 0.0,
            "daily_pnl_bps_sell": 0.0,
        })
        assert guard.needs_warmup_repair()

    def test_repair_needed_soft_inconsistency(self) -> None:
        """soft_triggered_today=false だが PnL < soft_limit → repair."""
        guard = self._make_guard()
        today = guard._utc_today()
        guard.import_state({
            "current_day": today,
            "daily_pnl_bps": -40.0,
            "daily_fill_count": 10,
            "halted": False,
            "soft_triggered_today": False,
            "daily_pnl_bps_buy": -20.0,
            "daily_pnl_bps_sell": -20.0,
        })
        assert guard.needs_warmup_repair()

    def test_no_repair_when_consistent(self) -> None:
        """per-side PnL と soft_triggered が正しい → repair 不要."""
        guard = self._make_guard()
        today = guard._utc_today()
        guard.import_state({
            "current_day": today,
            "daily_pnl_bps": -40.0,
            "daily_fill_count": 10,
            "halted": False,
            "soft_triggered_today": True,
            "daily_pnl_bps_buy": -15.0,
            "daily_pnl_bps_sell": -25.0,
        })
        assert not guard.needs_warmup_repair()


class TestDDSoftTriggerFix:
    """215# P0-A: hard halt 時に soft trigger もセットされることを検証."""

    def test_soft_triggered_on_hard_halt(self) -> None:
        """PnL が一気に hard limit を超えた場合でも soft_triggered_today=True."""
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
        )
        guard.maybe_reset_day()
        # 一発で -60bps → hard limit 超過
        result = guard.update_pnl(-60.0, side="buy")
        assert result["halted"] is True
        # 215#: soft も同時に trigger される (旧: if/elif で skip されていた)
        assert guard._soft_triggered_today is True

    def test_soft_triggers_before_hard(self) -> None:
        """段階的に PnL が下がる場合の通常動作."""
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
        )
        guard.maybe_reset_day()
        # step 1: -35bps → soft trigger
        r1 = guard.update_pnl(-35.0, side="sell")
        assert r1["soft_triggered"] is True
        assert r1["halted"] is False
        # step 2: -20bps → total -55bps → hard halt
        r2 = guard.update_pnl(-20.0, side="sell")
        assert r2["halted"] is True
        # soft should not re-trigger (already triggered)
        assert r2["soft_triggered"] is False


# ======================================================================
# P0-C: alert_mode.json
# ======================================================================

class TestAlertMode:
    """215# P0-C: ファイルタッチ型 alert_mode."""

    def setup_method(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        # Reset module-level cache
        import scripts.v460.lib.alert_mode as am
        am._last_logged_state = None

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_file_returns_inactive(self) -> None:
        result = load_alert_mode(self._tmpdir)
        assert not result.is_active
        assert not result.halt
        assert result.offset_mult == 1.0
        assert result.lot_mult == 1.0
        assert result.interval_mult == 1.0

    def test_halt_mode(self) -> None:
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text(json.dumps({"halt": True, "reason": "test"}))
        result = load_alert_mode(self._tmpdir)
        assert result.halt is True
        assert result.reason == "test"
        assert result.is_active

    def test_degraded_mode(self) -> None:
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text(json.dumps({
            "offset_mult": 2.0,
            "lot_mult": 0.5,
            "interval_mult": 3.0,
        }))
        result = load_alert_mode(self._tmpdir)
        assert not result.halt
        assert result.offset_mult == 2.0
        assert result.lot_mult == 0.5
        assert result.interval_mult == 3.0
        assert result.is_active

    def test_lot_mult_clamped(self) -> None:
        """lot_mult は 0.01~1.0 にクランプ."""
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text(json.dumps({"lot_mult": 5.0}))
        result = load_alert_mode(self._tmpdir)
        assert result.lot_mult == 1.0
        path.write_text(json.dumps({"lot_mult": 0.001}))
        # Reset cache
        import scripts.v460.lib.alert_mode as am
        am._last_logged_state = None
        result = load_alert_mode(self._tmpdir)
        assert result.lot_mult == 0.01

    def test_interval_mult_floor(self) -> None:
        """interval_mult は 1.0 以上."""
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text(json.dumps({"interval_mult": 0.5}))
        result = load_alert_mode(self._tmpdir)
        assert result.interval_mult == 1.0

    def test_invalid_json_returns_inactive(self) -> None:
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text("{invalid json")
        result = load_alert_mode(self._tmpdir)
        assert not result.is_active

    def test_empty_file_returns_inactive(self) -> None:
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text("")
        result = load_alert_mode(self._tmpdir)
        assert not result.is_active

    def test_non_dict_json_returns_inactive(self) -> None:
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text("[1, 2, 3]")
        result = load_alert_mode(self._tmpdir)
        assert not result.is_active

    def test_file_removal_clears_override(self) -> None:
        path = Path(self._tmpdir) / "alert_mode.json"
        path.write_text(json.dumps({"halt": True}))
        r1 = load_alert_mode(self._tmpdir)
        assert r1.halt is True
        os.remove(path)
        r2 = load_alert_mode(self._tmpdir)
        assert not r2.is_active


class TestAlertModeOverride:
    """AlertModeOverride dataclass tests."""

    def test_default_is_inactive(self) -> None:
        o = AlertModeOverride()
        assert not o.is_active

    def test_halt_is_active(self) -> None:
        o = AlertModeOverride(halt=True)
        assert o.is_active

    def test_offset_change_is_active(self) -> None:
        o = AlertModeOverride(offset_mult=1.5)
        assert o.is_active

    def test_frozen(self) -> None:
        o = AlertModeOverride(halt=True)
        with pytest.raises(AttributeError):
            o.halt = False  # type: ignore[misc]
