from __future__ import annotations

import pytest

from scripts.v460.lib.entry_gate_adjustments import apply_entry_gate_adjustments
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.regime_detector import FillTestRegime


class TestSpreadASGuard:
    def test_guard_triggers_below_threshold(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=True,
            spread_as_guard_spread_threshold_bps=1500.0,
            spread_as_guard_ev_penalty_bps=0.5,
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime=FillTestRegime.RANGING,
            spread_bps=1200.0,
            base_ev_bps=0.4,
        )

        assert result.spread_as_guard_triggered is True
        assert result.spread_as_guard_action == "apply"
        assert result.adjusted_ev_bps == pytest.approx(-0.1)

    def test_guard_skips_above_threshold(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=True,
            spread_as_guard_spread_threshold_bps=1500.0,
            spread_as_guard_ev_penalty_bps=0.5,
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime=FillTestRegime.RANGING,
            spread_bps=2000.0,
            base_ev_bps=0.4,
        )

        assert result.spread_as_guard_triggered is False
        assert result.spread_as_guard_action == "none"
        assert result.adjusted_ev_bps == pytest.approx(0.4)

    def test_guard_disabled_observe_mode(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=False,
            spread_as_guard_spread_threshold_bps=1500.0,
            spread_as_guard_ev_penalty_bps=0.5,
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime=FillTestRegime.RANGING,
            spread_bps=1000.0,
            base_ev_bps=0.4,
        )

        assert result.spread_as_guard_triggered is True
        assert result.spread_as_guard_action == "observe"
        assert result.adjusted_ev_bps == pytest.approx(0.4)

    def test_config_hot_reload_fields_registered(self) -> None:
        expected = {
            "spread_as_guard_enabled",
            "spread_as_guard_spread_threshold_bps",
            "spread_as_guard_ev_penalty_bps",
            "regime_guard_overrides_enabled",
            "regime_guard_ev_threshold_premiums",
            "regime_guard_spread_as_penalty_multipliers",
        }
        assert expected <= _HOT_RELOADABLE_FIELDS

