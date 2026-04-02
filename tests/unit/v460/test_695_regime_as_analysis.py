from __future__ import annotations

import pytest

from scripts.v460.analysis.sections.section_regime_as_deep_dive import (
    build_regime_as_deep_dive_section,
)
from scripts.v460.lib.entry_gate_adjustments import apply_entry_gate_adjustments
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.regime_detector import FillTestRegime


def _record(
    *,
    ts: float,
    regime: str,
    spread_at_order: float,
    pnl: float,
    adverse_selected: bool,
    skip_gate_bypassed: bool = False,
) -> dict[str, object]:
    return {
        "cycle_id": f"r_{ts}",
        "timestamp": ts,
        "side": "sell",
        "filled": True,
        "regime": regime,
        "spread_at_order": spread_at_order,
        "post_fill_30s_pnl": pnl,
        "adverse_selected": adverse_selected,
        "skip_gate_bypassed": skip_gate_bypassed,
    }


class TestRegimeASAnalysis:
    def test_crosstab_regime_spread_as(self) -> None:
        payload = build_regime_as_deep_dive_section(
            [
                _record(ts=1.0, regime="ranging", spread_at_order=1200.0, pnl=-1.0, adverse_selected=True),
                _record(ts=2.0, regime="ranging", spread_at_order=1800.0, pnl=0.5, adverse_selected=False),
                _record(ts=3.0, regime="trending_down", spread_at_order=2600.0, pnl=1.2, adverse_selected=False),
            ]
        )

        ranging_bucket = payload["regime_spread_crosstab"]["ranging"]["0_1500"]
        assert ranging_bucket["filled"] == 1
        assert ranging_bucket["adverse_selection_rate_pct"] == pytest.approx(100.0)

    def test_regime_guard_adapter_ranging(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=True,
            spread_as_guard_spread_threshold_bps=15.0,
            spread_as_guard_ev_penalty_bps=0.5,
            regime_guard_overrides_enabled=True,
            regime_guard_ev_threshold_premiums={"ranging": 0.3},
            regime_guard_spread_as_penalty_multipliers={"ranging": 1.5},
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime=FillTestRegime.RANGING,
            spread_bps=12.0,
            base_ev_bps=1.2,
        )

        assert result.regime_guard_ev_premium_bps == pytest.approx(0.3)
        assert result.regime_guard_penalty_multiplier == pytest.approx(1.5)
        assert result.adjusted_ev_bps == pytest.approx(1.2 - 0.75 - 0.3)

    def test_regime_guard_adapter_trending(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=True,
            spread_as_guard_spread_threshold_bps=15.0,
            spread_as_guard_ev_penalty_bps=0.5,
            regime_guard_overrides_enabled=True,
            regime_guard_ev_threshold_premiums={"trending_down": 0.0},
            regime_guard_spread_as_penalty_multipliers={"trending_down": 1.0},
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime=FillTestRegime.TRENDING_DOWN,
            spread_bps=12.0,
            base_ev_bps=1.0,
        )

        assert result.adjusted_ev_bps == pytest.approx(0.5)

    def test_regime_guard_adapter_disabled(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=True,
            spread_as_guard_spread_threshold_bps=15.0,
            spread_as_guard_ev_penalty_bps=0.5,
            regime_guard_overrides_enabled=False,
            regime_guard_ev_threshold_premiums={"ranging": 0.3},
            regime_guard_spread_as_penalty_multipliers={"ranging": 1.5},
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime=FillTestRegime.RANGING,
            spread_bps=12.0,
            base_ev_bps=1.0,
        )

        assert result.regime_guard_ev_premium_bps == pytest.approx(0.0)
        assert result.regime_guard_penalty_multiplier == pytest.approx(1.0)

    def test_regime_guard_adapter_unknown_regime(self) -> None:
        cfg = FillTestConfig(
            spread_as_guard_enabled=True,
            spread_as_guard_spread_threshold_bps=15.0,
            spread_as_guard_ev_penalty_bps=0.5,
            regime_guard_overrides_enabled=True,
            regime_guard_ev_threshold_premiums={"ranging": 0.3},
            regime_guard_spread_as_penalty_multipliers={"ranging": 1.5},
        )

        result = apply_entry_gate_adjustments(
            config=cfg,
            regime="mystery_regime",
            spread_bps=12.0,
            base_ev_bps=1.0,
        )

        assert result.regime_guard_ev_premium_bps == pytest.approx(0.0)
        assert result.regime_guard_penalty_multiplier == pytest.approx(1.0)

    def test_spread_distribution_is_exposed(self) -> None:
        payload = build_regime_as_deep_dive_section(
            [
                {
                    "cycle_id": "a",
                    "timestamp": 1.0,
                    "side": "sell",
                    "filled": True,
                    "regime": "ranging",
                    "spread_bps": 6.0,
                    "spread_at_order": 900.0,
                    "mid_at_order": 1_500_000.0,
                    "post_fill_30s_pnl": 1.0,
                    "adverse_selected": False,
                },
                {
                    "cycle_id": "b",
                    "timestamp": 2.0,
                    "side": "sell",
                    "filled": True,
                    "regime": "trending_down",
                    "spread_bps": 18.0,
                    "spread_at_order": 2_700.0,
                    "mid_at_order": 1_500_000.0,
                    "post_fill_30s_pnl": -1.0,
                    "adverse_selected": True,
                },
            ]
        )

        assert payload["spread_distribution"]["quantiles_bps"]["p50"] == pytest.approx(12.0)
        assert payload["spread_distribution"]["threshold_impact"]["15.0"]["count"] == 1
