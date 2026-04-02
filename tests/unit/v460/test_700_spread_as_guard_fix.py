from __future__ import annotations

import pytest

from scripts.v460.analysis.sections.section_spread_distribution import (
    build_spread_distribution_section,
)
from scripts.v460.lib.entry_gate_adjustments import apply_entry_gate_adjustments
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_validation import validate_fill_config
from scripts.v460.lib.regime_detector import FillTestRegime


def test_guard_fires_at_correct_bps() -> None:
    cfg = FillTestConfig(
        spread_as_guard_enabled=True,
        spread_as_guard_spread_threshold_bps=15.0,
        spread_as_guard_ev_penalty_bps=0.5,
    )

    result = apply_entry_gate_adjustments(
        config=cfg,
        regime=FillTestRegime.RANGING,
        spread_bps=10.0,
        base_ev_bps=2.0,
    )

    assert result.adjusted_ev_bps == pytest.approx(1.5)
    assert result.spread_as_guard_triggered is True


def test_boundary_exact_threshold_passes() -> None:
    cfg = FillTestConfig(
        spread_as_guard_enabled=True,
        spread_as_guard_spread_threshold_bps=15.0,
        spread_as_guard_ev_penalty_bps=0.5,
    )

    result = apply_entry_gate_adjustments(
        config=cfg,
        regime=FillTestRegime.RANGING,
        spread_bps=15.0,
        base_ev_bps=2.0,
    )

    assert result.adjusted_ev_bps == pytest.approx(2.0)
    assert result.spread_as_guard_triggered is False


def test_config_validation_rejects_extreme_threshold() -> None:
    cfg = FillTestConfig()
    cfg.spread_as_guard_spread_threshold_bps = 1500.0

    with pytest.raises(ValueError, match="spread_as_guard_spread_threshold_bps"):
        validate_fill_config(cfg)


def test_backward_compat_threshold_key_is_supported() -> None:
    cfg = FillTestConfig.from_yaml(
        {
            "spread_as_guard": {
                "enabled": True,
                "threshold": 12.0,
                "ev_penalty": 0.25,
            }
        }
    )

    assert cfg.spread_as_guard_spread_threshold_bps == pytest.approx(12.0)
    assert cfg.spread_as_guard_ev_penalty_bps == pytest.approx(0.25)


def test_spread_distribution_estimates_threshold_impact() -> None:
    payload = build_spread_distribution_section(
        [
            {"filled": True, "spread_bps": 4.0, "adverse_selected": False},
            {"filled": True, "spread_bps": 9.0, "adverse_selected": False},
            {"filled": True, "spread_bps": 16.0, "adverse_selected": True},
            {"filled": True, "spread_bps": 21.0, "adverse_selected": True},
        ]
    )

    assert payload["quantiles_bps"]["p50"] == pytest.approx(12.5)
    assert payload["threshold_impact"]["15.0"]["count"] == 2
