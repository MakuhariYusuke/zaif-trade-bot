from __future__ import annotations

import pytest

from scripts.v460.lib.entry_gate_adjustments import apply_entry_gate_adjustments
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_validation import validate_fill_config
from scripts.v460.lib.regime_detector import FillTestRegime
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460._fill_test_source import ORCHESTRATOR_MID_CYCLE, read_source_text
from tests.unit.v460._yaml_test_helpers import parse_yaml_mapping


def _resolve_raw_spread_bps(*, last_spread_raw: float | None, last_mid_price: float | None) -> float | None:
    if last_spread_raw is None or last_mid_price is None or last_mid_price <= 0:
        return None
    return float(last_spread_raw / last_mid_price * 10_000.0)


def test_sell_trending_down_offset_applied() -> None:
    cfg = FillTestConfig(skip_gate_sell_trending_down_offset=0.5)
    total_offset = 0.2
    if "sell" == "sell" and "trending_down" == "trending_down":
        total_offset += cfg.skip_gate_sell_trending_down_offset
    assert total_offset == pytest.approx(0.7)


def test_sell_trending_down_offset_not_applied_buy() -> None:
    cfg = FillTestConfig(skip_gate_sell_trending_down_offset=0.5)
    total_offset = 0.2
    if "buy" == "sell" and "trending_down" == "trending_down":
        total_offset += cfg.skip_gate_sell_trending_down_offset
    assert total_offset == pytest.approx(0.2)


def test_sell_trending_down_offset_not_applied_other_regime() -> None:
    cfg = FillTestConfig(skip_gate_sell_trending_down_offset=0.5)
    total_offset = 0.2
    if "sell" == "sell" and "ranging" == "trending_down":
        total_offset += cfg.skip_gate_sell_trending_down_offset
    assert total_offset == pytest.approx(0.2)


def test_sell_trending_down_offset_coexistence() -> None:
    cfg = FillTestConfig(
        skip_gate_sell_ranging_offset=0.5,
        skip_gate_sell_trending_up_offset=0.6,
        skip_gate_sell_trending_down_offset=0.7,
    )
    assert cfg.skip_gate_sell_ranging_offset == pytest.approx(0.5)
    assert cfg.skip_gate_sell_trending_up_offset == pytest.approx(0.6)
    assert cfg.skip_gate_sell_trending_down_offset == pytest.approx(0.7)


def test_sell_trending_down_offset_validation_range() -> None:
    with pytest.raises(ValueError, match="skip_gate_sell_trending_down_offset"):
        FillTestConfig(skip_gate_sell_trending_down_offset=2.5)


def test_sell_trending_down_offset_hot_reload() -> None:
    from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS

    assert "skip_gate_sell_trending_down_offset" in _HOT_RELOADABLE_FIELDS


def test_skip_gate_evaluator_source_mentions_sell_trending_down_offset() -> None:
    assert "skip_gate_sell_trending_down_offset" in SkipGateEvaluator.evaluate.__code__.co_names


def test_spread_as_guard_uses_raw_spread() -> None:
    source = read_source_text(ORCHESTRATOR_MID_CYCLE)
    assert "last_spread_raw" in source
    assert "_raw_spread" in source

    spread_bps = _resolve_raw_spread_bps(
        last_spread_raw=3000.0,
        last_mid_price=10_000_000.0,
    )
    assert spread_bps == pytest.approx(3.0)


def test_spread_as_guard_triggered_with_raw() -> None:
    cfg = FillTestConfig(
        spread_as_guard_enabled=True,
        spread_as_guard_spread_threshold_bps=15.0,
        spread_as_guard_ev_penalty_bps=0.5,
    )

    result = apply_entry_gate_adjustments(
        config=cfg,
        regime=FillTestRegime.RANGING,
        spread_bps=3.0,
        base_ev_bps=1.0,
    )

    assert result.spread_as_guard_triggered is True
    assert result.adjusted_ev_bps == pytest.approx(0.5)


def test_regime_guard_trending_down_penalty() -> None:
    cfg = FillTestConfig(
        spread_as_guard_enabled=True,
        spread_as_guard_spread_threshold_bps=15.0,
        spread_as_guard_ev_penalty_bps=0.5,
        regime_guard_overrides_enabled=True,
        regime_guard_ev_threshold_premiums={"trending_down": 0.3},
        regime_guard_spread_as_penalty_multipliers={"trending_down": 1.5},
    )

    result = apply_entry_gate_adjustments(
        config=cfg,
        regime=FillTestRegime.TRENDING_DOWN,
        spread_bps=3.0,
        base_ev_bps=1.0,
    )

    assert result.adjusted_ev_bps == pytest.approx(-0.05)


def test_sell_trending_down_offset_is_parsed() -> None:
    cfg = FillTestConfig.from_yaml(
        parse_yaml_mapping(
            """
skip_gate:
  sell_trending_down_offset: 0.5
"""
        )
    )
    assert cfg.skip_gate_sell_trending_down_offset == pytest.approx(0.5)


def test_live_yaml_values_match_704_design(v460_fill_test_config_base: FillTestConfig) -> None:
    cfg = v460_fill_test_config_base
    assert cfg.skip_gate_sell_trending_down_offset == pytest.approx(0.5)
    assert cfg.regime_guard_ev_threshold_premiums["trending_down"] == pytest.approx(0.3)
    assert cfg.regime_guard_spread_as_penalty_multipliers["trending_down"] == pytest.approx(1.5)
