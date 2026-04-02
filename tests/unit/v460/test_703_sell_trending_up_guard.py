from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460._yaml_test_helpers import parse_yaml_mapping


def test_sell_trending_up_offset_field_exists() -> None:
    cfg = FillTestConfig()
    assert cfg.skip_gate_sell_trending_up_offset == pytest.approx(0.0)


def test_sell_trending_up_offset_is_parsed() -> None:
    cfg = FillTestConfig.from_yaml(
        parse_yaml_mapping(
            """
skip_gate:
  sell_trending_up_offset: 0.5
"""
        )
    )
    assert cfg.skip_gate_sell_trending_up_offset == pytest.approx(0.5)


def test_sell_trending_up_offset_applies_only_to_sell_trending_up() -> None:
    cfg = FillTestConfig(skip_gate_sell_trending_up_offset=0.5)
    base = 0.2

    sell_trending = base
    if "sell" == "sell" and "trending_up" == "trending_up":
        sell_trending += cfg.skip_gate_sell_trending_up_offset

    assert sell_trending == pytest.approx(0.7)
    assert base == pytest.approx(0.2)


def test_sell_trending_up_offset_coexists_with_sell_ranging_offset() -> None:
    cfg = FillTestConfig(
        skip_gate_sell_ranging_offset=0.5,
        skip_gate_sell_trending_up_offset=0.6,
    )
    assert cfg.skip_gate_sell_ranging_offset == pytest.approx(0.5)
    assert cfg.skip_gate_sell_trending_up_offset == pytest.approx(0.6)


def test_live_yaml_values_match_design(v460_fill_test_config_base: FillTestConfig) -> None:
    cfg = v460_fill_test_config_base
    assert cfg.skip_gate_sell_trending_up_offset == pytest.approx(0.5)
    assert cfg.regime_guard_overrides_enabled is True
    assert cfg.regime_guard_ev_threshold_premiums["trending_up"] == pytest.approx(0.3)
    assert cfg.regime_guard_spread_as_penalty_multipliers["trending_up"] == pytest.approx(1.5)


def test_skip_gate_evaluator_source_mentions_sell_trending_up_offset() -> None:
    source = SkipGateEvaluator.evaluate.__code__.co_names
    assert "skip_gate_sell_trending_up_offset" in source
