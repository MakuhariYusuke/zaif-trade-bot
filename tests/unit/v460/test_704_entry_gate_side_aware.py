from __future__ import annotations

import time

import pytest

from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS
from scripts.v460.lib.entry_gate_guard import EntryGateGuard, EntryGateGuardConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_validation import validate_fill_config
from tests.unit.v460._yaml_test_helpers import parse_yaml_mapping


def _make_guard() -> EntryGateGuard:
    guard = EntryGateGuard(
        EntryGateGuardConfig(
            max_consecutive_blocks=50,
            max_block_rate=0.95,
            min_eval_count_for_rate=20,
            staleness_threshold_sec=600.0,
            buy_suppress_ev_threshold=-0.5,
        )
    )
    guard.notify_calibration_update()
    return guard


def test_buy_suppress_mild_negative_ev() -> None:
    assert _make_guard().should_suppress_block(ev=-0.3, regime="ranging", side="buy") is True


def test_buy_block_severe_negative_ev() -> None:
    assert _make_guard().should_suppress_block(ev=-1.0, regime="ranging", side="buy") is False


def test_sell_not_suppressed() -> None:
    assert _make_guard().should_suppress_block(ev=-0.3, regime="ranging", side="sell") is False


def test_buy_suppress_threshold_boundary() -> None:
    assert _make_guard().should_suppress_block(ev=-0.5, regime="ranging", side="buy") is True


def test_auto_disable_still_works() -> None:
    guard = _make_guard()
    guard.state.auto_disabled = True
    assert guard.should_suppress_block(ev=-1.0, regime="ranging", side="sell") is True


def test_staleness_overrides_side_aware() -> None:
    guard = _make_guard()
    guard.state.last_calibration_update_ts = time.time() - 1200.0
    assert guard.should_suppress_block(ev=-0.3, regime="ranging", side="buy") is True
    assert "stale" in guard.state.auto_disable_reason


def test_fill_config_parses_buy_suppress_threshold_flat() -> None:
    cfg = FillTestConfig.from_yaml(
        parse_yaml_mapping(
            """
entry_gate_buy_suppress_ev_threshold: -0.4
"""
        )
    )
    assert cfg.entry_gate_buy_suppress_ev_threshold == pytest.approx(-0.4)


def test_fill_config_parses_buy_suppress_threshold_nested() -> None:
    cfg = FillTestConfig.from_yaml(
        parse_yaml_mapping(
            """
entry_gate:
  buy_suppress_ev_threshold: -0.25
"""
        )
    )
    assert cfg.entry_gate_buy_suppress_ev_threshold == pytest.approx(-0.25)


def test_buy_suppress_threshold_validation_range() -> None:
    with pytest.raises(ValueError, match="entry_gate_buy_suppress_ev_threshold"):
        FillTestConfig(entry_gate_buy_suppress_ev_threshold=0.1)


def test_buy_suppress_threshold_is_hot_reloadable() -> None:
    assert "entry_gate_buy_suppress_ev_threshold" in _HOT_RELOADABLE_FIELDS


def test_live_yaml_matches_entry_gate_side_aware(v460_fill_test_config_base: FillTestConfig) -> None:
    assert v460_fill_test_config_base.entry_gate_max_consecutive_blocks == 50
    assert v460_fill_test_config_base.entry_gate_max_block_rate == pytest.approx(0.95)
    assert v460_fill_test_config_base.entry_gate_buy_suppress_ev_threshold == pytest.approx(-0.5)
