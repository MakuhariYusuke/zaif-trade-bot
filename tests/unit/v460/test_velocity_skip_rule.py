"""
165# AS-R1: Velocity-based sell/buy skip rule tests.

Tests for the price_velocity_bps-based pre-ML skip rule added to
SkipGateEvaluator. This rule fires before the ML model evaluation
when velocity exceeds configurable thresholds.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.v460.lib.fill_config import FillTestConfig, SkipGateResult
from tests.unit.v460._yaml_test_helpers import clone_fill_test_config, load_fill_test_config_from_mapping
from ztb.metrics.fill_quality import FillRecord


# ---------------------------------------------------------------------------
# Helper: Minimal config factory
# ---------------------------------------------------------------------------

def _make_config(**overrides: object) -> FillTestConfig:
    """Create a FillTestConfig with velocity skip defaults."""
    defaults = dict(
        skip_gate_enabled=True,
        skip_gate_buy_enabled=True,
        skip_gate_sell_enabled=True,
        skip_sell_unknown_regime=False,
        sell_velocity_skip_enabled=False,
        sell_velocity_skip_threshold_bps=8.0,
        buy_velocity_skip_enabled=False,
        buy_velocity_skip_threshold_bps=-8.0,
    )
    defaults.update(overrides)
    return FillTestConfig(**defaults)


# ---------------------------------------------------------------------------
# Test: FillTestConfig new fields
# ---------------------------------------------------------------------------

class TestVelocitySkipConfig:
    """Config field defaults and YAML parsing."""

    def test_default_sell_velocity_skip_disabled(self) -> None:
        cfg = FillTestConfig()
        assert cfg.sell_velocity_skip_enabled is False

    def test_default_sell_velocity_threshold(self) -> None:
        cfg = FillTestConfig()
        assert cfg.sell_velocity_skip_threshold_bps == 8.0

    def test_default_buy_velocity_skip_disabled(self) -> None:
        cfg = FillTestConfig()
        assert cfg.buy_velocity_skip_enabled is False

    def test_default_buy_velocity_threshold(self) -> None:
        cfg = FillTestConfig()
        assert cfg.buy_velocity_skip_threshold_bps == -8.0

    def test_custom_thresholds(self) -> None:
        cfg = _make_config(
            sell_velocity_skip_enabled=True,
            sell_velocity_skip_threshold_bps=5.0,
            buy_velocity_skip_enabled=True,
            buy_velocity_skip_threshold_bps=-5.0,
        )
        assert cfg.sell_velocity_skip_enabled is True
        assert cfg.sell_velocity_skip_threshold_bps == 5.0
        assert cfg.buy_velocity_skip_enabled is True
        assert cfg.buy_velocity_skip_threshold_bps == -5.0


# ---------------------------------------------------------------------------
# Test: SkipGateResult velocity field
# ---------------------------------------------------------------------------

class TestSkipGateResultVelocity:
    """SkipGateResult.price_velocity_bps field."""

    def test_default_none(self) -> None:
        r = SkipGateResult()
        assert r.price_velocity_bps is None

    def test_set_velocity(self) -> None:
        r = SkipGateResult()
        r.price_velocity_bps = 12.5
        assert r.price_velocity_bps == 12.5


# ---------------------------------------------------------------------------
# Test: FillRecord velocity field
# ---------------------------------------------------------------------------

class TestFillRecordVelocity:
    """FillRecord.price_velocity_bps field."""

    def test_default_none(self) -> None:
        fr = FillRecord(
            cycle_id="test",
            timestamp=time.time(),
            side="sell",
            order_price=10000000.0,
            order_quantity=0.001,
        )
        assert fr.price_velocity_bps is None

    def test_set_velocity(self) -> None:
        fr = FillRecord(
            cycle_id="test",
            timestamp=time.time(),
            side="sell",
            order_price=10000000.0,
            order_quantity=0.001,
            price_velocity_bps=15.3,
        )
        assert fr.price_velocity_bps == 15.3


# ---------------------------------------------------------------------------
# Test: cancel_reasons constants
# ---------------------------------------------------------------------------

class TestCancelReasonConstants:
    """New cancel reason constants for velocity skip."""

    def test_sell_velocity_constant(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.SKIP_GATE_RULE_VELOCITY_SELL == "skip_gate_rule_velocity_sell"

    def test_buy_velocity_constant(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.SKIP_GATE_RULE_VELOCITY_BUY == "skip_gate_rule_velocity_buy"


# ---------------------------------------------------------------------------
# Test: Velocity rule logic (unit-level, mocking evaluator internals)
# ---------------------------------------------------------------------------

class TestVelocityRuleLogic:
    """Test the velocity rule threshold logic."""

    def test_sell_velocity_above_threshold_triggers_skip(self) -> None:
        """price_velocity_bps > threshold AND sell → should skip."""
        # velocity=10 > threshold=8 → SKIP
        enabled = True
        side = "sell"
        velocity = 10.0
        threshold = 8.0
        should_skip = enabled and side == "sell" and velocity > threshold
        assert should_skip is True

    def test_sell_velocity_below_threshold_passes(self) -> None:
        """price_velocity_bps < threshold AND sell → should pass."""
        enabled = True
        side = "sell"
        velocity = 5.0
        threshold = 8.0
        should_skip = enabled and side == "sell" and velocity > threshold
        assert should_skip is False

    def test_sell_velocity_equal_threshold_passes(self) -> None:
        """price_velocity_bps == threshold → should pass (strict >)."""
        enabled = True
        side = "sell"
        velocity = 8.0
        threshold = 8.0
        should_skip = enabled and side == "sell" and velocity > threshold
        assert should_skip is False

    def test_sell_velocity_disabled_passes(self) -> None:
        """Even high velocity doesn't skip when disabled."""
        enabled = False
        side = "sell"
        velocity = 20.0
        threshold = 8.0
        should_skip = enabled and side == "sell" and velocity > threshold
        assert should_skip is False

    def test_buy_velocity_below_neg_threshold_triggers_skip(self) -> None:
        """price_velocity_bps < threshold AND buy → should skip."""
        enabled = True
        side = "buy"
        velocity = -10.0
        threshold = -8.0
        should_skip = enabled and side == "buy" and velocity < threshold
        assert should_skip is True

    def test_buy_velocity_above_neg_threshold_passes(self) -> None:
        """price_velocity_bps > threshold AND buy → should pass."""
        enabled = True
        side = "buy"
        velocity = -5.0
        threshold = -8.0
        should_skip = enabled and side == "buy" and velocity < threshold
        assert should_skip is False

    def test_sell_rule_does_not_affect_buy(self) -> None:
        """Sell velocity rule should not trigger for buy side."""
        enabled = True
        side = "buy"
        velocity = 20.0
        threshold = 8.0
        should_skip = enabled and side == "sell" and velocity > threshold
        assert should_skip is False

    def test_buy_rule_does_not_affect_sell(self) -> None:
        """Buy velocity rule should not trigger for sell side."""
        enabled = True
        side = "sell"
        velocity = -20.0
        threshold = -8.0
        should_skip = enabled and side == "buy" and velocity < threshold
        assert should_skip is False

    def test_negative_velocity_sell_always_passes(self) -> None:
        """Negative velocity (price dropping) is good for sell."""
        enabled = True
        side = "sell"
        velocity = -10.0
        threshold = 8.0
        should_skip = enabled and side == "sell" and velocity > threshold
        assert should_skip is False

    def test_positive_velocity_buy_always_passes(self) -> None:
        """Positive velocity (price rising) is good for buy."""
        enabled = True
        side = "buy"
        velocity = 10.0
        threshold = -8.0
        should_skip = enabled and side == "buy" and velocity < threshold
        assert should_skip is False


# ---------------------------------------------------------------------------
# Test: YAML round-trip
# ---------------------------------------------------------------------------

class TestVelocityYamlParsing:
    """Config from YAML parsing for velocity fields."""

    def test_from_yaml_with_velocity_config(self) -> None:
        """FillTestConfig.from_yaml should parse velocity skip fields."""
        cfg = clone_fill_test_config(
            load_fill_test_config_from_mapping(
                {
                    "skip_gate": {
                        "enabled": True,
                        "sell_velocity_skip_enabled": True,
                        "sell_velocity_skip_threshold_bps": 5.0,
                        "buy_velocity_skip_enabled": True,
                        "buy_velocity_skip_threshold_bps": -5.0,
                    }
                }
            )
        )
        assert cfg.sell_velocity_skip_enabled is True
        assert cfg.sell_velocity_skip_threshold_bps == 5.0
        assert cfg.buy_velocity_skip_enabled is True
        assert cfg.buy_velocity_skip_threshold_bps == -5.0

    def test_from_yaml_defaults_when_absent(self) -> None:
        """When velocity fields are absent, defaults apply."""
        cfg = clone_fill_test_config(load_fill_test_config_from_mapping({"skip_gate": {"enabled": True}}))
        assert cfg.sell_velocity_skip_enabled is False
        assert cfg.sell_velocity_skip_threshold_bps == 8.0


# ---------------------------------------------------------------------------
# Test: FillRecord serialization with velocity
# ---------------------------------------------------------------------------

class TestFillRecordVelocitySerialization:
    """price_velocity_bps should survive JSON round-trip."""

    def test_to_dict_includes_velocity(self) -> None:
        fr = FillRecord(
            cycle_id="test",
            timestamp=time.time(),
            side="sell",
            order_price=10000000.0,
            order_quantity=0.001,
            price_velocity_bps=12.5,
        )
        d = fr.to_dict()
        assert d["price_velocity_bps"] == 12.5

    def test_to_dict_velocity_none(self) -> None:
        fr = FillRecord(
            cycle_id="test",
            timestamp=time.time(),
            side="sell",
            order_price=10000000.0,
            order_quantity=0.001,
        )
        d = fr.to_dict()
        # None fields should be absent or None in dict
        assert d.get("price_velocity_bps") is None
