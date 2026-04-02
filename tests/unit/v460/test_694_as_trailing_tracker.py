from __future__ import annotations

from pathlib import Path

import pytest

import ztb.trading.common.cancel_reasons as CR
from scripts.v460.lib.as_trailing_tracker import ASTrailingConfig, ASTrailingTracker
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460._yaml_test_helpers import load_fill_test_config_from_mapping, load_yaml_mapping


def _enabled_config(**overrides: object) -> ASTrailingConfig:
    base = ASTrailingConfig(
        enabled=True,
        window_size=10,
        spread_bucket_edges=(1500.0, 2500.0, 3500.0),
        soft_threshold=0.30,
        hard_veto_threshold=0.45,
        offset_boost_factor=1.3,
        min_samples=4,
    )
    return ASTrailingConfig(**{**base.__dict__, **overrides})


def _record_many(
    tracker: ASTrailingTracker,
    *,
    regime: str,
    spread: float,
    adverse_count: int,
    total_count: int,
) -> None:
    for index in range(total_count):
        tracker.record_fill(
            regime=regime,
            spread=spread,
            is_adverse=index < adverse_count,
            timestamp=float(index),
        )


class TestASTrailingTracker:
    def test_no_action_when_disabled(self) -> None:
        tracker = ASTrailingTracker(ASTrailingConfig(enabled=False))
        assert tracker.evaluate(regime="ranging", spread=1200.0, side="buy") == (
            "none",
            None,
            None,
        )

    def test_no_action_below_min_samples(self) -> None:
        tracker = ASTrailingTracker(_enabled_config(min_samples=5))
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=4, total_count=4)
        assert tracker.evaluate(regime="ranging", spread=1200.0, side="buy") == (
            "none",
            None,
            None,
        )

    def test_boost_at_soft_threshold(self) -> None:
        tracker = ASTrailingTracker(_enabled_config())
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=3, total_count=10)
        action, offset_mult, as_rate = tracker.evaluate(
            regime="ranging",
            spread=1200.0,
            side="sell",
        )
        assert action == "boost"
        assert offset_mult == pytest.approx(1.3)
        assert as_rate == pytest.approx(0.30)

    def test_veto_at_hard_threshold(self) -> None:
        tracker = ASTrailingTracker(_enabled_config())
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=5, total_count=10)
        action, offset_mult, as_rate = tracker.evaluate(
            regime="ranging",
            spread=1200.0,
            side="sell",
        )
        assert action == "veto"
        assert offset_mult is None
        assert as_rate == pytest.approx(0.50)

    def test_no_action_below_threshold(self) -> None:
        tracker = ASTrailingTracker(_enabled_config())
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=2, total_count=10)
        assert tracker.evaluate(regime="ranging", spread=1200.0, side="buy") == (
            "none",
            None,
            pytest.approx(0.20),
        )

    def test_spread_bucket_separation(self) -> None:
        tracker = ASTrailingTracker(_enabled_config())
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=5, total_count=10)
        _record_many(tracker, regime="ranging", spread=3200.0, adverse_count=1, total_count=10)
        assert tracker.evaluate(regime="ranging", spread=1200.0, side="buy")[0] == "veto"
        assert tracker.evaluate(regime="ranging", spread=3200.0, side="buy")[0] == "none"

    def test_regime_separation(self) -> None:
        tracker = ASTrailingTracker(_enabled_config())
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=5, total_count=10)
        _record_many(tracker, regime="trending_up", spread=1200.0, adverse_count=1, total_count=10)
        assert tracker.evaluate(regime="ranging", spread=1200.0, side="buy")[0] == "veto"
        assert tracker.evaluate(regime="trending_up", spread=1200.0, side="buy")[0] == "none"

    def test_window_eviction(self) -> None:
        tracker = ASTrailingTracker(_enabled_config(window_size=5, min_samples=3))
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=5, total_count=5)
        _record_many(tracker, regime="ranging", spread=1200.0, adverse_count=0, total_count=5)
        action, _offset_mult, as_rate = tracker.evaluate(regime="ranging", spread=1200.0, side="buy")
        assert action == "none"
        assert as_rate == pytest.approx(0.0)

    def test_record_fill_and_get_rate(self) -> None:
        tracker = ASTrailingTracker(_enabled_config(min_samples=2))
        tracker.record_fill(regime="ranging", spread=1000.0, is_adverse=True, timestamp=1.0)
        tracker.record_fill(regime="ranging", spread=1000.0, is_adverse=False, timestamp=2.0)
        as_rate, sample_count = tracker.get_as_rate(regime="ranging", spread=1000.0)
        assert sample_count == 2
        assert as_rate == pytest.approx(0.5)


class TestASTrailingConfigIntegration:
    def test_config_yaml_roundtrip(self) -> None:
        config = load_fill_test_config_from_mapping(
            {
                "skip_gate": {
                    "as_trailing_gate_enabled": True,
                    "as_trailing_gate_window_size": 200,
                    "as_trailing_gate_spread_bucket_edges": "1000,2000",
                    "as_trailing_gate_soft_threshold": 0.25,
                    "as_trailing_gate_hard_veto_threshold": 0.40,
                    "as_trailing_gate_offset_boost_factor": 1.4,
                    "as_trailing_gate_min_samples": 8,
                }
            }
        )
        grouped = config.as_trailing_gate
        assert grouped.enabled is True
        assert grouped.window_size == 200
        assert grouped.spread_bucket_edges == (1000.0, 2000.0)
        assert grouped.soft_threshold == pytest.approx(0.25)
        assert grouped.hard_veto_threshold == pytest.approx(0.40)
        assert grouped.offset_boost_factor == pytest.approx(1.4)
        assert grouped.min_samples == 8

    def test_cancel_reason_in_taxonomy(self) -> None:
        assert CR.AS_TRAILING_GATE_VETO == "as_trailing_gate_veto"

    def test_live_yaml_contains_as_trailing_gate_defaults(self) -> None:
        yaml_cfg = load_yaml_mapping(Path("configs/v460/fill_test.yaml"))
        skip_gate = yaml_cfg["skip_gate"]
        assert skip_gate["as_trailing_gate_enabled"] is False
        assert skip_gate["as_trailing_gate_window_size"] == 100
        assert skip_gate["as_trailing_gate_spread_bucket_edges"] == "1500,2500,3500"

    def test_result_fields_populated_on_early_skip(self, tmp_path: Path) -> None:
        del tmp_path
        config = FillTestConfig(
            as_trailing_gate_enabled=True,
            as_trailing_gate_window_size=10,
            as_trailing_gate_min_samples=4,
            as_trailing_gate_soft_threshold=0.30,
            as_trailing_gate_hard_veto_threshold=0.45,
        )
        evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
        evaluator._config = config
        evaluator._as_trailing_tracker = ASTrailingTracker(config.as_trailing_gate)
        _record_many(
            evaluator._as_trailing_tracker,
            regime="ranging",
            spread=1200.0,
            adverse_count=5,
            total_count=10,
        )

        action, offset_mult, as_rate = evaluator._apply_as_trailing_gate(
            regime="ranging",
            spread=1200.0,
            side="buy",
        )

        assert action == "veto"
        assert offset_mult is None
        assert as_rate == pytest.approx(0.50)
