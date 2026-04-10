"""
Unit tests for:
- ztb.ml.skip_gate_contracts  (FillTestConfig, _SkipGateLike, _SkipDecisionLike)
- scripts.v460.lib.skip_gate_ev_weighted  (SkipGateEvWeightedMixin)
- scripts.v460.lib.skip_gate_model_loader (SkipGateModelLoaderMixin)
- scripts.v460.lib.skip_gate_evaluator    (SkipGateEvaluator)
- scripts.v460.lib.tasks.sac_train        (_as_str_map, build_env_config)
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import pickle

import numpy as np
from numpy import typing as npt
import pytest

from ztb.ml.skip_gate_contracts import (
    FillTestConfig,
    _SkipDecisionLike,
    _SkipGateLike,
)
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from scripts.v460.lib.tasks.sac_train import _as_str_map, build_env_config


# ---------------------------------------------------------------------------
# Module-level stubs (must be at module scope to be picklable)
# ---------------------------------------------------------------------------


class _StubDecision:
    """Picklable _SkipDecisionLike stub."""

    should_skip: bool = False
    confidence: float = 0.9
    expected_value: float = 1.0


class _StubGate:
    """Picklable _SkipGateLike stub."""

    def predict(self, features: npt.NDArray[np.float32]) -> _StubDecision:
        return _StubDecision()

    def predict_proba(self, features: npt.NDArray[np.float32]) -> float:
        return 0.9


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_decision(*, should_skip: bool, confidence: float = 0.8, ev: float = 1.0) -> _SkipDecisionLike:
    """Return a mock _SkipDecisionLike."""
    d = MagicMock()
    d.should_skip = should_skip
    d.confidence = confidence
    d.expected_value = ev
    return d  # type: ignore[return-value]


def _make_gate(*, should_skip: bool = False, ev: float = 1.0) -> _SkipGateLike:
    """Return a mock _SkipGateLike."""
    g = MagicMock()
    g.predict.return_value = _make_decision(should_skip=should_skip, ev=ev)
    g.predict_proba.return_value = 0.9
    return g  # type: ignore[return-value]


def _cfg(**kw: object) -> FillTestConfig:
    """Return a minimal FillTestConfig (with overrides via kw)."""
    defaults: dict[str, object] = {
        "ev_threshold": 0.0,
        "ev_consecutive_skip_limit": 5,
        "reload_interval_seconds": 9999.0,
    }
    defaults.update(kw)
    return FillTestConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FillTestConfig tests
# ---------------------------------------------------------------------------


class TestFillTestConfig:
    def test_defaults(self) -> None:
        cfg = FillTestConfig()
        assert cfg.gate_path is None
        assert cfg.gate_alt_buy_path is None
        assert cfg.gate_alt_sell_path is None
        assert cfg.ev_threshold == 0.0
        assert cfg.ev_consecutive_skip_limit == 5
        assert cfg.reload_interval_seconds == 60.0
        assert cfg.side_model_slots == ("buy", "sell")
        assert cfg.alt_model_slots == ("alt_buy", "alt_sell")

    def test_custom_values(self) -> None:
        root = Path("/tmp/models")
        cfg = FillTestConfig(
            project_root=root,
            ev_threshold=0.5,
            reload_interval_seconds=30.0,
        )
        assert cfg.project_root == root
        assert cfg.ev_threshold == 0.5
        assert cfg.reload_interval_seconds == 30.0


# ---------------------------------------------------------------------------
# SkipGateEvaluator — basic construction
# ---------------------------------------------------------------------------


class TestSkipGateEvaluatorInit:
    def test_init_defaults(self) -> None:
        cfg = FillTestConfig()
        ev = SkipGateEvaluator(cfg)
        assert ev._config is cfg
        assert ev._skip_gate is None
        assert ev._gate_alt_buy is None
        assert ev._gate_alt_sell is None
        assert ev._ev_consecutive_skip_count == 0
        assert ev._model_file_hash == ""
        assert ev._last_reload_check is None

    def test_class_slots(self) -> None:
        assert SkipGateEvaluator._SIDE_MODEL_SLOTS == ("buy", "sell")
        assert SkipGateEvaluator._ALT_MODEL_SLOTS == ("alt_buy", "alt_sell")

    def test_reset(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        ev._ev_consecutive_skip_count = 3
        ev.reset()
        assert ev._ev_consecutive_skip_count == 0


# ---------------------------------------------------------------------------
# SkipGateEvaluator — primary gate
# ---------------------------------------------------------------------------


class TestSkipGateEvaluatorPrimaryGate:
    def _features(self) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)

    def test_no_gate_returns_false(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        assert ev.evaluate(self._features(), side="buy") is False

    def test_primary_gate_skip(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        ev._skip_gate = _make_gate(should_skip=True)
        assert ev.evaluate(self._features(), side="buy") is True

    def test_primary_gate_execute(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        ev._skip_gate = _make_gate(should_skip=False)
        assert ev.evaluate(self._features(), side="buy") is False


# ---------------------------------------------------------------------------
# SkipGateEvWeightedMixin — EV logic
# ---------------------------------------------------------------------------


class TestEvWeightedMixin:
    def _features(self) -> np.ndarray:
        return np.zeros(10, dtype=np.float32)

    def test_ev_above_threshold_no_skip(self) -> None:
        cfg = _cfg(ev_threshold=0.5)
        ev = SkipGateEvaluator(cfg)
        ev._gate_alt_buy = _make_gate(should_skip=False, ev=1.0)
        result = ev._ev_should_skip(self._features(), side="buy")
        assert result is False

    def test_ev_below_threshold_skip(self) -> None:
        cfg = _cfg(ev_threshold=0.5)
        ev = SkipGateEvaluator(cfg)
        ev._gate_alt_buy = _make_gate(should_skip=False, ev=0.1)
        result = ev._ev_should_skip(self._features(), side="buy")
        assert result is True
        assert ev._ev_consecutive_skip_count == 1

    def test_ev_force_execute_after_limit(self) -> None:
        cfg = _cfg(ev_threshold=0.5, ev_consecutive_skip_limit=3)
        ev = SkipGateEvaluator(cfg)
        ev._gate_alt_buy = _make_gate(should_skip=False, ev=0.0)
        # First two calls should skip.
        assert ev._ev_should_skip(self._features(), side="buy") is True
        assert ev._ev_should_skip(self._features(), side="buy") is True
        # Third call hits the limit → force execute.
        assert ev._ev_should_skip(self._features(), side="buy") is False
        assert ev._ev_consecutive_skip_count == 0

    def test_no_alt_gate_no_skip(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        assert ev._ev_should_skip(self._features(), side="sell") is False

    def test_skip_probability_no_gate(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        assert ev._ev_skip_probability(self._features(), side="buy") == 0.0

    def test_skip_probability_with_gate(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        ev._gate_alt_buy = _make_gate()
        ev._gate_alt_buy.predict_proba.return_value = 0.7  # type: ignore[union-attr]
        p = ev._ev_skip_probability(self._features(), side="buy")
        assert abs(p - 0.3) < 1e-9

    def test_select_alt_gate_buy(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        gate = _make_gate()
        ev._gate_alt_buy = gate
        assert ev._select_alt_gate("buy") is gate

    def test_select_alt_gate_sell(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        gate = _make_gate()
        ev._gate_alt_sell = gate
        assert ev._select_alt_gate("sell") is gate

    def test_select_alt_gate_unknown(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        assert ev._select_alt_gate("unknown") is None


# ---------------------------------------------------------------------------
# SkipGateModelLoaderMixin — path resolution
# ---------------------------------------------------------------------------


class TestModelLoaderMixin:
    def test_resolve_absolute_path(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        p = Path("/absolute/model.pkl")
        assert ev._resolve_gate_path(p) == p

    def test_resolve_relative_path(self) -> None:
        root = Path("/project")
        ev = SkipGateEvaluator(FillTestConfig(project_root=root))
        assert ev._resolve_gate_path(Path("models/gate.pkl")) == root / "models/gate.pkl"

    def test_resolve_none(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        assert ev._resolve_gate_path(None) is None

    def test_load_skip_gate_missing_file(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig(gate_path=Path("/nonexistent/gate.pkl")))
        ev.load_skip_gate()
        assert ev._skip_gate is None
        assert ev._model_file_hash == ""

    def test_load_skip_gate_none_path(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        ev.load_skip_gate()
        assert ev._skip_gate is None

    def test_load_skip_gate_binary(self) -> None:
        """Test loading a pickled model (uses a picklable module-level stub)."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            pickle.dump(_StubGate(), fh)
            tmp_path = Path(fh.name)

        try:
            ev = SkipGateEvaluator(FillTestConfig(gate_path=tmp_path))
            ev.load_skip_gate()
            assert ev._skip_gate is not None
            assert ev._model_file_hash != ""
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_maybe_reload_within_interval(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig(reload_interval_seconds=9999.0))
        ev._last_reload_check = time.monotonic()
        reloaded = ev.maybe_reload_skip_gate()
        assert reloaded is False

    def test_get_slot_names(self) -> None:
        ev = SkipGateEvaluator(FillTestConfig())
        slots = ev.get_slot_names()
        assert "buy" in slots
        assert "sell" in slots
        assert "alt_buy" in slots
        assert "alt_sell" in slots


# ---------------------------------------------------------------------------
# sac_train helpers
# ---------------------------------------------------------------------------


class TestSacTrainHelpers:
    def test_as_str_map_with_dict(self) -> None:
        result = _as_str_map({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_as_str_map_with_none(self) -> None:
        assert _as_str_map(None) == {}

    def test_as_str_map_with_int(self) -> None:
        assert _as_str_map(42) == {}

    def test_as_str_map_with_mapping(self) -> None:
        from collections import OrderedDict
        od: OrderedDict[str, int] = OrderedDict([("x", 10)])
        result = _as_str_map(od)
        assert result == {"x": 10}

    def test_build_env_config_empty(self) -> None:
        result = build_env_config({})
        assert result == {}

    def test_build_env_config_merges(self) -> None:
        val_cfg = {
            "environment": {"max_steps": 1000},
            "evaluation": {"eval_freq": 500},
        }
        result = build_env_config(val_cfg)
        assert result["max_steps"] == 1000
        assert result["eval_freq"] == 500

    def test_build_env_config_invalid_values(self) -> None:
        """Non-Mapping values for 'environment' / 'evaluation' are ignored."""
        val_cfg = {"environment": 42, "evaluation": None}
        result = build_env_config(val_cfg)
        assert result == {}
