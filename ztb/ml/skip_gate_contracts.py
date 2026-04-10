"""
Skip-gate type contracts.

Defines Protocols and configuration dataclasses shared between
SkipGateEvaluator and its Mixin components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy import typing as npt


# ---------------------------------------------------------------------------
# Decision / prediction protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class _SkipDecisionLike(Protocol):
    """Protocol for objects that represent a skip/execute decision."""

    @property
    def should_skip(self) -> bool:
        """Return True if the trade should be skipped."""
        ...

    @property
    def confidence(self) -> float:
        """Confidence level of the decision (0.0 – 1.0)."""
        ...

    @property
    def expected_value(self) -> float:
        """Expected value estimate for the potential trade."""
        ...


# ---------------------------------------------------------------------------
# Skip-gate model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class _SkipGateLike(Protocol):
    """Protocol for loaded skip-gate models."""

    def predict(
        self,
        features: npt.NDArray[np.float32],
    ) -> _SkipDecisionLike:
        """Predict whether to skip given feature vector."""
        ...

    def predict_proba(
        self,
        features: npt.NDArray[np.float32],
    ) -> float:
        """Return the probability that the gate would *execute* (not skip)."""
        ...


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FillTestConfig:
    """Configuration for the fill-test (backtesting) skip-gate evaluator."""

    # -- Model paths ----------------------------------------------------------
    project_root: Path = field(default_factory=Path.cwd)
    """Root directory used to resolve relative model paths."""

    gate_path: Path | None = None
    """Path to the primary skip-gate model file (None = gate disabled)."""

    gate_alt_buy_path: Path | None = None
    """Path to the alternative buy-side skip-gate model (None = use primary)."""

    gate_alt_sell_path: Path | None = None
    """Path to the alternative sell-side skip-gate model (None = use primary)."""

    # -- EV / threshold settings ----------------------------------------------
    ev_threshold: float = 0.0
    """Minimum expected-value threshold; trades below this are skipped."""

    ev_consecutive_skip_limit: int = 5
    """Maximum consecutive EV-based skips before forcing an execute."""

    # -- Reload settings ------------------------------------------------------
    reload_interval_seconds: float = 60.0
    """How often (in seconds) to check whether the model file has changed."""

    # -- Slot definitions -----------------------------------------------------
    side_model_slots: tuple[str, ...] = ("buy", "sell")
    """Named slots for side-specific models."""

    alt_model_slots: tuple[str, ...] = ("alt_buy", "alt_sell")
    """Named slots for alternative models."""
