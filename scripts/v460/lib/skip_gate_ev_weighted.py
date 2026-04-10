"""
EV-weighted skip-gate evaluation mixin.

Provides logic for computing expected-value (EV) weighted skip decisions.
This mixin is consumed by SkipGateEvaluator; all attributes listed below
are initialised by SkipGateEvaluator.__init__.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ztb.ml.skip_gate_contracts import FillTestConfig, _SkipGateLike

if TYPE_CHECKING:
    pass


class SkipGateEvWeightedMixin:
    """
    Mixin that adds EV-weighted skip-gate decision logic.

    Attribute declarations mirror SkipGateEvaluator.__init__ assignments so
    that mypy can resolve ``self.<attr>`` references without ``type: ignore``.
    """

    # ------------------------------------------------------------------
    # Attribute declarations (set by SkipGateEvaluator.__init__)
    # ------------------------------------------------------------------
    _config: FillTestConfig
    _gate_alt_buy: _SkipGateLike | None
    _gate_alt_sell: _SkipGateLike | None
    _ev_consecutive_skip_count: int

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _ev_should_skip(
        self,
        features: npt.NDArray[np.float32],
        *,
        side: str,
    ) -> bool:
        """
        Return True when the EV-weighted gate recommends skipping.

        Parameters
        ----------
        features:
            1-D float32 feature vector for the current bar.
        side:
            ``"buy"`` or ``"sell"`` – selects the appropriate alt model.

        Returns
        -------
        bool
            True  → skip the trade.
            False → execute the trade.
        """
        gate = self._select_alt_gate(side)
        if gate is None:
            return False

        decision = gate.predict(features)
        ev = decision.expected_value

        if ev < self._config.ev_threshold:
            self._ev_consecutive_skip_count += 1
            if self._ev_consecutive_skip_count >= self._config.ev_consecutive_skip_limit:
                # Force-execute after too many consecutive skips.
                self._ev_consecutive_skip_count = 0
                return False
            return True

        self._ev_consecutive_skip_count = 0
        return False

    def _ev_skip_probability(
        self,
        features: npt.NDArray[np.float32],
        *,
        side: str,
    ) -> float:
        """
        Return the skip probability (0.0 = execute, 1.0 = skip) from the alt gate.

        Falls back to 0.0 when no gate is loaded.
        """
        gate = self._select_alt_gate(side)
        if gate is None:
            return 0.0
        # predict_proba returns execute probability; invert for skip probability.
        return 1.0 - gate.predict_proba(features)

    def _reset_ev_consecutive_skip_count(self) -> None:
        """Reset the consecutive-skip counter (e.g. at episode start)."""
        self._ev_consecutive_skip_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_alt_gate(self, side: str) -> _SkipGateLike | None:
        """Return the alt gate corresponding to *side*."""
        if side == "buy":
            return self._gate_alt_buy
        if side == "sell":
            return self._gate_alt_sell
        return None
