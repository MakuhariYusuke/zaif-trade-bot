"""
Skip-gate evaluator — v460.

Composes SkipGateEvWeightedMixin and SkipGateModelLoaderMixin into a
single SkipGateEvaluator class that drives the skip-gate logic used
during fill-test backtesting.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ztb.ml.skip_gate_contracts import FillTestConfig, _SkipGateLike

from scripts.v460.lib.skip_gate_ev_weighted import SkipGateEvWeightedMixin
from scripts.v460.lib.skip_gate_model_loader import SkipGateModelLoaderMixin

if TYPE_CHECKING:
    pass


class SkipGateEvaluator(SkipGateEvWeightedMixin, SkipGateModelLoaderMixin):
    """
    Skip-gate evaluator for v460 fill-test backtesting.

    Inherits:
    - SkipGateEvWeightedMixin — EV-weighted skip decision logic
    - SkipGateModelLoaderMixin — lazy model loading / hot-reload logic

    All ``self._*`` attributes referenced by the Mixins are initialised
    here in ``__init__``.
    """

    # ------------------------------------------------------------------
    # Class-level slot constants
    # ------------------------------------------------------------------
    _SIDE_MODEL_SLOTS: tuple[str, ...] = ("buy", "sell")
    _ALT_MODEL_SLOTS: tuple[str, ...] = ("alt_buy", "alt_sell")

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, config: FillTestConfig) -> None:
        """
        Initialise the evaluator.

        Parameters
        ----------
        config:
            Full fill-test configuration.  All paths that are relative will
            be resolved against ``config.project_root``.
        """
        super().__init__()

        # Shared config
        self._config: FillTestConfig = config

        # Project root (used by SkipGateModelLoaderMixin._resolve_gate_path)
        self._project_root: Path = config.project_root

        # Model paths / instances
        self._gate_path: Path | None = config.gate_path
        self._skip_gate: _SkipGateLike | None = None
        self._model_file_hash: str = ""

        # Alt-gate instances (populated on demand)
        self._gate_alt_buy: _SkipGateLike | None = None
        self._gate_alt_sell: _SkipGateLike | None = None

        # EV state
        self._ev_consecutive_skip_count: int = 0

        # Reload bookkeeping
        self._last_reload_check: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        features: npt.NDArray[np.float32],
        *,
        side: str,
    ) -> bool:
        """
        Evaluate whether to skip the next trade.

        Parameters
        ----------
        features:
            1-D float32 feature vector.
        side:
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        bool
            True  → skip.
            False → execute.
        """
        # Hot-reload if the model file changed.
        self.maybe_reload_skip_gate()

        # Primary gate check.
        if self._skip_gate is not None:
            decision = self._skip_gate.predict(features)
            if decision.should_skip:
                return True

        # EV-weighted alt-gate check.
        return self._ev_should_skip(features, side=side)

    def reset(self) -> None:
        """Reset per-episode state (call at the start of each episode)."""
        self._reset_ev_consecutive_skip_count()
