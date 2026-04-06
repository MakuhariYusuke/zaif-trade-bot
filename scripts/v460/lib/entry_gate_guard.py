from __future__ import annotations

from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntryGateGuardConfig:
    """Safety configuration for entry-gate blocking."""

    max_consecutive_blocks: int = 50
    max_block_rate: float = 0.95
    min_eval_count_for_rate: int = 20
    staleness_threshold_sec: float = 600.0
    buy_suppress_ev_threshold: float = -0.5


@dataclass
class EntryGateGuardState:
    """Mutable state for entry-gate safety suppression."""

    consecutive_blocks: int = 0
    total_evals: int = 0
    total_blocks: int = 0
    auto_disabled: bool = False
    auto_disable_reason: str = ""
    last_calibration_update_ts: float = 0.0


class EntryGateGuard:
    """Safety suppression for entry-gate blocking."""

    def __init__(self, config: EntryGateGuardConfig) -> None:
        self._config = config
        self._state = EntryGateGuardState(last_calibration_update_ts=time.time())

    @property
    def state(self) -> EntryGateGuardState:
        return self._state

    def should_suppress_block(self, *, ev: float, regime: str, side: str) -> bool:
        """Return True when the guard should suppress an EV<=0 block."""
        del regime
        if self._is_stale():
            self._auto_disable("stale_calibration_map")
            return True

        # 708# buy-side mild-negative EV suppression must remain reachable even if
        # the global auto-disable counters have tripped. Stale calibration still
        # wins because that is a stronger safety condition than side-aware routing.
        if side == "buy" and ev >= self._config.buy_suppress_ev_threshold:
            return True

        if self._state.auto_disabled:
            return True

        if self._state.consecutive_blocks >= self._config.max_consecutive_blocks:
            self._auto_disable("max_consecutive_blocks")
            return True

        if (
            self._state.total_evals >= self._config.min_eval_count_for_rate
            and self._state.total_blocks / max(self._state.total_evals, 1)
            >= self._config.max_block_rate
        ):
            self._auto_disable("max_block_rate")
            return True

        return False

    def record_eval(self, *, blocked: bool) -> None:
        """Record one enabled entry-gate evaluation."""
        self._state.total_evals += 1
        if blocked:
            self._state.total_blocks += 1
            self._state.consecutive_blocks += 1
        else:
            self._state.consecutive_blocks = 0

    def notify_calibration_update(self) -> None:
        """Refresh the calibration staleness timer."""
        self._state.last_calibration_update_ts = time.time()

    def reset_auto_disable(self) -> None:
        """Reset suppression state after an operator/config toggle."""
        last_update = self._state.last_calibration_update_ts
        self._state = EntryGateGuardState(last_calibration_update_ts=last_update)

    def _is_stale(self) -> bool:
        last_update = self._state.last_calibration_update_ts
        if last_update <= 0.0:
            return True
        return (time.time() - last_update) >= self._config.staleness_threshold_sec

    def _auto_disable(self, reason: str) -> None:
        if self._state.auto_disabled:
            return
        self._state.auto_disabled = True
        self._state.auto_disable_reason = reason
        logger.warning("[690#] entry_gate auto-disabled: %s", reason)
