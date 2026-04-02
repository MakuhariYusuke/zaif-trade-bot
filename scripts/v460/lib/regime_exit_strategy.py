from __future__ import annotations

import dataclasses
from collections import deque


@dataclasses.dataclass(frozen=True)
class RegimeExitConfig:
    enabled: bool = False
    max_trending_down_buy_fills: int = 10
    tracking_window_sec: float = 3600.0
    escalated_max_factor: float = 0.7
    nfq_trigger_imbalance: float = 0.3


@dataclasses.dataclass(frozen=True)
class RegimeExitResult:
    should_escalate_skewing: bool
    effective_max_factor: float | None
    should_trigger_nfq: bool
    buy_count_in_window: int
    reason: str

    @classmethod
    def noop(cls) -> RegimeExitResult:
        return cls(
            should_escalate_skewing=False,
            effective_max_factor=None,
            should_trigger_nfq=False,
            buy_count_in_window=0,
            reason="disabled",
        )


class RegimeExitTracker:
    """Track trending-down buy exposure inside a rolling time window."""

    def __init__(self, config: RegimeExitConfig) -> None:
        self._config = config
        self._buy_fills: deque[float] = deque()

    def record_fill(self, side: str, timestamp: float) -> None:
        if side == "buy":
            self._buy_fills.append(timestamp)
        self._prune(timestamp)

    def evaluate(
        self,
        *,
        regime: str | None,
        imbalance: float,
        now: float,
    ) -> RegimeExitResult:
        if not self._config.enabled:
            return RegimeExitResult.noop()

        self._prune(now)
        buy_count = len(self._buy_fills)
        if regime != "trending_down":
            return RegimeExitResult(
                should_escalate_skewing=False,
                effective_max_factor=None,
                should_trigger_nfq=False,
                buy_count_in_window=buy_count,
                reason="regime_inactive",
            )

        should_escalate = buy_count > self._config.max_trending_down_buy_fills
        should_trigger_nfq = should_escalate and imbalance >= self._config.nfq_trigger_imbalance
        reason = "watch"
        if should_trigger_nfq:
            reason = "nfq"
        elif should_escalate:
            reason = "escalate"

        return RegimeExitResult(
            should_escalate_skewing=should_escalate,
            effective_max_factor=(
                self._config.escalated_max_factor if should_escalate else None
            ),
            should_trigger_nfq=should_trigger_nfq,
            buy_count_in_window=buy_count,
            reason=reason,
        )

    def _prune(self, now: float) -> None:
        cutoff = now - self._config.tracking_window_sec
        while self._buy_fills and self._buy_fills[0] < cutoff:
            self._buy_fills.popleft()
