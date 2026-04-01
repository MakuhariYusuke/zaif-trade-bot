from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PnlAccumulator:
    """Finite PnL 値だけを集計するストリーム集計器."""

    count: int = 0
    total_bps: float = 0.0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        numeric = float(value)
        if not np.isfinite(numeric):
            return
        self.count += 1
        self.total_bps += numeric

    @property
    def mean_bps(self) -> float:
        return self.total_bps / self.count if self.count else 0.0


@dataclass
class PnlWinAccumulator:
    """PnL の平均と勝率を同時に集計するストリーム集計器."""

    pnl: PnlAccumulator = field(default_factory=PnlAccumulator)
    positive_count: int = 0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        numeric = float(value)
        if not np.isfinite(numeric):
            return
        self.pnl.count += 1
        self.pnl.total_bps += numeric
        if numeric > 0:
            self.positive_count += 1

    @property
    def count(self) -> int:
        return self.pnl.count

    @property
    def total_bps(self) -> float:
        return self.pnl.total_bps

    @property
    def mean_bps(self) -> float:
        return self.pnl.mean_bps

    @property
    def win_rate(self) -> float:
        return self.positive_count / self.pnl.count if self.pnl.count else 0.0


__all__ = ["PnlAccumulator", "PnlWinAccumulator"]
