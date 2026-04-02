"""Round-trip fill metrics shared helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord


@dataclass
class RoundTripRecord:
    """往復取引記録 (buy→sell / sell→buy 双方向対応)."""

    entry_record: FillRecord
    exit_record: FillRecord
    pnl_bps: float
    pnl_jpy: float
    hold_sec: float
    direction: str

    @property
    def buy_record(self) -> FillRecord:
        return self.entry_record if self.direction == "buy_first" else self.exit_record

    @property
    def sell_record(self) -> FillRecord:
        return self.exit_record if self.direction == "buy_first" else self.entry_record


@dataclass
class RoundTripMetrics:
    """Round-trip 集計指標."""

    total_pairs: int = 0
    pnl_mean_bps: float = 0.0
    pnl_median_bps: float = 0.0
    pnl_std_bps: float = 0.0
    pnl_total_jpy: float = 0.0
    win_rate: float = 0.0
    hold_sec_median: float = 0.0
    unpaired_buys: int = 0
    unpaired_sells: int = 0
    net_inventory: int = 0


def compute_round_trip_metrics(
    records: list[FillRecord],
) -> tuple[RoundTripMetrics, list[RoundTripRecord]]:
    """双方向 FIFO ペアリングで往復損益を算出."""
    filled = [record for record in records if record.filled and record.fill_price is not None]
    filled.sort(key=lambda record: record.timestamp)

    pending_buys: deque[FillRecord] = deque()
    pending_sells: deque[FillRecord] = deque()
    trips: list[RoundTripRecord] = []

    for record in filled:
        if record.side == "buy":
            if pending_sells:
                sell_entry = pending_sells.popleft()
                sell_price = sell_entry.fill_price
                buy_price = record.fill_price
                if sell_price is None or buy_price is None:
                    continue
                qty = min(record.order_quantity, sell_entry.order_quantity)
                trips.append(
                    RoundTripRecord(
                        entry_record=sell_entry,
                        exit_record=record,
                        pnl_bps=(sell_price - buy_price) / buy_price * 10_000,
                        pnl_jpy=(sell_price - buy_price) * qty,
                        hold_sec=record.timestamp - sell_entry.timestamp,
                        direction="sell_first",
                    )
                )
            else:
                pending_buys.append(record)
        elif record.side == "sell":
            if pending_buys:
                buy_entry = pending_buys.popleft()
                sell_price = record.fill_price
                buy_price = buy_entry.fill_price
                if sell_price is None or buy_price is None:
                    continue
                qty = min(record.order_quantity, buy_entry.order_quantity)
                trips.append(
                    RoundTripRecord(
                        entry_record=buy_entry,
                        exit_record=record,
                        pnl_bps=(sell_price - buy_price) / buy_price * 10_000,
                        pnl_jpy=(sell_price - buy_price) * qty,
                        hold_sec=record.timestamp - buy_entry.timestamp,
                        direction="buy_first",
                    )
                )
            else:
                pending_sells.append(record)

    if not trips:
        return (
            RoundTripMetrics(
                unpaired_buys=len(pending_buys),
                unpaired_sells=len(pending_sells),
                net_inventory=len(pending_buys) - len(pending_sells),
            ),
            [],
        )

    pnl_arr = [trip.pnl_bps for trip in trips]
    hold_arr = [trip.hold_sec for trip in trips]
    return (
        RoundTripMetrics(
            total_pairs=len(trips),
            pnl_mean_bps=float(np.mean(pnl_arr)),
            pnl_median_bps=float(np.median(pnl_arr)),
            pnl_std_bps=float(np.std(pnl_arr)),
            pnl_total_jpy=sum(trip.pnl_jpy for trip in trips),
            win_rate=sum(1 for pnl in pnl_arr if pnl > 0) / len(pnl_arr),
            hold_sec_median=float(np.median(hold_arr)),
            unpaired_buys=len(pending_buys),
            unpaired_sells=len(pending_sells),
            net_inventory=len(pending_buys) - len(pending_sells),
        ),
        trips,
    )


__all__ = [
    "RoundTripRecord",
    "RoundTripMetrics",
    "compute_round_trip_metrics",
]
