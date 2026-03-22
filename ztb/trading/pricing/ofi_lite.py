"""543# OFI-Lite: Cycle-to-cycle OB depth delta (Cont-Kukanov-Stoikov 2014).

Order Flow Imbalance を 5-level OB volume delta で近似する。
新規 API 呼び出し不要（既存 OB snapshot の差分のみ）。
"""

from __future__ import annotations

from typing import Sequence


def compute_ofi_lite(
    prev_bids: Sequence[tuple[float, float]],
    prev_asks: Sequence[tuple[float, float]],
    curr_bids: Sequence[tuple[float, float]],
    curr_asks: Sequence[tuple[float, float]],
    depth: int = 5,
) -> float:
    """Compute OFI-Lite from two consecutive OB snapshots.

    Returns:
        Normalised OFI ∈ [-1, +1].
        +1 = strong buy pressure (bid volume increasing, ask decreasing).
        -1 = strong sell pressure.
         0 = balanced or no data.
    """
    bid_prev = sum(qty for _, qty in prev_bids[:depth])
    bid_curr = sum(qty for _, qty in curr_bids[:depth])
    ask_prev = sum(qty for _, qty in prev_asks[:depth])
    ask_curr = sum(qty for _, qty in curr_asks[:depth])

    bid_delta = bid_curr - bid_prev
    ask_delta = ask_curr - ask_prev

    denom = abs(bid_delta) + abs(ask_delta)
    if denom < 1e-12:
        return 0.0

    return (bid_delta - ask_delta) / denom
