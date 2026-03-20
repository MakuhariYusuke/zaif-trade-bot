from __future__ import annotations

import math
from collections import deque


def update_inventory_counters(
    history: deque[str],
    buy_count: int,
    side: str,
) -> tuple[int, float]:
    """inventory fill deque を更新し、buy_count / imbalance を返す."""
    if history.maxlen is not None and len(history) == history.maxlen:
        evicted = history[0]
        if evicted == "buy":
            buy_count -= 1

    history.append(side)
    if side == "buy":
        buy_count += 1

    n = len(history)
    imbalance = 0.0 if n == 0 else (2 * buy_count - n) / n
    return buy_count, imbalance


def decayed_inventory_imbalance(
    raw_imbalance: float,
    *,
    last_update_time: float,
    tau_sec: float | int | None,
    now: float,
) -> float:
    """time-decay を適用した inventory imbalance を返す."""
    if not isinstance(tau_sec, (int, float)) or tau_sec <= 0 or last_update_time <= 0:
        return raw_imbalance

    elapsed = now - last_update_time
    if elapsed <= 0:
        return raw_imbalance

    return raw_imbalance * math.exp(-elapsed / float(tau_sec))
