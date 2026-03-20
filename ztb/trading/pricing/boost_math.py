from __future__ import annotations

import math


def decayed_loss_boost_multiplier(
    *,
    base_multiplier: float,
    elapsed_sec: float,
    tau_sec: float,
) -> float:
    """Return the decayed loss-boost multiplier.

    The multiplier decays toward 1.0 as:
        1 + (M - 1) * exp(-t / tau)

    Invalid or non-positive `tau_sec` falls back to the raw multiplier.
    """
    if base_multiplier <= 1.0:
        return 1.0
    if tau_sec <= 0 or elapsed_sec <= 0:
        return base_multiplier
    return 1.0 + (base_multiplier - 1.0) * math.exp(-elapsed_sec / tau_sec)


__all__ = ["decayed_loss_boost_multiplier"]
