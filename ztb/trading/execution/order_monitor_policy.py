from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffectiveTimeoutPolicy:
    base_timeout: float
    timeout_multiplier: float
    effective_timeout: float
    regime_reprice_offset: int
    reason: str
    sell_age_cap_applied: bool = False


@dataclass(frozen=True, slots=True)
class StaleRepricePolicy:
    stale_check_sec: float
    stale_drift_bps: float
    stale_max_reprice: int


def compute_effective_timeout_policy(
    *,
    side: str,
    order_timeout_sec: float,
    order_timeout_sec_sell: float | None,
    timeout_override_sec: float | None,
    timeout_reason: str | None,
    regime_name: str | None,
    regime_timeout_multipliers: dict[str, float],
    regime_reprice_adjustments: dict[str, int],
    sell_age_cap_sec: int | None,
) -> EffectiveTimeoutPolicy:
    if timeout_override_sec is not None:
        base_timeout = float(timeout_override_sec)
        timeout_multiplier = 1.0
        reason = timeout_reason or "explicit_override"
    else:
        base_timeout = (
            order_timeout_sec_sell
            if side == "sell" and order_timeout_sec_sell is not None
            else order_timeout_sec
        )
        timeout_multiplier = (
            regime_timeout_multipliers.get(regime_name, 1.0)
            if regime_name is not None
            else 1.0
        )
        if side == "sell" and order_timeout_sec_sell is not None:
            reason = "sell_default"
        else:
            reason = "base_default"
        if regime_name is not None and timeout_multiplier != 1.0:
            reason = f"{reason}_x_regime_{regime_name}"
    effective_timeout = base_timeout * timeout_multiplier
    sell_age_cap_applied = False
    if (
        side == "sell"
        and sell_age_cap_sec is not None
        and sell_age_cap_sec > 0
        and effective_timeout > sell_age_cap_sec
    ):
        effective_timeout = float(sell_age_cap_sec)
        sell_age_cap_applied = True
        reason = f"{reason}_sell_age_cap"
    regime_reprice_offset = (
        regime_reprice_adjustments.get(regime_name, 0)
        if regime_name is not None
        else 0
    )
    return EffectiveTimeoutPolicy(
        base_timeout=float(base_timeout),
        timeout_multiplier=float(timeout_multiplier),
        effective_timeout=float(effective_timeout),
        regime_reprice_offset=int(regime_reprice_offset),
        reason=reason,
        sell_age_cap_applied=sell_age_cap_applied,
    )


def compute_stale_reprice_policy(
    *,
    side: str,
    stale_check_after_sec: float,
    stale_check_after_sec_buy: float | None,
    stale_check_after_sec_sell: float | None,
    stale_drift_bps: float,
    stale_drift_bps_buy: float | None,
    stale_drift_bps_sell: float | None,
    stale_max_reprice: int,
    stale_max_reprice_buy: int | None,
    stale_max_reprice_sell: int | None,
    chase_drift_bps_override: float | None,
    chase_max_reprice_override: int | None,
    regime_reprice_offset: int,
) -> StaleRepricePolicy:
    side_check_sec = (
        stale_check_after_sec_buy if side == "buy" else stale_check_after_sec_sell
    )
    resolved_check_sec = (
        side_check_sec if side_check_sec is not None else stale_check_after_sec
    )

    side_drift = stale_drift_bps_buy if side == "buy" else stale_drift_bps_sell
    resolved_drift = side_drift if side_drift is not None else stale_drift_bps
    if chase_drift_bps_override is not None:
        resolved_drift = chase_drift_bps_override

    side_max_reprice = (
        stale_max_reprice_buy if side == "buy" else stale_max_reprice_sell
    )
    resolved_max_reprice = (
        side_max_reprice if side_max_reprice is not None else stale_max_reprice
    )
    if chase_max_reprice_override is not None:
        resolved_max_reprice = chase_max_reprice_override

    return StaleRepricePolicy(
        stale_check_sec=float(resolved_check_sec),
        stale_drift_bps=float(resolved_drift),
        stale_max_reprice=max(0, int(resolved_max_reprice) + int(regime_reprice_offset)),
    )
