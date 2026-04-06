"""695# Entry-gate adjustment helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.regime_detector import FillTestRegime


@dataclass(frozen=True)
class EntryGateAdjustmentResult:
    base_ev_bps: float
    adjusted_ev_bps: float
    spread_as_guard_triggered: bool
    spread_as_guard_action: str
    spread_as_guard_penalty_bps: float
    regime_guard_ev_premium_bps: float
    regime_guard_penalty_multiplier: float


def _resolve_spread_as_guard_penalty(
    *,
    spread_bps: float,
    enabled: bool,
    spread_threshold_bps: float,
    ev_penalty_bps: float,
    redesign_enabled: bool,
    active_threshold_bps: float,
    inverse_penalty_reference_bps: float,
    inverse_penalty_floor_bps: float,
    inverse_penalty_cap_bps: float,
) -> tuple[bool, str, float]:
    if not enabled and spread_bps >= spread_threshold_bps:
        return False, "none", 0.0
    if enabled and not redesign_enabled and spread_bps >= spread_threshold_bps:
        return False, "none", 0.0
    if enabled and redesign_enabled and spread_bps > active_threshold_bps:
        return False, "none", 0.0

    action = "apply" if enabled else "observe"
    if not redesign_enabled:
        return True, action, float(ev_penalty_bps)

    denominator = max(spread_bps, 0.1)
    inverse_penalty = ev_penalty_bps * (
        inverse_penalty_reference_bps / denominator
    )
    resolved_penalty = min(
        inverse_penalty_cap_bps,
        max(inverse_penalty_floor_bps, inverse_penalty),
    )
    if not math.isfinite(resolved_penalty):
        resolved_penalty = inverse_penalty_cap_bps
    return True, action, float(resolved_penalty)


def apply_entry_gate_adjustments(
    *,
    config: FillTestConfig,
    regime: FillTestRegime | str | None,
    spread_bps: float | None,
    base_ev_bps: float,
) -> EntryGateAdjustmentResult:
    """Compose spread/regime guard adjustments on top of the base EV."""
    normalized_regime = _normalize_regime(regime)
    spread_guard = config.spread_as_guard
    regime_override = config.get_regime_guard_override(
        normalized_regime.value if normalized_regime is not None else None
    )

    spread_triggered = False
    spread_action = "none"
    spread_penalty = 0.0
    if spread_bps is not None:
        spread_triggered, spread_action, spread_penalty = _resolve_spread_as_guard_penalty(
            spread_bps=float(spread_bps),
            enabled=spread_guard.enabled,
            spread_threshold_bps=float(spread_guard.spread_threshold_bps),
            ev_penalty_bps=float(spread_guard.ev_penalty_bps),
            redesign_enabled=spread_guard.redesign_enabled,
            active_threshold_bps=float(spread_guard.active_threshold_bps),
            inverse_penalty_reference_bps=float(spread_guard.inverse_penalty_reference_bps),
            inverse_penalty_floor_bps=float(spread_guard.inverse_penalty_floor_bps),
            inverse_penalty_cap_bps=float(spread_guard.inverse_penalty_cap_bps),
        )

    regime_premium = float(regime_override.ev_threshold_premium_bps)
    regime_multiplier = float(regime_override.spread_as_guard_penalty_multiplier)
    if not regime_override.enabled:
        regime_premium = 0.0
        regime_multiplier = 1.0

    applied_spread_penalty = spread_penalty * regime_multiplier if spread_action == "apply" else 0.0
    adjusted_ev_bps = float(base_ev_bps - applied_spread_penalty - regime_premium)

    return EntryGateAdjustmentResult(
        base_ev_bps=float(base_ev_bps),
        adjusted_ev_bps=adjusted_ev_bps,
        spread_as_guard_triggered=spread_triggered,
        spread_as_guard_action=spread_action,
        spread_as_guard_penalty_bps=spread_penalty,
        regime_guard_ev_premium_bps=regime_premium,
        regime_guard_penalty_multiplier=regime_multiplier,
    )


def _normalize_regime(regime: FillTestRegime | str | None) -> FillTestRegime | None:
    if regime is None:
        return None
    if isinstance(regime, FillTestRegime):
        return regime
    try:
        return FillTestRegime(regime)
    except ValueError:
        return None
