"""Cross-venue lead-lag helper for safe maker retreat/veto.

433# §3 の BitFlyer 案を 434# §4.2 に沿って安全側へ補正した実装基盤。
初手は directional override ではなく、adverse-side の retreat / veto hint
のみを返す。
"""

from __future__ import annotations

from dataclasses import dataclass

from ztb.trading.live.exchanges.base.broker_interfaces import IBroker


@dataclass(frozen=True)
class VenueMidSnapshot:
    """Single venue mid-price snapshot."""

    exchange: str
    mid_price: float
    timestamp: float


@dataclass(frozen=True)
class CrossVenueLeadLagHint:
    """Lead-lag hint derived from reference venue movement."""

    direction: str
    adverse_side: str
    spread_bps: float
    reference_velocity_bps: float
    age_sec: float
    reference_exchange: str


def build_reference_adapter(
    reference_exchange: str,
    *,
    primary_adapter: object,
) -> IBroker | None:
    """Build a secondary public-market adapter using the existing broker registry.

    The secondary adapter inherits the primary adapter's dry-run flag when that
    attribute is available. Unknown exchanges are treated as disabled.
    """
    normalized = reference_exchange.strip().lower()
    if not normalized:
        return None

    from ztb.trading.live.registry.broker_registry import get_broker_registry

    registry = get_broker_registry()
    if not registry.has_broker(normalized):
        return None

    dry_run_value = getattr(primary_adapter, "dry_run", True)
    dry_run = dry_run_value if isinstance(dry_run_value, bool) else True
    return registry.create_adapter(normalized, dry_run=dry_run)


def compute_cross_venue_lead_lag_hint(
    *,
    local_snapshot: VenueMidSnapshot,
    reference_snapshot: VenueMidSnapshot,
    previous_reference_snapshot: VenueMidSnapshot | None,
    max_age_sec: float,
    spread_bps_threshold: float,
    velocity_bps_threshold: float,
) -> CrossVenueLeadLagHint | None:
    """Compute a safe lead-lag hint from local/reference venue mids.

    Returns ``None`` unless:
      - local/reference snapshots are fresh enough
      - reference venue has a valid previous snapshot
      - reference spread and velocity agree in sign
      - both magnitudes exceed the configured thresholds

    ``reference_velocity_bps`` is normalised to a per-second basis.
    """
    if (
        local_snapshot.mid_price <= 0.0
        or reference_snapshot.mid_price <= 0.0
        or previous_reference_snapshot is None
        or previous_reference_snapshot.mid_price <= 0.0
    ):
        return None

    age_sec = abs(reference_snapshot.timestamp - local_snapshot.timestamp)
    if age_sec > max_age_sec:
        return None

    dt_sec = reference_snapshot.timestamp - previous_reference_snapshot.timestamp
    if dt_sec <= 0.0:
        return None

    spread_bps = (
        (reference_snapshot.mid_price - local_snapshot.mid_price)
        / local_snapshot.mid_price
    ) * 10_000.0
    reference_velocity_bps = (
        (reference_snapshot.mid_price - previous_reference_snapshot.mid_price)
        / previous_reference_snapshot.mid_price
    ) * 10_000.0 / dt_sec

    if abs(spread_bps) < spread_bps_threshold:
        return None
    if abs(reference_velocity_bps) < velocity_bps_threshold:
        return None
    if spread_bps == 0.0 or reference_velocity_bps == 0.0:
        return None
    if spread_bps * reference_velocity_bps <= 0.0:
        return None

    direction = "up" if spread_bps > 0.0 else "down"
    adverse_side = "sell" if direction == "up" else "buy"
    return CrossVenueLeadLagHint(
        direction=direction,
        adverse_side=adverse_side,
        spread_bps=spread_bps,
        reference_velocity_bps=reference_velocity_bps,
        age_sec=age_sec,
        reference_exchange=reference_snapshot.exchange,
    )
