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
    # 442# L5 拡張: microprice + depth imbalance
    microprice: float | None = None     # weighted mid (Gatheral 2018)
    bid_depth: float = 0.0              # 板 bid 側合計出来高
    ask_depth: float = 0.0              # 板 ask 側合計出来高


@dataclass(frozen=True)
class CrossVenueLeadLagHint:
    """Lead-lag hint derived from reference venue movement."""

    direction: str
    adverse_side: str
    spread_bps: float
    reference_velocity_bps: float
    age_sec: float
    reference_exchange: str
    # 442# L5 拡張
    microprice_spread_bps: float | None = None   # microprice ベースの spread
    depth_imbalance: float | None = None          # (bid_depth - ask_depth) / total


def build_cross_venue_event_details(
    hint: CrossVenueLeadLagHint,
) -> dict[str, object]:
    """Serialize a hint into the event-log payload shape."""
    return {
        "reference_exchange": hint.reference_exchange,
        "direction": hint.direction,
        "adverse_side": hint.adverse_side,
        "spread_bps": hint.spread_bps,
        "velocity_bps": hint.reference_velocity_bps,
        "age_sec": hint.age_sec,
        "microprice_spread_bps": hint.microprice_spread_bps,
        "depth_imbalance": hint.depth_imbalance,
    }


def build_cross_venue_fill_fields(
    *,
    enabled: bool,
    hint: CrossVenueLeadLagHint | None,
    side: str,
    vetoed: bool,
) -> dict[str, object]:
    """Serialize a hint into FillRecord-compatible fields."""
    if not enabled and hint is None and not vetoed:
        return {
            "cross_venue_reference_exchange": None,
            "cross_venue_lead_lag_direction": None,
            "cross_venue_lead_lag_adverse_side": None,
            "cross_venue_lead_lag_spread_bps": None,
            "cross_venue_lead_lag_velocity_bps": None,
            "cross_venue_lead_lag_age_sec": None,
            "cross_venue_lead_lag_applied": None,
            "cross_venue_lead_lag_vetoed": None,
            "cross_venue_microprice_spread_bps": None,
            "cross_venue_depth_imbalance": None,
        }

    applied = bool(hint is not None and hint.adverse_side == side)
    return {
        "cross_venue_reference_exchange": (
            hint.reference_exchange if hint is not None else None
        ),
        "cross_venue_lead_lag_direction": hint.direction if hint is not None else None,
        "cross_venue_lead_lag_adverse_side": (
            hint.adverse_side if hint is not None else None
        ),
        "cross_venue_lead_lag_spread_bps": hint.spread_bps if hint is not None else None,
        "cross_venue_lead_lag_velocity_bps": (
            hint.reference_velocity_bps if hint is not None else None
        ),
        "cross_venue_lead_lag_age_sec": hint.age_sec if hint is not None else None,
        "cross_venue_lead_lag_applied": applied,
        "cross_venue_lead_lag_vetoed": vetoed,
        "cross_venue_microprice_spread_bps": (
            hint.microprice_spread_bps if hint is not None else None
        ),
        "cross_venue_depth_imbalance": (
            hint.depth_imbalance if hint is not None else None
        ),
    }


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

    # 442# microprice ベースの spread (存在すれば)
    microprice_spread_bps: float | None = None
    if (
        reference_snapshot.microprice is not None
        and local_snapshot.microprice is not None
        and local_snapshot.microprice > 0.0
    ):
        microprice_spread_bps = (
            (reference_snapshot.microprice - local_snapshot.microprice)
            / local_snapshot.microprice
        ) * 10_000.0
    elif reference_snapshot.microprice is not None and local_snapshot.mid_price > 0.0:
        microprice_spread_bps = (
            (reference_snapshot.microprice - local_snapshot.mid_price)
            / local_snapshot.mid_price
        ) * 10_000.0

    # 442# depth imbalance: (bid - ask) / (bid + ask)
    depth_imbalance: float | None = None
    ref_total = reference_snapshot.bid_depth + reference_snapshot.ask_depth
    if ref_total > 0.0:
        depth_imbalance = (
            (reference_snapshot.bid_depth - reference_snapshot.ask_depth) / ref_total
        )

    return CrossVenueLeadLagHint(
        direction=direction,
        adverse_side=adverse_side,
        spread_bps=spread_bps,
        reference_velocity_bps=reference_velocity_bps,
        age_sec=age_sec,
        reference_exchange=reference_snapshot.exchange,
        microprice_spread_bps=microprice_spread_bps,
        depth_imbalance=depth_imbalance,
    )
