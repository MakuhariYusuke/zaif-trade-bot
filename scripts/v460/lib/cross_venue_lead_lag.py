"""Cross-venue lead-lag helper for safe maker retreat/veto.

433# §3 の BitFlyer 案を 434# §4.2 に沿って安全側へ補正した実装基盤。
初手は directional override ではなく、adverse-side の retreat / veto hint
のみを返す。
445# EMA平滑化 + confidence-weighted scoring 追加。
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


@dataclass
class CrossVenueEMAState:
    """445# EMA state for smoothed cross-venue spread tracking.

    120s cycle interval でのpoint velocity ノイズを平滑化し、
    sign_disagree による hint 脱落を防ぐ。
    506# ema_basis_bps: CC/BF 構造的 basis の rolling EMA 推定。
    """

    ema_ref_mid: float
    ema_spread_bps: float
    last_timestamp: float
    n_updates: int = 0
    ema_basis_bps: float = 0.0  # 506# rolling basis EMA (初期値 0.0 = 保守的)


def update_cross_venue_ema(
    state: CrossVenueEMAState | None,
    ref_mid: float,
    spread_bps: float,
    timestamp: float,
    alpha: float,
    *,
    basis_alpha: float = 0.0,
) -> CrossVenueEMAState:
    """445# Update or initialize EMA state with new observation.

    506# basis_alpha > 0 の場合、spread_bps の rolling EMA を
    ema_basis_bps として追跡 (日次スケールの構造的 basis 推定).
    """
    if state is None or state.n_updates == 0:
        return CrossVenueEMAState(
            ema_ref_mid=ref_mid,
            ema_spread_bps=spread_bps,
            last_timestamp=timestamp,
            n_updates=1,
            ema_basis_bps=spread_bps if basis_alpha > 0 else 0.0,
        )
    _new_basis = (
        basis_alpha * spread_bps + (1.0 - basis_alpha) * state.ema_basis_bps
        if basis_alpha > 0
        else state.ema_basis_bps
    )
    return CrossVenueEMAState(
        ema_ref_mid=alpha * ref_mid + (1.0 - alpha) * state.ema_ref_mid,
        ema_spread_bps=alpha * spread_bps + (1.0 - alpha) * state.ema_spread_bps,
        last_timestamp=timestamp,
        n_updates=state.n_updates + 1,
        ema_basis_bps=_new_basis,
    )


@dataclass(frozen=True)
class CrossVenueLeadLagHint:
    """Lead-lag hint derived from reference venue movement."""

    direction: str
    adverse_side: str
    spread_bps: float                             # EMAモード: ema_spread_bps / Legacy: point spread
    reference_velocity_bps: float
    age_sec: float
    reference_exchange: str
    # 442# L5 拡張
    microprice_spread_bps: float | None = None   # microprice ベースの spread
    depth_imbalance: float | None = None          # (bid_depth - ask_depth) / total
    # 445# confidence scoring
    confidence: float = 1.0                       # 0.0〜1.0, gate/boost の強度制御
    # 448# F3修正: direction/veto/boost に使う spread と診断用の生値を分離
    point_spread_bps: float | None = None         # 瞬間 point spread (EMAモード時のみ記録)


def build_cross_venue_event_details(
    hint: CrossVenueLeadLagHint,
) -> dict[str, object]:
    """Serialize a hint into the event-log payload shape."""
    return {
        "reference_exchange": hint.reference_exchange,
        "direction": hint.direction,
        "adverse_side": hint.adverse_side,
        "spread_bps": hint.spread_bps,
        "point_spread_bps": hint.point_spread_bps,
        "velocity_bps": hint.reference_velocity_bps,
        "age_sec": hint.age_sec,
        "microprice_spread_bps": hint.microprice_spread_bps,
        "depth_imbalance": hint.depth_imbalance,
        "confidence": hint.confidence,
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
            "cross_venue_lead_lag_point_spread_bps": None,
            "cross_venue_lead_lag_velocity_bps": None,
            "cross_venue_lead_lag_age_sec": None,
            "cross_venue_lead_lag_applied": None,
            "cross_venue_lead_lag_vetoed": None,
            "cross_venue_microprice_spread_bps": None,
            "cross_venue_depth_imbalance": None,
            "cross_venue_confidence": None,
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
        "cross_venue_lead_lag_point_spread_bps": (
            hint.point_spread_bps if hint is not None else None
        ),
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
        "cross_venue_confidence": (
            hint.confidence if hint is not None else None
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
    # 445# EMA-based confidence scoring
    ema_spread_bps: float | None = None,
    min_confidence: float = 0.0,
    confidence_reference_spread_bps: float = 3.0,
    # 449# DRY: caller が既に計算済みの point_spread_bps を渡せる
    precomputed_point_spread_bps: float | None = None,
    # 449# confidence floor を外部設定可能に (市場理論根拠: Kyle 1985 λ の最小信頼区間)
    confidence_floor: float = 0.33,
    # 506# basis correction (de-meaning)
    basis_bps: float = 0.0,
) -> CrossVenueLeadLagHint | None:
    """Compute a safe lead-lag hint from local/reference venue mids.

    445# Two operation modes:
      - **Legacy (ema_spread_bps=None)**: Binary thresholds with hard velocity/sign
        gates. Returns hint or None.
      - **Confidence (ema_spread_bps provided)**: Continuous confidence scoring.
        Velocity and microprice become *modifiers* (0.5–1.0) instead of hard gates.
        Returns hint with ``confidence`` field (0.0–1.0), or None if below
        ``min_confidence``.

    Returns ``None`` when:
      - snapshots are invalid or stale
      - EMA spread (or point spread in legacy mode) below threshold
      - confidence below min_confidence (confidence mode only)
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

    # 449# DRY: caller が既に計算済みなら再計算をスキップ
    if precomputed_point_spread_bps is not None:
        spread_bps = precomputed_point_spread_bps
    else:
        spread_bps = (
            (reference_snapshot.mid_price - local_snapshot.mid_price)
            / local_snapshot.mid_price
        ) * 10_000.0
    reference_velocity_bps = (
        (reference_snapshot.mid_price - previous_reference_snapshot.mid_price)
        / previous_reference_snapshot.mid_price
    ) * 10_000.0 / dt_sec

    # 445# Confidence mode vs Legacy mode
    use_confidence = ema_spread_bps is not None

    if use_confidence:
        # --- Confidence mode: EMA-based gating + continuous scoring ---
        gating_spread = ema_spread_bps
        if abs(gating_spread) < spread_bps_threshold:
            return None

        # 506# De-meaning: 構造的 basis を差し引いて direction 判定
        # CC mid が BF mid より常に ~3.3bps 高いため、生 spread だと
        # direction="down" → adverse_side="buy" に偏る。
        # basis_bps を差し引くことで sell 側にも guard が機能する。
        adjusted_spread = gating_spread - basis_bps
        direction = "up" if adjusted_spread > 0.0 else "down"

        # Base confidence from spread magnitude
        # confidence_floor: Kyle (1985) λ の最小信頼区間に相当。
        # spread が小さくても完全に無視しない (information floor)。
        base_conf = min(
            1.0,
            max(confidence_floor, abs(gating_spread) / confidence_reference_spread_bps),
        )

        # Velocity factor (0.5–1.0): agreement boosts, disagreement dampens
        if abs(reference_velocity_bps) < velocity_bps_threshold:
            vel_factor = 0.8   # negligible velocity → neutral
        elif gating_spread * reference_velocity_bps > 0.0:
            vel_factor = 1.0   # agrees → full confidence
        else:
            vel_factor = 0.5   # disagrees → halved (not killed)

        confidence = base_conf * vel_factor
    else:
        # --- Legacy mode: original binary thresholds ---
        if abs(spread_bps) < spread_bps_threshold:
            return None
        if abs(reference_velocity_bps) < velocity_bps_threshold:
            return None
        if spread_bps == 0.0 or reference_velocity_bps == 0.0:
            return None
        if spread_bps * reference_velocity_bps <= 0.0:
            return None

        direction = "up" if spread_bps > 0.0 else "down"
        confidence = 1.0

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

    # 445# Microprice confidence modifier (confidence mode only)
    if use_confidence and microprice_spread_bps is not None:
        direction_sign = 1.0 if direction == "up" else -1.0
        if microprice_spread_bps * direction_sign > 0.0:
            mp_factor = 1.0   # microprice agrees with direction → full
        elif abs(microprice_spread_bps) < 0.5:
            mp_factor = 0.9   # negligible disagreement → nearly full
        else:
            mp_factor = 0.5   # microprice disagrees → halved
        confidence *= mp_factor

    # 442# depth imbalance: (bid - ask) / (bid + ask)
    depth_imbalance: float | None = None
    ref_total = reference_snapshot.bid_depth + reference_snapshot.ask_depth
    if ref_total > 0.0:
        depth_imbalance = (
            (reference_snapshot.bid_depth - reference_snapshot.ask_depth) / ref_total
        )

    # 445# Clamp confidence and apply min_confidence gate
    confidence = max(0.0, min(1.0, confidence))
    if use_confidence and confidence < min_confidence:
        return None

    # 448# F3修正: EMAモードではspread_bpsをema_spread_bpsに統一
    # direction/veto/boost が全て同じ情報源(EMA)を使うようにする
    effective_spread = ema_spread_bps if use_confidence else spread_bps

    return CrossVenueLeadLagHint(
        direction=direction,
        adverse_side=adverse_side,
        spread_bps=effective_spread,
        reference_velocity_bps=reference_velocity_bps,
        age_sec=age_sec,
        reference_exchange=reference_snapshot.exchange,
        microprice_spread_bps=microprice_spread_bps,
        depth_imbalance=depth_imbalance,
        confidence=confidence,
        point_spread_bps=spread_bps if use_confidence else None,
    )
