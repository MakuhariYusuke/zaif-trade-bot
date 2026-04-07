from __future__ import annotations

import logging
from typing import Final, Mapping


logger = logging.getLogger(__name__)

FillRecordPayload = dict[str, object]

_FIELD_ALIASES: Final[dict[str, str]] = {
    "price_velocity_60s": "price_velocity_bps",
}


def sanitize_fill_record_fields(
    values: Mapping[str, object],
    *,
    valid_field_names: frozenset[str],
    context: str,
    protected_keys: frozenset[str] = frozenset(),
) -> FillRecordPayload:
    """FillRecord に存在するキーだけ通し、不要キーは無視する."""
    filtered: FillRecordPayload = {}
    unknown_keys: list[str] = []
    protected_hits: list[str] = []

    for key, value in values.items():
        resolved = _FIELD_ALIASES.get(key, key)
        if resolved != key and resolved in values:
            unknown_keys.append(key)
            continue
        key = resolved

        if key in protected_keys:
            protected_hits.append(key)
            continue
        if key not in valid_field_names:
            unknown_keys.append(key)
            continue
        filtered[key] = value

    if protected_hits:
        logger.debug("%s: protected keys ignored: %s", context, sorted(protected_hits))
    if unknown_keys:
        logger.debug("%s: unknown fields ignored: %s", context, sorted(unknown_keys))
    return filtered


def build_skip_fill_record_payload(
    *,
    cycle_id: str,
    timestamp: float,
    side: str,
    order_price: float,
    order_quantity: float,
    cancel_reason: str,
    run_id: str | None,
    git_sha: str | None,
    spread_at_order: float | None,
    spread_offset_ratio: float | None,
    regime: str | None,
    balance_forced_switch: bool,
    ab_test_variant: str | None,
    extra: Mapping[str, object],
    valid_field_names: frozenset[str],
    protected_keys: frozenset[str],
) -> FillRecordPayload:
    """skip/監査系 FillRecord の payload shaping を集約する."""
    payload: FillRecordPayload = {
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "side": side,
        "order_price": order_price,
        "order_quantity": order_quantity,
        "filled": False,
        "cancelled": True,
        "cancel_reason": cancel_reason,
        "run_id": run_id,
        "git_sha": git_sha,
        "spread_at_order": spread_at_order,
        "spread_offset_ratio": spread_offset_ratio,
        "regime": regime,
        "balance_forced_switch": balance_forced_switch,
        "ab_test_variant": ab_test_variant,
    }
    payload.update(
        sanitize_fill_record_fields(
            extra,
            valid_field_names=valid_field_names,
            context="build_skip_fill_record",
            protected_keys=protected_keys,
        )
    )
    return payload


__all__ = [
    "FillRecordPayload",
    "build_skip_fill_record_payload",
    "sanitize_fill_record_fields",
]
