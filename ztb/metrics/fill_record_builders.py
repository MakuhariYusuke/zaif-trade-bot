from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from ztb.metrics.fill_record_payloads import (
    build_skip_fill_record_payload,
    sanitize_fill_record_fields,
)


FillRecordT = TypeVar("FillRecordT")


def build_typed_fill_record(
    record_cls: type[FillRecordT],
    *,
    valid_field_names: frozenset[str],
    context: str,
    data: Mapping[str, object],
) -> FillRecordT:
    """Build a typed fill record while filtering unknown payload fields."""

    return record_cls(  # type: ignore[call-arg]
        **sanitize_fill_record_fields(
            data,
            valid_field_names=valid_field_names,
            context=context,
        )
    )


def build_typed_skip_fill_record(
    record_cls: type[FillRecordT],
    *,
    valid_field_names: frozenset[str],
    protected_fields: frozenset[str],
    cycle_id: str,
    timestamp: float,
    side: str,
    order_price: float,
    order_quantity: float,
    cancel_reason: str,
    run_id: str | None,
    git_sha: str | None,
    spread_at_order: float | None = None,
    spread_offset_ratio: float | None = None,
    regime: str | None = None,
    balance_forced_switch: bool = False,
    ab_test_variant: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> FillRecordT:
    """Build a typed skip/audit fill record with protected field handling."""

    payload = build_skip_fill_record_payload(
        cycle_id=cycle_id,
        timestamp=timestamp,
        side=side,
        order_price=order_price,
        order_quantity=order_quantity,
        cancel_reason=cancel_reason,
        run_id=run_id,
        git_sha=git_sha,
        spread_at_order=spread_at_order,
        spread_offset_ratio=spread_offset_ratio,
        regime=regime,
        balance_forced_switch=balance_forced_switch,
        ab_test_variant=ab_test_variant,
        extra=dict(extra or {}),
        valid_field_names=valid_field_names,
        protected_keys=protected_fields,
    )
    return build_typed_fill_record(
        record_cls,
        valid_field_names=valid_field_names,
        context="build_skip_fill_record",
        data=payload,
    )


__all__ = [
    "build_typed_fill_record",
    "build_typed_skip_fill_record",
]
