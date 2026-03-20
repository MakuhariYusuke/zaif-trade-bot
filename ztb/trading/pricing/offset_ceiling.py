"""Pure helpers for final offset ceiling clamping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OffsetCeilingClampResult:
    """Result of applying a final offset ceiling."""

    updated_ratio: float
    clamped: bool


def clamp_offset_ratio_to_ceiling(
    *,
    effective_offset_ratio: float,
    ceiling_ratio: float,
) -> OffsetCeilingClampResult:
    """Clamp an offset ratio to the configured ceiling when enabled."""
    if ceiling_ratio > 0 and effective_offset_ratio > ceiling_ratio:
        return OffsetCeilingClampResult(
            updated_ratio=ceiling_ratio,
            clamped=True,
        )
    return OffsetCeilingClampResult(
        updated_ratio=effective_offset_ratio,
        clamped=False,
    )


__all__ = [
    "OffsetCeilingClampResult",
    "clamp_offset_ratio_to_ceiling",
]
