from __future__ import annotations


def compute_offset_jpy(
    *,
    spread: float,
    effective_offset_ratio: float,
    min_offset_jpy: float,
) -> float:
    """Resolve the concrete JPY offset from spread and ratio."""
    return max(min_offset_jpy, spread * effective_offset_ratio)


__all__ = ["compute_offset_jpy"]
