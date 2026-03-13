"""Pure helpers for SkipGate feature-name and vector handling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

FEATURE_NAME_MIGRATION: dict[str, str] = {
    "price_velocity_60s": "price_velocity_bps",
}


def migrate_skip_gate_feature_cols(feature_cols: Sequence[str]) -> list[str]:
    """Normalize legacy feature names to the current schema."""
    return [FEATURE_NAME_MIGRATION.get(str(col), str(col)) for col in feature_cols]


def build_skip_gate_feature_index(feature_cols: Sequence[str]) -> dict[str, int]:
    """Build a stable feature-name to index map."""
    return {col: idx for idx, col in enumerate(feature_cols)}


def _coerce_finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def build_skip_gate_feature_vector(
    feature_cols: Sequence[str],
    feature_index: Mapping[str, int],
    features: Mapping[str, object],
) -> tuple[np.ndarray, int]:
    """Pack sparse feature dict input into a dense vector."""
    vector = np.full(len(feature_cols), np.nan, dtype=np.float64)
    n_used = 0
    for name, raw_value in features.items():
        idx = feature_index.get(name)
        if idx is None:
            continue
        value = _coerce_finite_float(raw_value)
        if value is None:
            continue
        vector[idx] = value
        n_used += 1
    return vector, n_used
