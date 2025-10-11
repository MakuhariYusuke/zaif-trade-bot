"""Pricing helpers for HeavyTradingEnv."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ztb.utils.errors import ValidationError


def _resolve_price(self: Any, step: Optional[int] = None) -> float:
    """Resolve the current price for the given step."""
    step = self.current_step if step is None else max(0, min(step, self.n_steps - 1))
    if step >= self.n_steps:
        raise ValidationError(
            f"Step {step} is out of bounds (max: {self.n_steps - 1})",
            details={"step": step, "max_steps": self.n_steps - 1},
        )

    if self._price_array is not None and getattr(self._price_array, "size", 0) > 0:
        idx = min(step, self._price_array.shape[0] - 1)
        value = float(self._price_array[idx])
        if np.isfinite(value):
            return value

    try:
        row = self.df.iloc[step]
    except (IndexError, KeyError) as exc:
        raise ValidationError(
            f"Could not access data for step {step}",
            details={"step": step, "df_length": len(self.df), "error": str(exc)},
        ) from exc

    for column in ("price", "close", "adj_close", "open"):
        if column in row.index:
            value = row[column]
            if pd.notna(value):
                return float(value)

    numeric_candidates = [v for v in row.values if isinstance(v, (int, float, np.floating))]
    if numeric_candidates:
        return float(numeric_candidates[0])
    return 0.0


def _resolve_atr(self: Any, step: Optional[int] = None, default: float = 1.0) -> float:
    """Resolve ATR value for the given step with sensible fallbacks."""
    step = self.current_step if step is None else max(0, min(step, self.n_steps - 1))
    if self._atr_array is not None and getattr(self._atr_array, "size", 0) > 0:
        idx = min(step, self._atr_array.shape[0] - 1)
        value = float(self._atr_array[idx])
        if np.isfinite(value) and value > 0:
            return value

    if step >= len(self.df):
        return default

    row = self.df.iloc[step]
    for column in ("atr_10", "atr_14", "atr_simplified", "ATR", "ATR_simplified"):
        if column in row.index:
            value = row[column]
            if pd.notna(value) and value > 0:
                return float(value)
    return default
