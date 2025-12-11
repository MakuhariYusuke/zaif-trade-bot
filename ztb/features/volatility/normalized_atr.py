"""Compatibility shim for volatility normalized ATR feature.

Re-export compute_normalized_atr from the generators subpackage.
"""

from ztb.features.generators.technical.volatility.normalized_atr import (
    compute_normalized_atr,
)

__all__ = ["compute_normalized_atr"]
