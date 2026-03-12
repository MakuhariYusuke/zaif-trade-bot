"""Compatibility shim for market_regime_classifier

Re-export symbols from the internal `regime` package so older import paths
keep working for tests and external scripts.
"""
from .regime.market_regime_classifier import (
    MarketRegimeClassifier,
    RegimeType,
    RegimeDefinition,
    RegimeDetectionResult,
    RegimeMetrics,
)

__all__ = [
    "MarketRegimeClassifier",
    "RegimeType",
    "RegimeDefinition",
    "RegimeDetectionResult",
    "RegimeMetrics",
]
