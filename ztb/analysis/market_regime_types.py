"""Compatibility shim for market_regime_types

Re-export symbols from the internal `regime` package so older import paths
keep working for tests and external scripts.
"""
from .regime.market_regime_types import MarketRegime, RegimeDetectionResult

__all__ = ["MarketRegime", "RegimeDetectionResult"]
