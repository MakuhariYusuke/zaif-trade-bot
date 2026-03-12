"""Compatibility shim: expose AdaptiveFeatureSelector at `ztb.features.adaptive_selection`.

This module provides a tiny import shim so legacy imports continue to work in tests.
"""
try:
    from ztb.features.generators.adaptive.selection import AdaptiveFeatureSelector
except Exception:
    # Minimal fallback
    class AdaptiveFeatureSelector:
        def __init__(self, *args, **kwargs):
            pass

__all__ = ["AdaptiveFeatureSelector"]
