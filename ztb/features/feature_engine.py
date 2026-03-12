"""Compatibility wrapper for feature engine API.

Older modules import compute_features_batch from `ztb.features.feature_engine`.
Delegate to the current implementation in `ztb.features.core.engine`.
"""

from __future__ import annotations

from ztb.features.core.engine import compute_features_batch

__all__ = ["compute_features_batch"]
