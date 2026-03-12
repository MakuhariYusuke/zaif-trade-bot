"""skip_gate_features — backward-compatibility shim.

372# E11 解消: 実装は ztb/ml/skip_gate_features.py に移動済み。
"""
from ztb.ml.skip_gate_features import (  # noqa: F401
    FEATURE_NAME_MIGRATION,
    build_skip_gate_feature_index,
    build_skip_gate_feature_vector,
    migrate_skip_gate_feature_cols,
)

__all__ = [
    "FEATURE_NAME_MIGRATION",
    "build_skip_gate_feature_index",
    "build_skip_gate_feature_vector",
    "migrate_skip_gate_feature_cols",
]