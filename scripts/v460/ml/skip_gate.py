"""058# Skip Gate — backward-compatibility shim.

372# E11 解消: 実装は ztb/ml/skip_gate.py に移動済み。
このファイルは既存 import パスとの互換性を維持するための re-export shim。
新規コードでは ``from ztb.ml.skip_gate import ...`` を使用すること。
"""
from ztb.ml.skip_gate import (  # noqa: F401
    GATE_FEATURE_COLS,
    SkipDecision,
    SkipGate,
    SkipGateConfig,
    _BASE_FEATURE_COLS,
    _OB_FEATURE_COLS,
    build_features_from_market_state,
    get_gate_feature_cols,
    train_and_save_as_skip_gate,
    train_and_save_skip_gate,
    warm_start_skip_gate_thresholds,
)

__all__ = [
    "GATE_FEATURE_COLS",
    "SkipDecision",
    "SkipGate",
    "SkipGateConfig",
    "_BASE_FEATURE_COLS",
    "_OB_FEATURE_COLS",
    "build_features_from_market_state",
    "get_gate_feature_cols",
    "train_and_save_as_skip_gate",
    "train_and_save_skip_gate",
    "warm_start_skip_gate_thresholds",
]