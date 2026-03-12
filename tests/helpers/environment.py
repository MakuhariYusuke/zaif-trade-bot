from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BASE_OHLCV_FEATURES = ["open", "high", "low", "close", "volume"]


def make_schema_feature_env_config(
    df: pd.DataFrame,
    /,
    *,
    include_feature_names: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a light-weight environment config that reuses raw OHLCV columns."""
    missing = [column for column in BASE_OHLCV_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    scaler_frame = df[BASE_OHLCV_FEATURES].astype(float)
    scaler_std = scaler_frame.std(axis=0, ddof=0).replace(0.0, 1.0)

    config: dict[str, Any] = {
        "feature_set": "minimal",
        "correlation_reduction": False,
        "scaler_mean": scaler_frame.mean(axis=0).tolist(),
        "scaler_std": scaler_std.tolist(),
    }
    if include_feature_names:
        config["feature_names"] = BASE_OHLCV_FEATURES.copy()
    config.update(overrides)
    return config


def make_stub_multi_timeframe_features(
    df: pd.DataFrame,
    /,
    *,
    columns: int = 3,
) -> pd.DataFrame:
    """Build deterministic multi-timeframe feature rows for merge-contract tests."""
    frame = pd.DataFrame(index=df.index)
    base = np.linspace(0.0, 1.0, len(df), dtype=float)
    for idx in range(columns):
        frame[f"mtf_stub_{idx}"] = base * (idx + 1)
    return frame
