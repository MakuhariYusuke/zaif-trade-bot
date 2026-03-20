"""Shared SAC debug helpers."""

from __future__ import annotations

from typing import Protocol, cast

import pandas as pd

from ztb.utils.env_metrics import extract_env_metrics
from ztb.utils.memory_utils import get_memory_usage


class TrainingDebugEnvProtocol(Protocol):
    observation_space: object
    action_space: object


def _shape_tuple(space: object) -> tuple[int, ...] | None:
    """Gym-like space から shape を安全に取得する."""
    shape = getattr(space, "shape", None)
    if isinstance(shape, tuple):
        return tuple(int(v) for v in shape)
    return None


def build_training_debug_details(
    train_df: object,
    val_df: object,
    *,
    feature_columns_configured: int,
    env: TrainingDebugEnvProtocol | None = None,
) -> dict[str, object]:
    """学習時の DataFrame / env / process 状態を軽量サマリー化する."""
    train_frame = cast(pd.DataFrame, train_df)
    val_frame = cast(pd.DataFrame, val_df)

    details: dict[str, object] = {
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "columns": int(len(train_frame.columns)),
        "feature_columns_configured": int(feature_columns_configured),
        "train_memory_mb": round(float(train_frame.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
        "val_memory_mb": round(float(val_frame.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
        "process_rss_mb": round(float(get_memory_usage().get("rss", 0.0)), 1),
    }

    if "timestamp" in train_frame.columns and len(train_frame) > 0:
        ts_min = train_frame["timestamp"].min()
        ts_max = train_frame["timestamp"].max()
        details["train_timestamp_min"] = ts_min.timestamp() if hasattr(ts_min, "timestamp") else float(ts_min)
        details["train_timestamp_max"] = ts_max.timestamp() if hasattr(ts_max, "timestamp") else float(ts_max)
    if "timestamp" in val_frame.columns and len(val_frame) > 0:
        vs_min = val_frame["timestamp"].min()
        vs_max = val_frame["timestamp"].max()
        details["val_timestamp_min"] = vs_min.timestamp() if hasattr(vs_min, "timestamp") else float(vs_min)
        details["val_timestamp_max"] = vs_max.timestamp() if hasattr(vs_max, "timestamp") else float(vs_max)

    if env is not None:
        obs_shape = _shape_tuple(getattr(env, "observation_space", None))
        action_shape = _shape_tuple(getattr(env, "action_space", None))
        if obs_shape is not None:
            details["observation_shape"] = list(obs_shape)
        if action_shape is not None:
            details["action_shape"] = list(action_shape)
        env_metrics = extract_env_metrics(env, include_optional=False)
        if env_metrics:
            details["env_metrics"] = env_metrics

    return details


__all__ = ["TrainingDebugEnvProtocol", "build_training_debug_details"]
