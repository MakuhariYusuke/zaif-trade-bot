#!/usr/bin/env python3
"""Shared data/env-config helpers for v456 training scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ztb.io.data_loader import DataLoader
from ztb.trading.environment.utils.config import EnvironmentConfig

logger = logging.getLogger(__name__)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp/time column is parsed and set as datetime index when present."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV with strict checks and normalize datetime index."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return ensure_datetime_index(DataLoader.load_csv_strict(csv_path))


def load_and_split_data(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """Load CSV and split into train/val/test with 70/15/15 ratio."""
    logger.info(f"Loading data from {csv_path}")
    df = load_csv_data(csv_path)

    n = len(df)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    logger.info(
        "Loaded %s bars | train=%s val=%s test=%s",
        len(df),
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def env_config_to_dict(env_config: EnvironmentConfig) -> Dict[str, Any]:
    """Convert EnvironmentConfig to dict used by create_fast_intraday_env_v456."""
    config_dict = env_config.as_dict()
    if "initial_balance" not in config_dict and "initial_portfolio_value" in config_dict:
        config_dict["initial_balance"] = config_dict["initial_portfolio_value"]
    return config_dict


def build_env_config_from_training_defaults(
    training_config: Any,
    initial_balance: float | None = None,
) -> Dict[str, Any]:
    """Build env_config dict from ztb.config.environment_config.TrainingConfig-like object."""
    def _get(name: str, default: Any) -> Any:
        return getattr(training_config, name, default)

    return {
        "initial_balance": float(
            _get("INITIAL_BALANCE", 100000.0) if initial_balance is None else initial_balance
        ),
        "max_position_size": float(_get("MAX_POSITION", 1.0)),
        "transaction_cost": float(_get("FEE_RATE", 0.001)),
        "drawdown_limit": float(_get("DRAWDOWN_LIMIT", 0.3)),
        "max_steps": int(_get("MAX_STEPS", 500)),
        "prewarm_steps": int(_get("PREWARM_STEPS", 100)),
        "max_ttl_steps": int(_get("MAX_TTL_STEPS", 60)),
        "cooldown_steps": int(_get("COOLDOWN_STEPS", 5)),
        "max_delta_per_step": float(_get("MAX_DELTA_PER_STEP", 0.2)),
        "min_delta": float(_get("MIN_DELTA", 0.01)),
    }
