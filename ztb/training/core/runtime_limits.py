from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import pandas as pd


RuntimeConfig: TypeAlias = Mapping[str, object]
DataFrameLoader: TypeAlias = Callable[[str | Path], "pd.DataFrame"]


def _resolve_int_option(
    config: RuntimeConfig,
    key: str,
    *,
    include_ppo_section: bool = False,
) -> int | None:
    value = config.get(key)
    if value is None:
        memory_optimization = config.get("memory_optimization")
        if isinstance(memory_optimization, Mapping):
            value = memory_optimization.get(key)
    if value is None and include_ppo_section:
        ppo_config = config.get("ppo")
        if isinstance(ppo_config, Mapping):
            value = ppo_config.get(key)
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float, str)):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def resolve_data_rows_limit(config: RuntimeConfig) -> int | None:
    """Resolve the effective training-row cap from unified config."""
    return _resolve_int_option(config, "data_rows_limit")


def resolve_max_features(config: RuntimeConfig) -> int | None:
    """Resolve the effective feature cap from unified config."""
    return _resolve_int_option(config, "max_features", include_ppo_section=True)


def load_training_dataframe_with_limit(
    data_path: str | Path,
    *,
    config: RuntimeConfig,
    loader: DataFrameLoader,
    logger: logging.Logger,
) -> "pd.DataFrame":
    """Load training data and apply the shared memory-optimization limit."""
    df_full = loader(data_path)
    data_rows_limit = resolve_data_rows_limit(config)
    if data_rows_limit and len(df_full) > data_rows_limit:
        logger.info(
            "⚠️  MEMORY OPTIMIZATION: Limiting data from %s to %s rows",
            len(df_full),
            data_rows_limit,
        )
        df = df_full.iloc[:data_rows_limit]
        del df_full
        gc.collect()
        return df
    return df_full
