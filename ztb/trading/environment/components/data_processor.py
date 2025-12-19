# Data processing utilities for trading environment
# 取引環境のデータ処理ユーティリティ

import gc
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger
from ztb.utils.memory.dtypes import optimize_dtypes

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline

logger = get_logger(__name__)


class DataProcessor:
    """Handles data preprocessing, feature processing, and memory optimization."""

    def __init__(
        self,
        preprocess_chunk_size: int = 32,
        memory_logging_enabled: bool = False,
        gc_step_interval: int = 0,
    ):
        self._preprocess_chunk_size = preprocess_chunk_size
        self._memory_logging_enabled = memory_logging_enabled
        self._gc_step_interval = gc_step_interval

    def _log_memory_usage(
        self, context: str, *, df_override: Optional[pd.DataFrame] = None
    ) -> None:
        """Log memory usage for debugging."""
        if not self._memory_logging_enabled:
            return

        # Memory logging implementation would go here
        # (Simplified for extraction)
        pass

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データの前処理とメモリ最適化"""
        if df.empty:
            return df.copy()  # Empty DataFrame, safe to copy

        # Preserve timestamp metadata for downstream components (e.g. multi-timeframe
        # feature engineering, streaming), even if we drop it from the numeric
        # feature matrix later.
        df_processed = df.copy()
        if "timestamp" not in df_processed.columns:
            if isinstance(df_processed.index, pd.DatetimeIndex):
                df_processed["timestamp"] = df_processed.index
            elif str(getattr(df_processed.index, "name", "")).lower() == "timestamp":
                df_processed["timestamp"] = df_processed.index

        # Reset index only if needed, using inplace operation
        if not df_processed.index.equals(
            pd.RangeIndex(start=0, stop=len(df_processed), step=1)
        ):
            df_processed.reset_index(drop=True, inplace=True)
        # Note: Removed unnecessary copy() when index is already correct

        # Keep `timestamp`/`episode_id` for later phases; exclude other metadata by default
        preserved_cols = [
            "timestamp",
            "episode_id",
        ]
        preserved: dict[str, pd.Series] = {
            col: df_processed[col] for col in preserved_cols if col in df_processed.columns
        }
        if preserved:
            df_processed.drop(columns=list(preserved.keys()), inplace=True, errors="ignore")

        exclude_cols = [
            "ts",
            "exchange",
            "pair",
            "side",
            "source",
        ]
        drop_cols = [c for c in exclude_cols if c in df_processed.columns]
        if drop_cols:
            df_processed.drop(columns=drop_cols, inplace=True, errors="ignore")

        # Fill NaNs only on the numeric/feature part (avoid coercing timestamps)
        df_processed = df_processed.fillna(0)

        optimized, _ = optimize_dtypes(
            df_processed,
            target_float_dtype="float32",
            target_int_dtype="int32",
            convert_objects_to_category=True,
            chunk_size=self._preprocess_chunk_size,
            memory_report=self._memory_logging_enabled,
        )

        numeric_cols = optimized.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                optimized[col] = optimized[col].astype(np.float32, copy=False)

        bool_cols = optimized.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            optimized[bool_cols] = optimized[bool_cols].astype(np.int8, copy=False)

        # Reattach preserved metadata columns (keep original dtypes)
        if preserved:
            preserved_df = pd.DataFrame({col: series.values for col, series in preserved.items()})
            optimized = pd.concat([optimized, preserved_df], axis=1)

        self._log_memory_usage("preprocess", df_override=optimized)

        del df_processed
        if not self._gc_step_interval:
            gc.collect()

        return optimized

    def fetch_streaming_snapshot(
        self,
        streaming_pipeline: Optional["StreamingPipeline"],
        required_rows: int,
        stream_batch_size: int,
    ) -> pd.DataFrame:
        """ストリーミングパイプラインから初期スナップショットを取得"""
        if not streaming_pipeline:
            return pd.DataFrame()

        snapshot = streaming_pipeline.buffer.to_dataframe(
            last_n=max(required_rows, stream_batch_size)
        )
        if snapshot.empty:
            return snapshot
        return snapshot.reset_index(drop=True)

    def prepare_stream_batch(
        self, batch: pd.DataFrame, base_columns: List[str], df: pd.DataFrame
    ) -> pd.DataFrame:
        """環境が扱える形式にストリーミングデータを整形"""
        if batch.empty:
            return batch

        if not base_columns:
            base_columns = list(batch.columns)

        missing = [col for col in base_columns if col not in batch.columns]
        for col in missing:
            batch[col] = 0.0

        extra = [col for col in batch.columns if col not in base_columns]
        if extra:
            base_columns.extend(extra)
            df = df.reindex(columns=base_columns, fill_value=0)

        batch = batch[base_columns]
        return self.preprocess_data(batch)

    def apply_feature_storage_dtype(
        self, df: pd.DataFrame, features: List[str], config: dict[str, Any]
    ) -> None:
        """Ensure feature columns use the configured storage dtype"""
        feature_dtype = str(config.get("feature_storage_dtype", "float16")).lower()
        dtype_map = {"float16": np.float16, "float32": np.float32}
        target_dtype = dtype_map.get(feature_dtype, np.float32)

        protected = {str(col).lower() for col in config.get("precision_columns", [])}
        candidate_features = [
            col
            for col in features
            if col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and col.lower() not in protected
        ]
        if not candidate_features:
            return

        safe_features = []
        if target_dtype is np.float16:
            max_float16 = np.finfo(np.float16).max
            for col in candidate_features:
                if df[col].abs().max() <= max_float16:
                    safe_features.append(col)
        else:
            safe_features = candidate_features

        if not safe_features:
            return

        df[safe_features] = df[safe_features].astype(target_dtype, copy=False)
        if self._memory_logging_enabled:
            self._log_memory_usage("feature_dtype")
