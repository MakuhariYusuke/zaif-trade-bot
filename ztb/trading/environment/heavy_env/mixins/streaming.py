"""Streaming helpers for HeavyTradingEnv."""

from __future__ import annotations

import gc
import time
from typing import Any

import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def _fetch_streaming_snapshot(self: Any, required_rows: int) -> Any:
    """Fetch an initial snapshot from the streaming pipeline."""
    return self.streaming_handler.fetch_streaming_snapshot(required_rows)

def _prepare_stream_batch(self: Any, batch: pd.DataFrame) -> Any:
    """Normalize streaming data batches to match the environment schema."""
    if batch.empty:
        return batch

    if not getattr(self, "_base_columns", None):
        self._base_columns = list(batch.columns)

    missing = [col for col in self._base_columns if col not in batch.columns]
    for col in missing:
        batch[col] = 0.0

    extra = [col for col in batch.columns if col not in self._base_columns]
    if extra:
        self._base_columns.extend(extra)
        self.df = self.df.reindex(columns=self._base_columns, fill_value=0)

    batch = batch[self._base_columns]
    return self.data_processor.preprocess_data(batch)

def _append_streaming_rows(self: Any) -> bool:
    """Append new rows from the streaming buffer if available."""
    if not self.streaming_handler.streaming_pipeline:
        return False

    buffer_df = self.streaming_handler.streaming_pipeline.buffer.to_dataframe()
    if buffer_df.empty:
        return False

    if self._timestamp_column and "timestamp" in buffer_df.columns:
        buffer_df = buffer_df.sort_values("timestamp").reset_index(drop=True)
        if self._stream_last_timestamp is not None:
            buffer_df = buffer_df[buffer_df["timestamp"] > self._stream_last_timestamp]
    else:
        buffer_df = buffer_df.iloc[self._stream_rows_appended :]

    if buffer_df.empty:
        return False

    if self.streaming_handler.stream_batch_size:
        buffer_df = buffer_df.tail(self.streaming_handler.stream_batch_size)

    prepared = _prepare_stream_batch(self, buffer_df)
    if prepared.empty:
        return False

    self.df = pd.concat([self.df, prepared], ignore_index=True, copy=False)
    # Rolling window: keep at most 50000 rows to prevent OOM
    _max_streaming_rows = getattr(self, '_max_streaming_rows', 50000)
    if len(self.df) > _max_streaming_rows:
        self.df = self.df.iloc[-_max_streaming_rows:].reset_index(drop=True)
    self.n_steps = len(self.df)
    self._stream_rows_appended += len(prepared)

    if self._timestamp_column and "timestamp" in buffer_df.columns:
        self._stream_last_timestamp = pd.to_datetime(buffer_df["timestamp"]).max()

    self._refresh_features()
    self.data_processor.apply_feature_storage_dtype(
        self.df, self.features, self.config.__dict__
    )
    self._build_fast_access_buffers()
    self.memory_manager.log_memory_usage("stream_append", df_override=self.df)

    del prepared
    del buffer_df
    if self.memory_manager.should_collect_garbage:
        gc.collect()

    return True

def _ensure_data_available(self: Any, index: int) -> None:
    """Ensure enough data is available for the requested index."""
    if index < self.n_steps:
        return
    if not self.streaming_handler.streaming_pipeline:
        return
    self.streaming_handler.streaming_pipeline.prefetch_async()
    attempts = 0
    while index >= self.n_steps:
        if _append_streaming_rows(self):
            attempts = 0
            continue
        attempts += 1
        if attempts >= 5:
            break
        time.sleep(0.01)

def _prime_streaming_data(self: Any) -> None:
    """Prime streaming buffers during environment reset."""
    if not self.streaming_handler.streaming_pipeline:
        return
    _append_streaming_rows(self)
    _ensure_data_available(self, self.current_step)
