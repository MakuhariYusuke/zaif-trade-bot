# Streaming data handling utilities for trading environment
# 取引環境のストリーミングデータ処理ユーティリティ

import time
from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from ztb.data.streaming_pipeline import StreamingPipeline


class StreamingHandler:
    """Handles streaming data ingestion and processing."""

    def __init__(
        self,
        streaming_pipeline: Optional["StreamingPipeline"] = None,
        stream_batch_size: int = 256,
        timestamp_column: Optional[str] = "timestamp",
        episode_id_column: Optional[str] = "episode_id",
    ):
        self.streaming_pipeline = streaming_pipeline
        self.stream_batch_size = max(1, stream_batch_size)
        self._timestamp_column = timestamp_column
        self._episode_id_column = episode_id_column
        self._stream_last_timestamp: Optional[pd.Timestamp] = None
        self._stream_rows_appended = 0
        self._base_columns: List[str] = []

    def fetch_streaming_snapshot(self, required_rows: int) -> pd.DataFrame:
        """ストリーミングパイプラインから初期スナップショットを取得"""
        if not self.streaming_pipeline:
            return pd.DataFrame()

        snapshot = self.streaming_pipeline.buffer.to_dataframe(
            last_n=max(required_rows, self.stream_batch_size)
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
        return batch  # Note: preprocessing should be done by DataProcessor

    def append_streaming_rows(
        self,
        df: pd.DataFrame,
        base_columns: List[str],
        features: List[str],
        config: dict[str, Any],
    ) -> tuple[bool, pd.DataFrame, List[str]]:
        """ストリーミングバッファから新規行を取り込み"""
        if not self.streaming_pipeline:
            return False, df, features

        buffer_df = self.streaming_pipeline.buffer.to_dataframe()
        if buffer_df.empty:
            return False, df, features

        if self._timestamp_column and self._timestamp_column in buffer_df.columns:
            buffer_df = buffer_df.sort_values(self._timestamp_column).reset_index(drop=True)
            if self._stream_last_timestamp is not None:
                buffer_df = buffer_df[
                    buffer_df[self._timestamp_column] > self._stream_last_timestamp
                ]
        else:
            buffer_df = buffer_df.iloc[self._stream_rows_appended :]

        if buffer_df.empty:
            return False, df, features

        if self.stream_batch_size:
            buffer_df = buffer_df.tail(self.stream_batch_size)

        prepared = self.prepare_stream_batch(buffer_df, base_columns, df)
        if prepared.empty:
            return False, df, features

        df = pd.concat([df, prepared], ignore_index=True, copy=False)

        if self._timestamp_column and self._timestamp_column in buffer_df.columns:
            self._stream_last_timestamp = pd.to_datetime(buffer_df[self._timestamp_column]).max()

        # Update features and base columns
        exclude_cols = ["ts", "timestamp", "exchange", "pair", "episode_id"]
        features = [c for c in df.columns if c not in exclude_cols]
        if not features:
            features = list(df.columns)

        if extra := [col for col in df.columns if col not in base_columns]:
            base_columns.extend(extra)

        self._stream_rows_appended += len(prepared)

        return True, df, features

    def ensure_data_available(
        self, index: int, df: pd.DataFrame, n_steps: int
    ) -> tuple[pd.DataFrame, int]:
        """必要なインデックスまでデータを拡張"""
        if index < n_steps:
            return df, n_steps

        if not self.streaming_pipeline:
            return df, n_steps

        self.streaming_pipeline.prefetch_async()
        attempts = 0
        while index >= n_steps:
            success, df, _ = self.append_streaming_rows(df, [], [], {})
            if success:
                n_steps = len(df)
            attempts += 1
            if attempts >= 5:
                break
            time.sleep(0.01)

        return df, n_steps

    def prime_streaming_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """リセット時にストリーミングデータを確保"""
        if not self.streaming_pipeline:
            return df

        _, df, _ = self.append_streaming_rows(df, [], [], {})
        return df