"""Streaming buffer utilities for high-frequency data ingestion.

Provides a chunked circular buffer backed by pandas DataFrames to minimise
copy overhead while retaining ergonomic accessors for the training pipeline.
Improved with optional in-memory compression and zero-copy slicing helpers.
"""

from __future__ import annotations

import pickle
import threading
import zlib
import gzip
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from ztb.utils.observability import ObservabilityClient

try:  # Optional acceleration libraries
    import lz4.frame as lz4_frame
    HAS_LZ4 = True
except Exception:  # pragma: no cover - optional dependency
    lz4_frame = None
    HAS_LZ4 = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:  # pragma: no cover - optional dependency
    zstd = None
    HAS_ZSTD = False

DataLike = Union[pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]


class StreamBufferError(Exception):
    """Base exception for stream buffer operations."""


class SchemaMismatchError(StreamBufferError):
    """Raised when appended data does not match the expected schema."""


@dataclass(frozen=True)
class BufferStats:
    """Summary statistics describing the current buffer state."""

    rows: int
    chunks: int
    capacity: int
    memory_bytes: int


class _Chunk:
    """Internal storage wrapper that supports optional compression."""

    __slots__ = ("_df", "_compressed", "compression", "rows")

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        compression: Optional[str] = None,
        compress_min_rows: int = 0,
    ) -> None:
        df = df.reset_index(drop=True)
        self.rows = len(df)
        self.compression = compression if compression else None
        self._compressed: Optional[bytes] = None
        if self.compression and self.rows >= compress_min_rows:
            self._compressed = _compress_df(df, self.compression)
            self._df: Optional[pd.DataFrame] = None
        else:
            self._df = df

    # Data access -----------------------------------------------------
    def materialize(self) -> pd.DataFrame:
        if self._df is None:
            if self._compressed is None:
                raise StreamBufferError("Compressed chunk missing payload")
            self._df = _decompress_df(self._compressed, self.compression)
        return self._df

    def update(self, df: pd.DataFrame, *, compress_min_rows: int) -> None:
        df = df.reset_index(drop=True)
        self.rows = len(df)
        if self.compression and self.rows >= compress_min_rows:
            self._compressed = _compress_df(df, self.compression)
            self._df = None
        else:
            self._compressed = None
            self._df = df

    def memory_bytes(self) -> int:
        if self._df is not None:
            return int(self._df.memory_usage(index=True, deep=True).sum())
        if self._compressed is not None:
            return len(self._compressed)
        return 0

    def clone_dataframe(self) -> pd.DataFrame:
        return self.materialize().copy()


def _compress_df(df: pd.DataFrame, compression: str) -> bytes:
    payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    if compression == "zstd" and HAS_ZSTD and zstd is not None:
        return zstd.ZstdCompressor(level=3).compress(payload)
    if compression == "lz4" and HAS_LZ4 and lz4_frame is not None:
        return lz4_frame.compress(payload)
    if compression == "gzip":
        return gzip.compress(payload, compresslevel=6)
    return zlib.compress(payload, level=6)


def _decompress_df(data: bytes, compression: Optional[str]) -> pd.DataFrame:
    if compression == "zstd" and HAS_ZSTD and zstd is not None:
        payload = zstd.ZstdDecompressor().decompress(data)
    elif compression == "lz4" and HAS_LZ4 and lz4_frame is not None:
        payload = lz4_frame.decompress(data)
    elif compression == "gzip":
        payload = gzip.decompress(data)
    else:
        payload = zlib.decompress(data)
    return pickle.loads(payload)


class StreamBuffer:
    """A memory-efficient circular buffer for streaming market data.

    The buffer stores batches of rows as pandas `DataFrame` chunks inside a
    deque. This avoids the repeated copying cost of concatenating DataFrames
    while still allowing efficient trimming when capacity is exceeded.
    Optional lightweight compression keeps the resident footprint predictable
    during long-running experiments.
    """

    def __init__(
        self,
        capacity: int,
        *,
        columns: Optional[Sequence[str]] = None,
        dtypes: Optional[Mapping[str, Any]] = None,
        on_trim: Optional[Callable[[pd.DataFrame], None]] = None,
        compression: Optional[str] = None,
        compress_min_rows: int = 2000,
        observability: Optional[ObservabilityClient] = None,
        zero_copy: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if compress_min_rows < 0:
            raise ValueError("compress_min_rows must be non-negative")

        if compression and compression not in {"zlib", "zstd", "lz4", "gzip"}:
            raise ValueError(f"compression must be one of {{'zlib','zstd','lz4','gzip'}}")
        if compression == "zstd" and not HAS_ZSTD:
            raise RuntimeError("zstd compression requested but python-zstandard not installed")
        if compression == "lz4" and not HAS_LZ4:
            raise RuntimeError("lz4 compression requested but lz4.frame not installed")

        self.capacity = int(capacity)
        self._columns: Optional[Tuple[str, ...]] = tuple(columns) if columns else None
        self._dtype_map: Dict[str, Any] = dict(dtypes or {})
        self._chunks: Deque[_Chunk] = deque()
        self._rows = 0
        self._lock = threading.RLock()
        self._on_trim = on_trim
        self._compression = compression
        self._compress_min_rows = compress_min_rows
        self.observability = observability
        self.zero_copy = zero_copy
        self._copy_count = 0
        self._view_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def append(self, row: Mapping[str, Any]) -> None:
        """Append a single row to the buffer."""
        self.extend([row])

    def extend(self, data: DataLike) -> int:
        """Append one or more rows to the buffer."""
        df = self._coerce_dataframe(data)
        if df.empty:
            return 0

        with self._lock:
            df = self._prepare_chunk(df)
            chunk = _Chunk(
                df,
                compression=self._compression,
                compress_min_rows=self._compress_min_rows,
            )
            self._chunks.append(chunk)
            added = len(df)
            self._rows += added
            self._trim_to_capacity()
            return added

    def pop_oldest(self, rows: int) -> pd.DataFrame:
        """Remove and return up to `rows` oldest entries."""
        if rows <= 0:
            raise ValueError("rows must be positive")

        with self._lock:
            if self._rows == 0:
                raise StreamBufferError("buffer is empty")

            remaining = rows
            collected: List[pd.DataFrame] = []
            while remaining > 0 and self._chunks:
                chunk = self._chunks[0]
                df = chunk.materialize()
                if len(df) <= remaining:
                    collected.append(df)
                    self._chunks.popleft()
                    self._rows -= len(df)
                    remaining -= len(df)
                    if self._on_trim is not None:
                        self._on_trim(df)
                else:
                    head = df.iloc[:remaining].copy()
                    tail = df.iloc[remaining:]
                    collected.append(head)
                    chunk.update(tail, compress_min_rows=self._compress_min_rows)
                    self._rows -= len(head)
                    if self._on_trim is not None:
                        self._on_trim(head)
                    remaining = 0

            if not collected:
                return pd.DataFrame(columns=self.columns)
            return pd.concat(collected, ignore_index=True, copy=False)

    def to_dataframe(self, last_n: Optional[int] = None) -> pd.DataFrame:
        """Return the most recent `last_n` rows as a single `DataFrame`."""
        with self._lock:
            if self._rows == 0:
                return pd.DataFrame(columns=self.columns)

            if last_n is None or last_n >= self._rows:
                frames = [chunk.materialize() for chunk in self._chunks]
            else:
                frames: List[pd.DataFrame] = []
                rows_needed = last_n
                for chunk in reversed(self._chunks):
                    df = chunk.materialize()
                    if rows_needed <= 0:
                        break
                    if len(df) >= rows_needed:
                        frames.append(df.iloc[-rows_needed:])
                        rows_needed = 0
                    else:
                        frames.append(df)
                        rows_needed -= len(df)
                frames = list(reversed(frames))

            if not frames:
                return pd.DataFrame(columns=self.columns)
            copy_flag = not self.zero_copy
            result = pd.concat(frames, ignore_index=True, copy=copy_flag)
            if self.zero_copy:
                self._view_count += 1
            else:
                self._copy_count += 1
            return result

    def latest(self, rows: int = 1) -> pd.DataFrame:
        """Return the most recent `rows` entries."""
        if rows <= 0:
            raise ValueError("rows must be positive")
        return self.to_dataframe(rows)

    def __len__(self) -> int:
        return self._rows

    @property
    def columns(self) -> Tuple[str, ...]:
        if self._columns is None:
            return tuple()
        return self._columns

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._rows = 0

    def memory_usage(self, deep: bool = True) -> int:
        with self._lock:
            return sum(chunk.memory_bytes() for chunk in self._chunks)

    def stats(self) -> BufferStats:
        with self._lock:
            return BufferStats(
                rows=self._rows,
                chunks=len(self._chunks),
                capacity=self.capacity,
                memory_bytes=self.memory_usage(deep=True),
            )

    def iter_batches(self, batch_size: int, *, include_partial: bool = False) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        with self._lock:
            if self._rows == 0:
                return iter(())  # type: ignore[return-value]
            frames = [chunk.materialize() for chunk in self._chunks]

        total_rows = sum(len(frame) for frame in frames)
        if total_rows == 0:
            return iter(())  # type: ignore[return-value]

        def _generator() -> Iterator[pd.DataFrame]:
            start = 0
            concatenated = pd.concat(frames, ignore_index=True, copy=False)
            while start < total_rows:
                end = min(start + batch_size, total_rows)
                batch = concatenated.iloc[start:end]
                if not include_partial and len(batch) < batch_size:
                    break
                yield batch.reset_index(drop=True)
                start = end

        return _generator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_dataframe(self, data: DataLike) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy(deep=False)
        if isinstance(data, Mapping):
            return pd.DataFrame([data])
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            records = list(data)
            if not records:
                return pd.DataFrame(columns=self.columns)
            return pd.DataFrame(records)
        raise TypeError("data must be a DataFrame or iterable of mappings")

    def _prepare_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=False)
        if self._columns is None:
            self._columns = tuple(df.columns)
        else:
            missing = [col for col in self._columns if col not in df.columns]
            if missing:
                for col in missing:
                    df[col] = np.nan
            extra = [col for col in df.columns if col not in self._columns]
            if extra:
                raise SchemaMismatchError(f"unexpected columns: {extra}")
            df = df.loc[:, list(self._columns)]

        if self._dtype_map:
            cast_map = {col: dtype for col, dtype in self._dtype_map.items() if col in df.columns}
            if cast_map:
                df = df.astype(cast_map, copy=False)

        return df

    def _trim_to_capacity(self) -> None:
        if self._rows <= self.capacity:
            return

        excess = self._rows - self.capacity
        while excess > 0 and self._chunks:
            chunk = self._chunks[0]
            df = chunk.materialize()
            chunk_len = len(df)
            if chunk_len <= excess:
                removed = self._chunks.popleft()
                self._rows -= chunk_len
                excess -= chunk_len
                if self._on_trim is not None:
                    self._on_trim(df)
            else:
                head = df.iloc[:excess].copy()
                tail = df.iloc[excess:]
                chunk.update(tail, compress_min_rows=self._compress_min_rows)
                self._rows -= len(head)
                if self._on_trim is not None:
                    self._on_trim(head)
                excess = 0

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"StreamBuffer(rows={stats.rows}, chunks={stats.chunks}, "
            f"capacity={stats.capacity}, memory_bytes={stats.memory_bytes})"
        )
