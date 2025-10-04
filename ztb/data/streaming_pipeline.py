"""Streaming data pipeline integrating CoinGecko ingestion with feature generation."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, List, Optional, Sequence

import pandas as pd

from ztb.utils.errors import NetworkError, safe_operation

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]

from ztb.features import FeatureRegistry
from ztb.features.feature_engine import compute_features_batch
from ztb.utils.observability import ObservabilityClient

from .coin_gecko_stream import CoinGeckoStream, MarketDataBatch, StreamConfig
from .stream_buffer import BufferStats, StreamBuffer

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _default_validator(df: pd.DataFrame) -> pd.DataFrame:
    def perform_validation() -> pd.DataFrame:
        if df.empty:
            return df

        result = df.copy()
        if "timestamp" not in result.columns:
            raise ValueError("streamed dataframe must include 'timestamp' column")

        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        numeric_cols = [
            col for col in ["price", "market_cap", "volume"] if col in result.columns
        ]
        for col in numeric_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        result = result.dropna(subset=["timestamp", "price"])
        result = result.drop_duplicates(subset="timestamp").sort_values("timestamp")
        return result.reset_index(drop=True)

    return safe_operation(logger, perform_validation, "_default_validator()", df)


@dataclass
class StreamingHealth:
    """Represents the current health of the streaming subsystem."""

    status: str = "ok"
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    def update_system_stats(self) -> None:
        """Update memory and CPU usage statistics."""
        if HAS_PSUTIL and psutil is not None:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.cpu_usage_percent = process.cpu_percent(interval=0.1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "last_error_at": (
                self.last_error_at.isoformat() if self.last_error_at else None
            ),
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
        }


@dataclass
class PipelineStats:
    buffer: BufferStats
    last_batch_at: Optional[datetime] = None
    last_batch_rows: int = 0
    total_rows_streamed: int = 0
    health: StreamingHealth = field(default_factory=StreamingHealth)


class StreamingPipeline:
    """Coordinates streaming ingestion, buffering, and feature computation."""

    _shutdown: bool = False
    _background_stop: threading.Event
    _background_thread: Optional[threading.Thread] = None

    def __init__(
        self,
        stream_client: CoinGeckoStream,
        *,
        buffer_capacity: int = 1_000_000,
        feature_names: Optional[Sequence[str]] = None,
        lookback_rows: int = 512,
        stream_config: Optional[StreamConfig] = None,
        validator: Callable[[pd.DataFrame], pd.DataFrame] = _default_validator,
        metrics_callback: Optional[Callable[[PipelineStats], None]] = None,
        async_features: bool = True,
        feature_workers: int = 1,
        buffer_compression: Optional[
            str
        ] = "gzip",  # Compression algorithm: 'gzip', 'zlib', 'lz4', 'zstd', or None
        buffer_compress_min_rows: int = 2000,
        max_backoff_seconds: float = 60.0,
        observability: Optional[ObservabilityClient] = None,
    ) -> None:
        super().__init__()
        super().__init__()

        self._shutdown = False
        self._background_stop = threading.Event()
        self._background_thread = None

        if buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be positive")
        if lookback_rows < 0:
            raise ValueError("lookback_rows must be non-negative")
        if feature_workers <= 0:
            raise ValueError("feature_workers must be positive")

        FeatureRegistry.initialize()

        self.stream_client = stream_client
        self.stream_config = stream_config or StreamConfig()
        self.validator = validator
        self.metrics_callback = metrics_callback
        self.lookback_rows = lookback_rows
        self._lock = threading.RLock()
        self._health = StreamingHealth()
        self._backoff_seconds = 1.0
        self._max_backoff_seconds = max_backoff_seconds
        self.observability = observability
        self.observability = observability

        self.buffer = StreamBuffer(
            capacity=buffer_capacity,
            columns=None,
            compression=buffer_compression,
            compress_min_rows=buffer_compress_min_rows,
        )

        # Store buffer configuration for resizing
        self._buffer_compression = buffer_compression
        self._buffer_compress_min_rows = buffer_compress_min_rows

        registry_features = FeatureRegistry.list()
        self.feature_names: List[str] = (
            list(feature_names) if feature_names else registry_features
        )
        self._stats = PipelineStats(buffer=self.buffer.stats(), health=self._health)

        self._async_features = async_features
        self._feature_workers = feature_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_future: Optional[Future[Any]] = None
        self._connection_monitor_thread: Optional[threading.Thread] = None
        self._connection_monitor_stop = threading.Event()
        self._last_connection_check = datetime.now(timezone.utc)
        self._connection_check_interval = 30.0  # seconds
        self._max_reconnect_attempts = 10
        self._reconnect_attempts = 0
        self._adaptive_buffer = True
        self._buffer_target_utilization = 0.7  # 70% utilization target

        if self._async_features:
            self._executor = ThreadPoolExecutor(
                max_workers=self._feature_workers,
                thread_name_prefix="stream_features",
            )

    # ------------------------------------------------------------------
    def prime(
        self,
        start: datetime,
        end: datetime,
        *,
        granularity: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Warm up the pipeline with historical data."""
        cfg = self.stream_config
        batch = self.stream_client.fetch_range(
            cfg.coin_id,
            cfg.vs_currency,
            start,
            end,
            granularity=granularity or cfg.granularity,
            page_size=page_size or cfg.page_size,
        )
        return self._ingest_batch(batch)

    def stream(
        self,
        *,
        start_at: Optional[datetime] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[pd.DataFrame]:
        cfg = self.stream_config
        while True:
            try:
                for batch in self.stream_client.stream(
                    cfg, start_at=start_at, stop_event=stop_event
                ):
                    try:
                        features = self._ingest_batch(batch)
                        self._record_success()
                    except Exception as exc:  # pragma: no cover - defensive logging
                        self._record_failure(exc)
                        logger.exception("Failed to ingest streaming batch")
                        if stop_event and stop_event.is_set():
                            return
                        continue

                    if not features.empty:
                        yield features
            except Exception as exc:  # pragma: no cover - defensive logging
                self._record_failure(exc)
                logger.exception("Streaming client raised error; backing off")

                # Graceful degradation: increase poll interval on repeated failures
                degradation_factor = min(self._health.consecutive_failures, 10)
                extended_backoff = (
                    min(self._backoff_seconds, self._max_backoff_seconds)
                    * degradation_factor
                )

                if stop_event and stop_event.is_set():
                    break
                logger.info(
                    "Backing off for %.1f seconds (degradation factor: %d)",
                    extended_backoff,
                    degradation_factor,
                )
                time.sleep(extended_backoff)
                self._backoff_seconds = min(
                    self._backoff_seconds * 2, self._max_backoff_seconds
                )
                continue
            else:
                break

    def latest_features(self, rows: int) -> pd.DataFrame:
        if rows <= 0:
            raise ValueError("rows must be positive")
        return self._compute_features_for_latest(rows)

    def get_data(self, **kwargs: Any) -> pd.DataFrame:
        """Get market data from the streaming pipeline."""
        rows = kwargs.get("rows", 100)
        return self.latest_features(rows)

    def stats(self) -> PipelineStats:
        with self._lock:
            stats_copy = PipelineStats(
                buffer=self.buffer.stats(),
                last_batch_at=self._stats.last_batch_at,
                last_batch_rows=self._stats.last_batch_rows,
                total_rows_streamed=self._stats.total_rows_streamed,
                health=self._health,
            )
        return stats_copy

    def describe(self) -> str:
        cfg = self.stream_config
        return (
            f"StreamingPipeline(coin={cfg.coin_id}, vs={cfg.vs_currency}, "
            f"granularity={cfg.granularity}, features={len(self.feature_names)})"
        )

    def health(self) -> StreamingHealth:
        return self._health

    def close(self) -> None:
        if self._background_thread and self._background_thread.is_alive():
            pass
        self.stop_connection_monitor()
        if self._executor and not self._shutdown:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._shutdown = True
        if self.observability:
            self.observability.log_event("pipeline_closed")
            self.observability.close()

    def __del__(self) -> None:  # pragma: no cover - destructor defensive
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _ingest_batch(self, batch: MarketDataBatch) -> pd.DataFrame:
        cleaned = self.validator(batch.dataframe)
        if cleaned.empty:
            logger.debug("Received empty batch from CoinGecko; skipping")
            return pd.DataFrame()

        with self._lock:
            rows_added = self.buffer.extend(cleaned)
            self._stats = PipelineStats(
                buffer=self.buffer.stats(),
                last_batch_at=batch.fetched_at,
                last_batch_rows=rows_added,
                total_rows_streamed=self._stats.total_rows_streamed + rows_added,
                health=self._health,
            )

        if rows_added == 0:
            return pd.DataFrame()

        features = self._compute_features_for_latest(rows_added)

        if self.metrics_callback:
            try:
                self.metrics_callback(self._stats)
            except Exception:  # pragma: no cover - defensive logging only
                logger.exception("metrics_callback raised an exception")

        if self.observability:
            self.observability.log_event(
                "stream_batch",
                {
                    "rows_added": rows_added,
                    "total_rows_streamed": self._stats.total_rows_streamed,
                    "buffer_rows": self._stats.buffer.rows,
                    "health": self._health.status,
                },
            )
            self.observability.record_metrics(
                {
                    "stream_rows_total": float(self._stats.total_rows_streamed),
                    "stream_buffer_rows": float(self._stats.buffer.rows),
                }
            )

        return features

    def _compute_features_for_latest(self, rows: int) -> pd.DataFrame:
        window = rows + self.lookback_rows
        history = self.buffer.to_dataframe(last_n=window)
        if history.empty:
            return pd.DataFrame()

        if not self.feature_names:
            return history.tail(rows).reset_index(drop=True)

        if self._async_features and self._executor is not None:
            future: Future[pd.DataFrame] = self._executor.submit(
                self._compute_features_core,
                history,
                rows,
            )
            try:
                return future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                self._record_failure(exc)
                raise
        return self._compute_features_core(history, rows)

    def _compute_features_core(self, history: pd.DataFrame, rows: int) -> pd.DataFrame:
        features_df = compute_features_batch(
            history.copy(),
            feature_names=list(self.feature_names),
            report_interval=(rows + self.lookback_rows) + 1,
            verbose=False,
        )

        combined = pd.concat([history.reset_index(drop=True), features_df], axis=1)
        if rows >= len(combined):
            return combined
        return combined.iloc[-rows:].reset_index(drop=True)

    def prefetch_async(self) -> None:
        if not self._async_features or not self._executor or self._shutdown:
            return
        with self._lock:
            future = self._prefetch_future
            if future and not future.done():
                return
            self._prefetch_future = self._executor.submit(lambda: list(self.stream()))

    def start_background_stream(
        self, stop_event: Optional[threading.Event] = None
    ) -> threading.Event:
        if self._shutdown:
            raise RuntimeError("pipeline already shut down")
        if self._background_thread and self._background_thread.is_alive():
            return self._background_stop
        self._background_stop = stop_event or threading.Event()

        def _worker() -> None:
            try:
                for _ in self.stream(stop_event=self._background_stop):
                    if self._background_stop.is_set():
                        break
            except Exception:
                logger.exception("Background streaming thread terminated unexpectedly")

        self._background_thread = threading.Thread(
            target=_worker, name="streaming-background", daemon=True
        )
        self._background_thread.start()

        # Start connection monitoring
        self.start_connection_monitor()

        return self._background_stop

    def start_connection_monitor(self) -> None:
        """Start the connection monitoring thread."""
        if self._shutdown:
            return
        if (
            self._connection_monitor_thread
            and self._connection_monitor_thread.is_alive()
        ):
            return

        self._connection_monitor_stop.clear()

        def _monitor_worker() -> None:
            while not self._connection_monitor_stop.is_set() and not self._shutdown:
                try:
                    self._check_connection_health()
                    self._adjust_buffer_size()
                except Exception as exc:
                    logger.exception("Connection monitor error: %s", exc)

                self._connection_monitor_stop.wait(self._connection_check_interval)

        self._connection_monitor_thread = threading.Thread(
            target=_monitor_worker, name="connection-monitor", daemon=True
        )
        self._connection_monitor_thread.start()
        logger.info("Connection monitor started")

    def stop_connection_monitor(self) -> None:
        """Stop the connection monitoring thread."""
        self._connection_monitor_stop.set()
        if (
            self._connection_monitor_thread
            and self._connection_monitor_thread.is_alive()
        ):
            self._connection_monitor_thread.join(timeout=2.0)
        self._connection_monitor_thread = None
        logger.info("Connection monitor stopped")

    def _check_connection_health(self) -> None:
        """Check the health of the streaming connection."""
        now = datetime.now(timezone.utc)
        time_since_last_check = (now - self._last_connection_check).total_seconds()

        if time_since_last_check < self._connection_check_interval:
            return

        self._last_connection_check = now

        try:
            # Perform a lightweight health check by fetching recent data
            recent_batch = self.stream_client.fetch_range(
                self.stream_config.coin_id,
                self.stream_config.vs_currency,
                now - timedelta(minutes=5),
                now,
                granularity="1m",
                page_size=1,
            )
            is_healthy = recent_batch is not None and not recent_batch.dataframe.empty

            if is_healthy:
                if self._health.status != "ok":
                    logger.info("Connection health restored")
                    self._record_success()
                self._reconnect_attempts = 0
            else:
                logger.warning("Connection health check failed")
                self._handle_connection_failure("Health check failed")

        except Exception as exc:
            logger.warning("Connection health check error: %s", exc)
            self._handle_connection_failure(str(exc))

    def _handle_connection_failure(self, reason: str) -> None:
        """Handle connection failure with reconnection logic."""
        self._reconnect_attempts += 1

        if self._reconnect_attempts <= self._max_reconnect_attempts:
            backoff_time = min(
                self._backoff_seconds * (2 ** (self._reconnect_attempts - 1)),
                self._max_backoff_seconds,
            )
            logger.info(
                "Attempting reconnection %d/%d in %.1f seconds",
                self._reconnect_attempts,
                self._max_reconnect_attempts,
                backoff_time,
            )

            # Update health status
            self._health.status = "reconnecting"
            self._health.last_error = f"Connection failed: {reason}"

            # Schedule reconnection
            threading.Timer(backoff_time, self._attempt_reconnect).start()
        else:
            logger.error("Max reconnection attempts exceeded, entering degraded mode")
            self._health.status = "degraded"
            self._health.last_error = f"Max reconnection attempts exceeded: {reason}"

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the streaming service."""
        try:
            logger.info("Attempting to reconnect...")
            # Test the connection by fetching recent data
            now = datetime.now(timezone.utc)
            test_batch = self.stream_client.fetch_range(
                self.stream_config.coin_id,
                self.stream_config.vs_currency,
                now - timedelta(minutes=5),
                now,
                granularity="1m",
                page_size=1,
            )

            if test_batch and not test_batch.dataframe.empty:
                logger.info("Reconnection successful")
                self._record_success()
                self._reconnect_attempts = 0
                # Restart streaming if it was running
                if self._background_thread and not self._background_thread.is_alive():
                    self.start_background_stream()
            else:
                raise NetworkError("Reconnection test failed")

        except Exception as exc:
            logger.warning("Reconnection attempt failed: %s", exc)
            self._handle_connection_failure(str(exc))

    def _adjust_buffer_size(self) -> None:
        """Dynamically adjust buffer size based on memory usage and performance."""
        if not self._adaptive_buffer:
            return

        try:
            buffer_stats = self.buffer.stats()
            current_utilization = buffer_stats.rows / buffer_stats.capacity

            # Update system stats
            self._health.update_system_stats()

            memory_mb = self._health.memory_usage_mb

            # Adjust buffer size based on utilization and system resources
            target_capacity = buffer_stats.capacity

            if current_utilization > 0.9:  # Over 90% utilization
                # Increase buffer size if memory allows
                if memory_mb < 500:  # Less than 500MB memory usage
                    target_capacity = int(buffer_stats.capacity * 1.5)
                    logger.info(
                        "Increasing buffer capacity to %d due to high utilization",
                        target_capacity,
                    )
            elif (
                current_utilization < 0.3 and buffer_stats.capacity > 100000
            ):  # Under 30% utilization
                # Decrease buffer size to save memory
                target_capacity = max(100000, int(buffer_stats.capacity * 0.8))
                logger.info(
                    "Decreasing buffer capacity to %d due to low utilization",
                    target_capacity,
                )

            # Apply high memory pressure adjustments
            if memory_mb > 800:  # Over 800MB
                target_capacity = max(50000, int(buffer_stats.capacity * 0.5))
                logger.warning(
                    "Reducing buffer capacity to %d due to high memory usage (%.1f MB)",
                    target_capacity,
                    memory_mb,
                )

            # Resize buffer if needed
            if target_capacity != buffer_stats.capacity:
                logger.info(
                    "Resizing buffer from %d to %d rows",
                    buffer_stats.capacity,
                    target_capacity,
                )
                # Create new buffer with target capacity
                old_buffer = self.buffer
                self.buffer = StreamBuffer(
                    capacity=target_capacity,
                    columns=None,
                    compression=self._buffer_compression,
                    compress_min_rows=self._buffer_compress_min_rows,
                )
                # Copy existing data to new buffer
                if buffer_stats.rows > 0:
                    existing_data = old_buffer.to_dataframe(buffer_stats.rows)
                    self.buffer.extend(existing_data)
                logger.info("Buffer resized successfully")

        except Exception as exc:
            logger.exception("Error adjusting buffer size: %s", exc)

    def _record_failure(self, exc: Exception) -> None:
        self._health.consecutive_failures += 1
        self._health.last_error = repr(exc)
        self._health.last_error_at = datetime.now(timezone.utc)
        if self._health.consecutive_failures > 3:
            self._health.status = "degraded"
        if self._health.consecutive_failures > 6:
            self._health.status = "error"
        if self.observability:
            self.observability.log_event(
                "stream_failure",
                {
                    "error": repr(exc),
                    "consecutive_failures": self._health.consecutive_failures,
                },
                level=logging.WARNING,
            )
            self.observability.record_metrics(
                {"stream_failures": float(self._health.consecutive_failures)}
            )

    def _record_success(self) -> None:
        self._health.status = "ok"
        self._health.consecutive_failures = 0
        self._health.last_success_at = datetime.now(timezone.utc)
        self._backoff_seconds = 1.0
        if self.observability:
            self.observability.record_metrics(
                {"stream_failures": float(self._health.consecutive_failures)}
            )


# Factory function for registry integration
def create_streaming_pipeline(**kwargs: Any) -> StreamingPipeline:
    """Create a streaming pipeline using the registry pattern.

    This function maintains backward compatibility while enabling
    registry-based instantiation.
    """
    # Extract streaming-specific parameters
    buffer_capacity = kwargs.get("buffer_capacity", 1_000_000)
    feature_names = kwargs.get("feature_names")
    lookback_rows = kwargs.get("lookback_rows", 512)

    # Create CoinGecko stream client
    stream_client = CoinGeckoStream()

    # Create streaming pipeline
    pipeline = StreamingPipeline(
        stream_client,
        buffer_capacity=buffer_capacity,
        feature_names=feature_names,
        lookback_rows=lookback_rows,
    )

    return pipeline
