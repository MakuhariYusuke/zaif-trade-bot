import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import psutil
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types

if 'ztb.features' not in sys.modules:
    fake_features = types.ModuleType('ztb.features')

    class _FakeRegistry:
        @classmethod
        def initialize(cls) -> None:
            return None

        @classmethod
        def list(cls):
            return []

    fake_features.FeatureRegistry = _FakeRegistry
    sys.modules['ztb.features'] = fake_features

    fake_feature_engine = types.ModuleType('ztb.features.feature_engine')

    def _compute_features_batch(df, feature_names=None, **_kwargs):
        return pd.DataFrame(index=df.index)

    fake_feature_engine.compute_features_batch = _compute_features_batch
    sys.modules['ztb.features.feature_engine'] = fake_feature_engine

from ztb.data.coin_gecko_stream import MarketDataBatch, StreamConfig
from ztb.data.streaming_pipeline import StreamingPipeline


class SyntheticStream:
    """Minimal CoinGecko-compatible stream emitting synthetic batches."""

    def __init__(self, batch_rows: int = 64, *, fail_on: Optional[int] = None) -> None:
        self.batch_rows = batch_rows
        self._counter = 0
        self.fail_on = fail_on

    def _frame(self) -> pd.DataFrame:
        base_ts = pd.Timestamp.utcnow().floor('s')
        offsets = pd.to_timedelta(range(self.batch_rows), unit='s')
        prices = 100 + self._counter + pd.Series(range(self.batch_rows)) * 0.01
        df = pd.DataFrame(
            {
                "timestamp": base_ts + offsets,
                "price": prices.astype('float32'),
                "market_cap": (prices * 1000).astype('float32'),
                "volume": (prices * 10).astype('float32'),
            }
        )
        self._counter += 1
        return df

    def fetch_range(
        self,
        coin_id: str,
        vs_currency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        granularity: str = "1m",
        page_size: int = 200,
    ) -> MarketDataBatch:
        return MarketDataBatch(self._frame(), pd.Timestamp.utcnow(), {})

    def stream(
        self,
        config: StreamConfig,
        *,
        start_at: Optional[pd.Timestamp] = None,
        stop_event: Optional[object] = None,
    ) -> Iterator[MarketDataBatch]:
        while True:
            if self.fail_on is not None and self._counter == self.fail_on:
                self.fail_on = None
                raise RuntimeError("Synthetic network failure")
            yield self.fetch_range(config.coin_id, config.vs_currency, pd.Timestamp.utcnow(), pd.Timestamp.utcnow())


@pytest.fixture()
def synthetic_pipeline() -> StreamingPipeline:
    stream = SyntheticStream(batch_rows=64)
    pipeline = StreamingPipeline(
        stream_client=stream,
        buffer_capacity=4096,
        lookback_rows=128,
        buffer_compression="zlib",
    )
    now = pd.Timestamp.utcnow()
    pipeline.prime(now - pd.Timedelta(minutes=10), now)
    yield pipeline
    pipeline.close()


def test_streaming_pipeline_long_run_stability(synthetic_pipeline: StreamingPipeline) -> None:
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)

    iterator = synthetic_pipeline.stream()
    rows = 0
    for _ in range(200):
        batch = next(iterator)
        rows += len(batch)

    rss_after = process.memory_info().rss / (1024 * 1024)
    stats = synthetic_pipeline.stats()

    assert rows > 0
    assert stats.buffer.rows <= synthetic_pipeline.buffer.capacity
    assert stats.health.status == "ok"
    assert rss_after - rss_before < 50  # MB


def test_streaming_pipeline_recovers_from_failure() -> None:
    stream = SyntheticStream(batch_rows=32, fail_on=3)
    pipeline = StreamingPipeline(
        stream_client=stream,
        buffer_capacity=1024,
        lookback_rows=32,
        buffer_compression="zlib",
    )
    now = pd.Timestamp.utcnow()
    pipeline.prime(now - pd.Timedelta(minutes=5), now)

    iterator = pipeline.stream()
    batches = []
    for _ in range(6):
        try:
            batches.append(next(iterator))
        except RuntimeError:
            break

    # Allow pipeline to recover after the synthetic fault
    for _ in range(3):
        batches.append(next(iterator))

    stats = pipeline.stats()
    pipeline.close()

    assert any(len(batch) > 0 for batch in batches)
    assert stats.health.status in {"ok", "degraded"}
    assert stats.health.consecutive_failures == 0
