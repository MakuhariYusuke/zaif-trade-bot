from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List

import pandas as pd
import sys
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


class _FakeStream:
    def __init__(self, frames: List[pd.DataFrame]) -> None:
        self._frames = frames

    def fetch_range(
        self,
        coin_id: str,
        vs_currency: str,
        start: datetime,
        end: datetime,
        *,
        granularity: str = "1m",
        page_size: int = 200,
    ) -> MarketDataBatch:
        return MarketDataBatch(self._frames[0], datetime.now(timezone.utc), {})

    def stream(
        self,
        config: StreamConfig,
        *,
        start_at: datetime | None = None,
        stop_event=None,
    ) -> Iterator[MarketDataBatch]:
        for frame in self._frames[1:]:
            yield MarketDataBatch(frame, datetime.now(timezone.utc), {})


def _make_frame(start_ts: int) -> pd.DataFrame:
    timestamps = [start_ts + i for i in range(3)]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "price": [100.0 + i for i in range(3)],
            "market_cap": [1_000_000.0 + i for i in range(3)],
            "volume": [10_000.0 + i for i in range(3)],
        }
    )


def test_streaming_pipeline_prime_and_stream() -> None:
    frames = [_make_frame(1), _make_frame(10)]
    stream = _FakeStream(frames)
    pipeline = StreamingPipeline(
        stream_client=stream,
        buffer_capacity=10,
        feature_names=[],
        lookback_rows=2,
    )

    start = datetime.now(timezone.utc) - timedelta(minutes=2)
    end = datetime.now(timezone.utc)

    prime_features = pipeline.prime(start=start, end=end)
    assert not prime_features.empty
    assert len(pipeline.buffer) == len(frames[0])

    batch = next(pipeline.stream(start_at=None))
    assert not batch.empty
    assert len(pipeline.buffer) == len(frames[0]) + len(frames[1])

    stats = pipeline.stats()
    assert stats.buffer.rows == len(frames[0]) + len(frames[1])
