"""Benchmark harness for streaming pipeline performance.

Usage:
    python -m ztb.benchmarks.streaming_benchmark --steps 10000 --batch-size 256

The benchmark generates synthetic CoinGecko-like market data so it can run in
CI without external network calls. Results are written to JSON for downstream
analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.utils.observability import setup_observability, generate_correlation_id

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

import numpy as np
import psutil

from ztb.data.coin_gecko_stream import MarketDataBatch, StreamConfig
from ztb.utils.observability import setup_observability
from ztb.data.streaming_pipeline import PipelineStats, StreamingPipeline


@dataclass
class BenchmarkResult:
    steps: int
    rows_streamed: int
    duration_seconds: float
    rows_per_second: float
    batches_per_second: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    pipeline_stats: dict
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class SyntheticCoinGeckoStream:
    """CoinGeckoStream compatible generator for benchmarking."""

    def __init__(self, batch_rows: int, feature_count: int, *, jitter: float = 0.01) -> None:
        self.batch_rows = batch_rows
        self.jitter = jitter
        self._base_ts = datetime.now(timezone.utc) - timedelta(minutes=1000)
        self._counter = 0

    def _make_frame(self) -> pd.DataFrame:
        idx = np.arange(self.batch_rows, dtype=np.int64)
        timestamps = self._base_ts + pd.to_timedelta(self._counter * self.batch_rows + idx, unit="m")
        base_price = 100.0 + self._counter * 0.5
        price_noise = np.random.randn(self.batch_rows).astype(np.float32) * self.jitter
        prices = base_price + price_noise

        data = {
            "timestamp": timestamps,
            "price": prices,
            "market_cap": prices * 1000,
            "volume": np.abs(np.random.randn(self.batch_rows)).astype(np.float32) * 100,
        }

        self._counter += 1
        return pd.DataFrame(data)

    # CoinGecko API compatibility ------------------------------------
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
        df = self._make_frame()
        return MarketDataBatch(df, datetime.now(timezone.utc), {})

    def stream(
        self,
        config: StreamConfig,
        *,
        start_at: Optional[datetime] = None,
        stop_event: Optional[object] = None,
    ) -> Iterator[MarketDataBatch]:
        while True:
            yield self.fetch_range(config.coin_id, config.vs_currency, datetime.now(timezone.utc), datetime.now(timezone.utc))


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    correlation_id = generate_correlation_id()
    stream = SyntheticCoinGeckoStream(batch_rows=args.batch_size, feature_count=args.synthetic_features)
    observability = setup_observability(
        'streaming_benchmark',
        args.output.parent / 'observability',
        correlation_id
    )
    pipeline = StreamingPipeline(
        stream_client=stream,
        buffer_capacity=args.buffer_capacity,
        lookback_rows=args.lookback,
        async_features=args.async_features,
        buffer_compression=args.buffer_compression,
        observability=observability,
    )

    prime_end = datetime.now(timezone.utc)
    prime_start = prime_end - timedelta(minutes=args.batch_size)
    pipeline.prime(prime_start, prime_end)

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)

    rows = 0
    start_time = time.perf_counter()
    stream_iter = pipeline.stream()
    for _ in range(args.steps):
        features = next(stream_iter)
        rows += len(features)
    duration = time.perf_counter() - start_time

    rss_after = process.memory_info().rss / (1024 * 1024)
    stats = pipeline.stats()
    pipeline.close()

    batches_per_second = args.steps / duration if duration else math.inf
    rows_per_second = rows / duration if duration else math.inf

    result = BenchmarkResult(
        steps=args.steps,
        rows_streamed=rows,
        duration_seconds=duration,
        rows_per_second=rows_per_second,
        batches_per_second=batches_per_second,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=rss_after - rss_before,
        pipeline_stats={
            "total_rows_streamed": stats.total_rows_streamed,
            "last_batch_rows": stats.last_batch_rows,
            "health": stats.health.as_dict(),
        },
        timestamp=datetime.utcnow().isoformat(),
    )

    observability.log_event(
        'benchmark_complete',
        {
            'steps': args.steps,
            'rows_streamed': rows,
            'duration_seconds': duration,
        },
    )
    observability.record_metrics({
        'benchmark_rows_per_second': rows_per_second,
        'benchmark_batches_per_second': batches_per_second,
        'benchmark_rss_delta_mb': rss_after - rss_before,
    })
    observability.export_artifact('streaming_benchmark', asdict(result))
    observability.close()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming pipeline benchmark")
    parser.add_argument("--steps", type=int, default=2000, help="Number of streaming batches to process")
    parser.add_argument("--batch-size", type=int, default=256, help="Synthetic rows per batch")
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Stream buffer capacity")
    parser.add_argument("--lookback", type=int, default=512, help="Feature lookback window")
    parser.add_argument("--async-features", action="store_true", help="Enable async feature computation")
    parser.add_argument("--buffer-compression", default="zlib", help="Compression algorithm for buffer")
    parser.add_argument("--synthetic-features", type=int, default=8, help="Number of synthetic extra features")
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/streaming_benchmark.json"), help="Output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.to_json(), encoding="utf-8")
    print(result.to_json())


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
