"""Benchmark harness for checkpoint save/load performance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import psutil

from ztb.training.checkpoint_manager import TrainingCheckpointConfig, TrainingCheckpointManager
from ztb.utils.observability import setup_observability, generate_correlation_id


@dataclass
class CheckpointBenchmarkResult:
    saves: int
    save_duration_seconds: float
    loads: int
    load_duration_seconds: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    snapshot_metrics: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class _DummyOptimizer:
    def __init__(self, size: int) -> None:
        self.state = {"momentum": np.zeros(size, dtype=np.float32)}

    def state_dict(self) -> Dict[str, Any]:
        return {"state": {"momentum": self.state["momentum"].copy()}}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        momentum = state.get("state", {}).get("momentum")
        if momentum is not None:
            self.state["momentum"] = momentum.copy()


class _DummyPolicy:
    def __init__(self, size: int) -> None:
        self.weights = np.random.randn(size).astype(np.float32)
        self.optimizer = _DummyOptimizer(size)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.copy(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False) -> None:
        self.weights = state["weights"].copy()
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])


class _DummyModel:
    def __init__(self, size: int) -> None:
        self.policy = _DummyPolicy(size)
        self.replay_buffer = {"states": np.random.randn(size, 4).astype(np.float32)}

    def save_replay_buffer(self, path: str) -> None:
        with open(path, "wb") as fh:
            np.save(fh, self.replay_buffer["states"], allow_pickle=False)

    def load_replay_buffer(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.replay_buffer["states"] = np.load(fh, allow_pickle=False)


def run_benchmark(args: argparse.Namespace) -> CheckpointBenchmarkResult:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for file in args.output_dir.glob('checkpoint*.pkl*'):
        try:
            file.unlink()
        except OSError:
            pass

    correlation_id = generate_correlation_id()
    observability = setup_observability(
        'checkpoint_benchmark',
        args.output_dir / 'observability',
        correlation_id
    )

    manager = TrainingCheckpointManager(
        save_dir=str(args.output_dir),
        config=TrainingCheckpointConfig(
            interval_steps=args.interval_steps,
            compress=args.compression,
            async_save=False,
            include_replay_buffer=True,
            include_rng_state=False,
        ),
        observability=observability,
    )

    model = _DummyModel(args.model_size)
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 * 1024)

    start = time.perf_counter()
    for step in range(args.saves):
        manager.save(
            step=step + 1,
            model=model,
            metrics={"reward": float(step)},
            stream_state={},
            async_save=False,
        )
    save_duration = time.perf_counter() - start

    snapshot = manager.load_latest()
    load_start = time.perf_counter()
    for _ in range(args.loads):
        if snapshot is None:
            snapshot = manager.load_latest()
        manager.apply_snapshot(model, snapshot)
    load_duration = time.perf_counter() - load_start

    rss_after = process.memory_info().rss / (1024 * 1024)

    result = CheckpointBenchmarkResult(
        saves=args.saves,
        save_duration_seconds=save_duration,
        loads=args.loads,
        load_duration_seconds=load_duration,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=rss_after - rss_before,
        snapshot_metrics=snapshot.metadata if snapshot else {},
    )

    manager.shutdown()
    observability.log_event(
        'checkpoint_benchmark_complete',
        {
            'saves': args.saves,
            'loads': args.loads,
            'save_duration_seconds': save_duration,
            'load_duration_seconds': load_duration,
        },
    )
    observability.record_metrics({
        'benchmark_checkpoint_save_seconds': save_duration,
        'benchmark_checkpoint_load_seconds': load_duration,
        'benchmark_checkpoint_rss_delta_mb': rss_after - rss_before,
    })
    observability.export_artifact('checkpoint_benchmark', asdict(result))
    observability.close()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint performance benchmark")
    parser.add_argument("--saves", type=int, default=5, help="Number of save operations to benchmark")
    parser.add_argument("--loads", type=int, default=5, help="Number of load/apply operations")
    parser.add_argument("--model-size", type=int, default=100_000, help="Synthetic model parameter count")
    parser.add_argument("--interval-steps", type=int, default=1000)
    parser.add_argument("--compression", default="zlib")
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/checkpoint_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmarks/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(args)
    args.output.write_text(result.to_json(), encoding="utf-8")
    print(result.to_json())


if __name__ == "__main__":  # pragma: no cover
    main()
