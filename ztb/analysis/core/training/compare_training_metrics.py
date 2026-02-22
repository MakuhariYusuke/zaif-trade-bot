#!/usr/bin/env python3
"""Compare detailed TensorBoard metrics for reward improvement runs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tensorboard.backend.event_processing import event_accumulator
from ztb.io.json_io import write_json

RUNS = {
    "v378_scale": Path("checkpoints/ppo_reward_v378_scale/ppo_reward_v378_scale_1"),
    "v379_dynamic": Path(
        "checkpoints/ppo_reward_v379_dynamic/ppo_reward_v379_dynamic_1"
    ),
    "v380_aggressive": Path(
        "checkpoints/ppo_reward_v380_aggressive/ppo_reward_v380_aggressive_1"
    ),
}

SCALARS_OF_INTEREST = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/learning_rate",
]


@dataclass
class MetricStats:
    tag: str
    steps: List[int]
    values: List[float]

    @property
    def final_value(self) -> float:
        return self.values[-1]

    @property
    def max_value(self) -> float:
        return max(self.values)

    @property
    def min_value(self) -> float:
        return min(self.values)

    @property
    def mean_value(self) -> float:
        return statistics.fmean(self.values)

    @property
    def std_value(self) -> float:
        return statistics.pstdev(self.values)

    def to_dict(self) -> Dict[str, float]:
        return {
            "final": self.final_value,
            "max": self.max_value,
            "min": self.min_value,
            "mean": self.mean_value,
            "std": self.std_value,
        }


def load_metrics(events_dir: Path) -> Dict[str, MetricStats]:
    events_file = next(events_dir.glob("**/events.out.tfevents.*"), None)
    if not events_file:
        raise FileNotFoundError(f"No TensorBoard events file under {events_dir}")

    ea = event_accumulator.EventAccumulator(str(events_file))
    ea.Reload()

    stats: Dict[str, MetricStats] = {}
    tags_dict = ea.Tags()
    raw_scalar_tags = tags_dict["scalars"] if "scalars" in tags_dict else []
    scalar_tags = (
        list(raw_scalar_tags) if isinstance(raw_scalar_tags, (list, tuple)) else []
    )

    for tag in SCALARS_OF_INTEREST:
        if tag not in scalar_tags:
            continue
        scalars = ea.Scalars(tag)
        steps = [int(s.step) for s in scalars]
        values = [float(s.value) for s in scalars]
        stats[tag] = MetricStats(tag=tag, steps=steps, values=values)
    return stats


def main() -> None:
    comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
    print("=" * 100)
    print("Detailed TensorBoard Metric Comparison")
    print("=" * 100)

    for name, path in RUNS.items():
        print(f"\n--- {name} ---")
        metrics = load_metrics(path)
        run_summary: Dict[str, Dict[str, float]] = {}
        for tag in SCALARS_OF_INTEREST:
            if tag not in metrics:
                continue
            stats = metrics[tag]
            run_summary[tag] = stats.to_dict()
            print(
                f"{tag:<28} final={stats.final_value:8.3f}  max={stats.max_value:8.3f}  "
                f"min={stats.min_value:8.3f}  mean={stats.mean_value:8.3f}  std={stats.std_value:7.3f}"
            )
        comparison[name] = run_summary

    output_path = Path("reward_metric_comparison.json")
    write_json(output_path, comparison, indent=2, ensure_ascii=False)
    print(f"\nSaved metric summary to {output_path}")


def compare_training_metrics(
    logdirs: List[str] = None,
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compare training metrics across different runs.

    Args:
        logdirs: List of log directories to compare
        metrics: List of metrics to compare
        output_path: Path to save results

    Returns:
        Dictionary with comparison results
    """
    if logdirs is None:
        logdirs = [str(path) for path in RUNS.values()]

    if metrics is None:
        metrics = SCALARS_OF_INTEREST

    comparison = {}
    for logdir in logdirs:
        path = Path(logdir)
        if not path.exists():
            continue

        print(f"\n--- {path.name} ---")
        metrics_data = load_metrics(path)
        run_summary = {}
        for tag in metrics:
            if tag not in metrics_data:
                continue
            stats = metrics_data[tag]
            run_summary[tag] = stats.to_dict()
            print(
                f"{tag:<28} final={stats.final_value:8.3f}  max={stats.max_value:8.3f}  "
                f"min={stats.min_value:8.3f}  mean={stats.mean_value:8.3f}  std={stats.std_value:7.3f}"
            )
        comparison[path.name] = run_summary

    if output_path:
        output_file = Path(output_path)
        write_json(output_file, comparison, indent=2, ensure_ascii=False)
        print(f"\nSaved metric summary to {output_file}")

    return comparison


if __name__ == "__main__":
    main()
