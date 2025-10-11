#!/usr/bin/env python3
"""Inspect full TensorBoard tag categories for reward improvement runs."""

from pathlib import Path
from typing import Dict, Any

from tensorboard.backend.event_processing import event_accumulator

RUNS = {
    "v378_scale_short": Path("checkpoints/ppo_reward_v378_scale_short/ppo_reward_v378_scale_short_1"),
    "v379_dynamic_short": Path("checkpoints/ppo_reward_v379_dynamic_short/ppo_reward_v379_dynamic_short_1"),
    "v380_aggressive_short": Path("checkpoints/ppo_reward_v380_aggressive_short/ppo_reward_v380_aggressive_short_1"),
}


def inspect_tags(run_dir: Path) -> Dict[str, Any]:
    events_file = next(run_dir.glob("**/events.out.tfevents.*"), None)
    if not events_file:
        raise FileNotFoundError(f"No events file found in {run_dir}")
    ea = event_accumulator.EventAccumulator(
        str(events_file),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
            event_accumulator.HISTOGRAMS: 0,
        },
    )
    ea.Reload()
    return ea.Tags()


def main() -> None:
    for name, run_dir in RUNS.items():
        print("=" * 80)
        print(f"Tags for {name}")
        print("=" * 80)
        try:
            tags = inspect_tags(run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  Error: {exc}")
            continue
        for category, tag_list in tags.items():
            if not tag_list:
                continue
            print(f"[{category}]")
            for tag in sorted(tag_list):
                print(f"  {tag}")
            print()


if __name__ == "__main__":
    main()
