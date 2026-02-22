#!/usr/bin/env python3
"""List available TensorBoard scalar tags for reward improvement runs."""

from pathlib import Path
from typing import List

from tensorboard.backend.event_processing import event_accumulator

RUNS = {
    "v378_scale": Path("checkpoints/ppo_reward_v378_scale/ppo_reward_v378_scale_1"),
    "v379_dynamic": Path(
        "checkpoints/ppo_reward_v379_dynamic/ppo_reward_v379_dynamic_1"
    ),
    "v380_aggressive": Path(
        "checkpoints/ppo_reward_v380_aggressive/ppo_reward_v380_aggressive_1"
    ),
}


def list_scalars(run_dir: Path) -> List[str]:
    events_file = next(run_dir.glob("**/events.out.tfevents.*"), None)
    if not events_file:
        raise FileNotFoundError(f"No events file found in {run_dir}")

    ea = event_accumulator.EventAccumulator(str(events_file))
    ea.Reload()
    tags = ea.Tags()
    raw_scalars = tags.get("scalars")
    if isinstance(raw_scalars, (list, tuple)):
        scalars = sorted(raw_scalars)
    else:
        scalars = []
    return scalars


def main() -> None:
    for name, path in RUNS.items():
        print("=" * 80)
        print(f"Scalar tags for {name}")
        print("=" * 80)
        try:
            scalars = list_scalars(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  Error: {exc}")
            continue
        for tag in scalars:
            print(f"  {tag}")
        print()


if __name__ == "__main__":
    main()
