#!/usr/bin/env python3
"""Extract and compare action distribution metrics from short training runs."""

import json
from pathlib import Path
from typing import Any, Dict

from tensorboard.backend.event_processing import event_accumulator

RUNS = {
    "v378_scale_short": Path(
        "checkpoints/ppo_reward_v378_scale_short/ppo_reward_v378_scale_short_1"
    ),
    "v379_dynamic_short": Path(
        "checkpoints/ppo_reward_v379_dynamic_short/ppo_reward_v379_dynamic_short_1"
    ),
    "v380_aggressive_short": Path(
        "checkpoints/ppo_reward_v380_aggressive_short/ppo_reward_v380_aggressive_short_1"
    ),
}


def extract_action_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract action distribution metrics from TensorBoard events."""
    events_file = next(run_dir.glob("**/events.out.tfevents.*"), None)
    if not events_file:
        raise FileNotFoundError(f"No events file found in {run_dir}")

    ea = event_accumulator.EventAccumulator(
        str(events_file),
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )
    ea.Reload()

    metrics = {}

    # Extract action counts and percentages
    for action in ["hold", "buy", "sell"]:
        count_tag = f"pan_action_counts/{action}"
        pct_tag = f"pan_action_pct/{action}"

        if count_tag in ea.Tags()["scalars"]:
            count_values = [s.value for s in ea.Scalars(count_tag)]
            metrics[f"{action}_count_mean"] = sum(count_values) / len(count_values)
            metrics[f"{action}_count_final"] = count_values[-1]

        if pct_tag in ea.Tags()["scalars"]:
            pct_values = [s.value for s in ea.Scalars(pct_tag)]
            metrics[f"{action}_pct_mean"] = sum(pct_values) / len(pct_values)
            metrics[f"{action}_pct_final"] = pct_values[-1]

    # Calculate total actions and ratios
    if all(f"{action}_count_final" in metrics for action in ["hold", "buy", "sell"]):
        total_final = sum(
            metrics[f"{action}_count_final"] for action in ["hold", "buy", "sell"]
        )
        for action in ["hold", "buy", "sell"]:
            metrics[f"{action}_ratio_final"] = (
                metrics[f"{action}_count_final"] / total_final
            )

    return metrics


def main() -> None:
    results = {}

    for name, run_dir in RUNS.items():
        print(f"Processing {name}...")
        try:
            metrics = extract_action_metrics(run_dir)
            results[name] = metrics
            print(f"  ✅ Extracted {len(metrics)} metrics")
        except Exception as exc:
            print(f"  ❌ Error: {exc}")
            results[name] = {"error": str(exc)}

    # Save results
    output_file = Path("action_distribution_comparison.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION COMPARISON")
    print("=" * 80)

    for name, metrics in results.items():
        if "error" in metrics:
            print(f"{name}: ERROR - {metrics['error']}")
            continue

        print(f"\n{name}:")
        print(".1f")
        print(".1f")
        print(".1f")

        if "hold_ratio_final" in metrics:
            print(".1%")
            print(".1%")
            print(".1%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
