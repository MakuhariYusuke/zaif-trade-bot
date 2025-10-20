#!/usr/bin/env python3
"""
Analyze and compare reward function improvement results (v378, v379, v380)

This script analyzes the training results from three reward configurations:
- v378: Scale-adjusted rewards
- v379: Dynamic market-adaptive rewards
- v380: Aggressive anti-HOLD rewards

Compares:
- HOLD rates
- Total rewards
- Action distributions
- Performance metrics
"""

import json
import os
from typing import Dict, Optional

# Checkpoint directories
CHECKPOINTS = {
    "v378_scale": "checkpoints/ppo_reward_v378_scale/ppo_reward_v378_scale_1",
    "v379_dynamic": "checkpoints/ppo_reward_v379_dynamic/ppo_reward_v379_dynamic_1",
    "v380_aggressive": "checkpoints/ppo_reward_v380_aggressive/ppo_reward_v380_aggressive_1",
}

# Config files
CONFIGS = {
    "v378_scale": "configs/training/ppo_reward_v378_scale.json",
    "v379_dynamic": "configs/training/ppo_reward_v379_dynamic.json",
    "v380_aggressive": "configs/training/ppo_reward_v380_aggressive.json",
}


def load_config(config_path: str) -> Dict:
    """Load configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_tensorboard_events(checkpoint_dir: str) -> Optional[Dict]:
    """Extract metrics from tensorboard events file."""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        events_file = None
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    events_file = os.path.join(root, file)
                    break
            if events_file:
                break

        if not events_file:
            print(f"  ⚠️ No tensorboard events file found in {checkpoint_dir}")
            return None

        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()

        metrics = {}

        # Extract available scalars
        for tag in ea.Tags()["scalars"]:
            values = ea.Scalars(tag)
            if values:
                # Get final value
                metrics[tag] = values[-1].value
                # Also store trajectory
                metrics[f"{tag}_trajectory"] = [(v.step, v.value) for v in values]

        return metrics
    except Exception as e:
        print(f"  ❌ Error loading tensorboard events: {e}")
        return None


def analyze_checkpoint_dir(name: str, checkpoint_dir: str, config_path: str) -> Dict:
    """Analyze a single checkpoint directory."""
    print(f"\n{'─'*80}")
    print(f"Analyzing: {name}")
    print(f"{'─'*80}")

    result = {
        "name": name,
        "config_path": config_path,
        "checkpoint_dir": checkpoint_dir,
        "exists": os.path.exists(checkpoint_dir),
    }

    if not result["exists"]:
        print(f"  ❌ Checkpoint directory not found: {checkpoint_dir}")
        return result

    # Load config
    config = load_config(config_path)
    result["config"] = config
    result["reward_settings"] = config.get("environment", {}).get("reward_settings", {})

    print("  ✅ Config loaded")
    print("  Reward settings:")
    for key, value in result["reward_settings"].items():
        print(f"    {key}: {value}")

    # Analyze tensorboard events
    print("\n  Analyzing tensorboard events...")
    metrics = analyze_tensorboard_events(checkpoint_dir)
    if metrics:
        result["metrics"] = metrics
        print(
            f"  ✅ Found {len([k for k in metrics.keys() if not k.endswith('_trajectory')])} metrics"
        )

        # Print key metrics
        key_metrics = [
            "rollout/ep_rew_mean",
            "train/entropy_loss",
            "train/policy_gradient_loss",
            "train/value_loss",
        ]

        print("\n  Key metrics (final values):")
        for metric in key_metrics:
            if metric in metrics:
                print(f"    {metric}: {metrics[metric]:.4f}")

    return result


def compare_results(results: Dict[str, Dict]) -> None:
    """Compare results across all configurations."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    # Create comparison table
    headers = ["Metric", "v378_scale", "v379_dynamic", "v380_aggressive"]
    print(f"{headers[0]:<40} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20}")
    print("─" * 100)

    # Reward settings comparison
    print("\n📊 Reward Settings:")
    print("─" * 100)

    reward_keys = [
        "hold_penalty_weight",
        "consecutive_hold_penalty",
        "trading_frequency_bonus",
        "profit_reward_multiplier",
    ]

    for key in reward_keys:
        row = [key]
        for name in ["v378_scale", "v379_dynamic", "v380_aggressive"]:
            if name in results and "reward_settings" in results[name]:
                value = results[name]["reward_settings"].get(key, "N/A")
                row.append(str(value))
            else:
                row.append("N/A")
        print(f"{row[0]:<40} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    # Performance metrics comparison
    print("\n📈 Performance Metrics (Final Values):")
    print("─" * 100)

    metric_keys = [
        "rollout/ep_rew_mean",
        "train/entropy_loss",
        "train/policy_gradient_loss",
        "train/value_loss",
    ]

    for key in metric_keys:
        row = [key]
        for name in ["v378_scale", "v379_dynamic", "v380_aggressive"]:
            if name in results and "metrics" in results[name]:
                value = results[name]["metrics"].get(key, "N/A")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            else:
                row.append("N/A")
        print(f"{row[0]:<40} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    # Analysis and recommendations
    print(f"\n{'='*80}")
    print("ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*80}\n")

    # Compare rewards
    rewards = {}
    for name in ["v378_scale", "v379_dynamic", "v380_aggressive"]:
        if name in results and "metrics" in results[name]:
            rew = results[name]["metrics"].get("rollout/ep_rew_mean")
            if rew is not None:
                rewards[name] = rew

    if rewards:
        best = max(rewards.items(), key=lambda x: x[1])
        print(f"🏆 Best performing config: {best[0]} (reward: {best[1]:.4f})")

        print("\n📊 Reward comparison:")
        for name in sorted(rewards.keys(), key=lambda x: rewards[x], reverse=True):
            print(f"  {name}: {rewards[name]:.4f}")

    print(f"\n{'='*80}")
    print("Next steps:")
    print("1. Review tensorboard logs for detailed training curves")
    print("2. Check action distribution (HOLD/BUY/SELL rates)")
    print("3. Run backtest on best performing model")
    print("4. Analyze risk-adjusted metrics (Sharpe ratio, max drawdown)")
    print(f"{'='*80}\n")


def main() -> None:
    """Main analysis function."""
    print("=" * 80)
    print("Reward Function Improvement Analysis")
    print("=" * 80)

    results = {}

    # Analyze each checkpoint
    for name, checkpoint_dir in CHECKPOINTS.items():
        config_path = CONFIGS[name]
        results[name] = analyze_checkpoint_dir(name, checkpoint_dir, config_path)

    # Compare results
    compare_results(results)

    # Save detailed results
    output_file = "reward_improvements_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        # Remove trajectories for cleaner JSON
        clean_results = {}
        for name, data in results.items():
            clean_data = {k: v for k, v in data.items() if k != "metrics"}
            if "metrics" in data:
                clean_data["metrics"] = {
                    k: v
                    for k, v in data["metrics"].items()
                    if not k.endswith("_trajectory")
                }
            clean_results[name] = clean_data

        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
