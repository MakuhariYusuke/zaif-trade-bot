#!/usr/bin/env python3
"""Visualize policy updates with different clip_range values."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.training.ppo_trainer import PPOTrainer  # noqa: E402

LOGGER = logging.getLogger(__name__)


def simulate_policy_updates(
    clip_range: float, n_updates: int = 50
) -> Dict[str, List[float]]:
    """Simulate policy updates with given clip_range and track metrics."""
    try:
        config = {
            "algorithm": "ppo",
            "data_path": "data/ml-dataset-enhanced-balanced.csv",
            "total_timesteps": 10000,  # Short training for visualization
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": clip_range,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "ent_coef": 0.5,
            "tensorboard_log": f"logs/clip_viz_{clip_range}",
            "model_dir": f"models/clip_viz_{clip_range}",
            "checkpoint_dir": f"checkpoints/clip_viz_{clip_range}",
            "log_dir": f"logs/clip_viz_{clip_range}",
            "offline_mode": True,
            "feature_set": "full",
            "timeframe": "1m",
            "reward_scaling": 1.0,
            "transaction_cost": 0.0,
            "max_position_size": 1.0,
            "seed": 42,
        }

        trainer = PPOTrainer(
            data_path=config["data_path"],
            config=config,
            checkpoint_dir=config["checkpoint_dir"],
        )

        # Track metrics over updates
        rewards = []
        kl_divs = []
        clip_fractions = []
        policy_losses = []

        # Monkey patch to capture metrics
        original_update = trainer.model.update

        def patched_update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            # Extract metrics from the update result
            if hasattr(result, "infos") and result.infos:
                info = result.infos[0]  # First environment info
                rewards.append(info.get("reward", 0))
                kl_divs.append(info.get("kl_div", 0))
                clip_fractions.append(info.get("clip_fraction", 0))
                policy_losses.append(info.get("policy_loss", 0))
            return result

        trainer.model.update = patched_update

        # Train and collect metrics
        trainer.train(session_id=f"clip_viz_{clip_range}")

        return {
            "rewards": rewards[:n_updates],
            "kl_divs": kl_divs[:n_updates],
            "clip_fractions": clip_fractions[:n_updates],
            "policy_losses": policy_losses[:n_updates],
        }

    except Exception as e:
        LOGGER.error(f"Simulation failed for clip_range {clip_range}: {e}")
        return {
            "rewards": [],
            "kl_divs": [],
            "clip_fractions": [],
            "policy_losses": [],
        }


def create_visualization(clip_ranges: List[float], output_dir: Path) -> None:
    """Create visualization comparing different clip_range values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for clip_range in clip_ranges:
        LOGGER.info(f"Simulating clip_range = {clip_range}")
        metrics = simulate_policy_updates(clip_range)
        all_metrics[clip_range] = metrics

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Policy Update Behavior with Different Clip Ranges", fontsize=16)

    colors = ["blue", "red", "green", "orange"]

    for i, (clip_range, metrics) in enumerate(all_metrics.items()):
        color = colors[i % len(colors)]
        label = f"clip_range={clip_range}"

        # Rewards
        axes[0, 0].plot(metrics["rewards"], color=color, label=label, alpha=0.7)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Update Step")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()

        # KL Divergence
        axes[0, 1].plot(metrics["kl_divs"], color=color, label=label, alpha=0.7)
        axes[0, 1].set_title("KL Divergence")
        axes[0, 1].set_xlabel("Update Step")
        axes[0, 1].set_ylabel("KL Div")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()

        # Clip Fraction
        axes[1, 0].plot(metrics["clip_fractions"], color=color, label=label, alpha=0.7)
        axes[1, 0].set_title("Clip Fraction")
        axes[1, 0].set_xlabel("Update Step")
        axes[1, 0].set_ylabel("Clip Fraction")
        axes[1, 0].legend()

        # Policy Loss
        axes[1, 1].plot(metrics["policy_losses"], color=color, label=label, alpha=0.7)
        axes[1, 1].set_title("Policy Loss")
        axes[1, 1].set_xlabel("Update Step")
        axes[1, 1].set_ylabel("Policy Loss")
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "clip_range_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create summary table
    summary_data = []
    for clip_range, metrics in all_metrics.items():
        if metrics["rewards"]:
            summary_data.append(
                {
                    "clip_range": clip_range,
                    "mean_reward": np.mean(metrics["rewards"]),
                    "std_reward": np.std(metrics["rewards"]),
                    "mean_kl_div": (
                        np.mean(metrics["kl_divs"]) if metrics["kl_divs"] else 0
                    ),
                    "mean_clip_fraction": (
                        np.mean(metrics["clip_fractions"])
                        if metrics["clip_fractions"]
                        else 0
                    ),
                    "final_policy_loss": (
                        metrics["policy_losses"][-1] if metrics["policy_losses"] else 0
                    ),
                }
            )

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(output_dir / "clip_range_summary.csv", index=False)

        # Display summary
        LOGGER.info("Clip Range Comparison Summary:")
        LOGGER.info(df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize PPO policy updates with different clip_range values"
    )
    parser.add_argument(
        "--clip-ranges",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Clip range values to compare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/clip_viz"),
        help="Output directory for plots and data",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    LOGGER.info(f"Comparing clip ranges: {args.clip_ranges}")
    create_visualization(args.clip_ranges, args.output_dir)

    LOGGER.info(f"Visualization completed! Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
