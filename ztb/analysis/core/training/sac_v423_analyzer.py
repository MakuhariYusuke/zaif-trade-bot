#!/usr/bin/env python3
"""
SAC v423 Training Results Analyzer

Integrated analyzer for SAC v423 training results from archived script.
Provides comprehensive analysis of training metrics, reward progression, and action distribution.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.common.plot_utils import save_plot
from ztb.io.json_io import read_json
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv423Analyzer:
    """Analyzer for SAC v423 training results."""


    def find_latest_training_results(self) -> list[Path]:
        """Find all recent training results."""
        results = []

        if result_path.suffix == ".json" and "training_report" in result_path.name:
            try:
                report_data = read_json(result_path)
                # Extract metrics from the report structure
                training_stats = report_data.get("training_stats", {})
                metrics = {
                    "final_episode_reward": training_stats.get("final_reward", 0),
                    "best_episode_reward": training_stats.get(
                        "final_reward", 0
                    ),  # Use final_reward as best
                    "training_time_seconds": training_stats.get("training_time", 0),
                    "action_distribution": training_stats.get(
                        "action_distribution", {}
                    ),
                    "total_timesteps": training_stats.get("total_timesteps", 0),
                }
                return metrics
            except Exception as e:
                logger.error(f"Failed to extract metrics: {e}")
                return {}
        """Analyze the latest training results."""
        print("🔍 Analyzing SAC v423 Training Results")
        print("=" * 50)

        result_paths = self.find_latest_training_results()
        if not result_paths:
            print("❌ No training results found in results/sac_v423/ or reports/")
            return

        for result_path in result_paths:
            print(f"\n📁 Analyzing results from: {result_path}")

            # Load data
            metrics = self.load_training_metrics(result_path)
            config = self.load_model_config(result_path)

            if not metrics:
                print("❌ Training metrics not found")
                continue

            if not config:
                print("❌ Model config not found")
                continue

            # Display basic information
            print("\n📊 Training Summary:")
            print(f"   Model: {config.get('model_name', 'Unknown')}")
            print(f"   Algorithm: {config.get('algorithm', 'Unknown')}")
            print(
                f"   Total Timesteps: {config.get('training', {}).get('total_timesteps', 'Unknown'):,}"
            )

            # Analyze action distribution if available
            if "action_distribution" in metrics:
                actions = metrics["action_distribution"]
                print("\n🎯 Action Distribution:")
                total_actions = sum(actions.values())
                for action, count in actions.items():
                    percentage = (count / total_actions) * 100
                    print(f"   {action}: {count} ({percentage:.1f}%)")

            # Performance interpretation
            training_time = metrics.get("training_time_seconds", 0)
            total_timesteps = config.get("training", {}).get("total_timesteps", 0)
            if training_time > 0 and total_timesteps > 0:
                steps_per_sec = total_timesteps / training_time
                print(f"   Steps/sec: {steps_per_sec:.2f}")
                if steps_per_sec > 100:
                    print("   ✅ High training speed (good performance)")
                elif steps_per_sec > 50:
                    print("   ⚠️ Moderate training speed")
                else:
                    print("   ❌ Low training speed (optimization needed)")

            print(f"\n✅ Analysis complete for: {result_path}")

    def plot_reward_progression(self, rewards: list, output_dir: Path) -> None:
        """Plot reward progression over episodes."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards, label="Episode Reward", alpha=0.7)
            plt.title("SAC v423 Training - Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Save plot
            plot_file = output_dir / "reward_progression.png"
            save_plot(plot_file, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"   📊 Reward plot saved: {plot_file}")

        except Exception as e:
            logger.warning(f"Failed to create reward plot: {e}")
