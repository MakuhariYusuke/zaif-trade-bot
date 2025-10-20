#!/usr/bin/env python3
"""
UI and display utilities for Unified Trainer.
"""

import time
from typing import Any, Dict, List, Optional

from ztb.utils.logging_utils import get_logger


class TrainingUI:
    """Enhanced UI for training progress and statistics display."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.start_time = None

    def print_header(self, algorithm: str, config_name: str):
        """Print training header."""
        print("\n" + "=" * 80)
        print("🤖 ZAIF TRADE BOT - UNIFIED TRAINER")
        print("=" * 80)
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Config: {config_name}")
        print("=" * 80)

    def print_config_summary(self, config: Dict[str, Any]):
        """Print configuration summary."""
        print("\n📋 CONFIGURATION SUMMARY:")
        print("-" * 40)

        # Algorithm info
        algorithm = config.get("algorithm", "unknown")
        print(f"Algorithm: {algorithm}")

        # Training parameters
        total_timesteps = config.get("total_timesteps", "unknown")
        if isinstance(total_timesteps, (int, float)):
            print(f"Total Timesteps: {total_timesteps:,}")
        else:
            print(f"Total Timesteps: {total_timesteps}")

        # Data info
        data_path = config.get("data_path", "unknown")
        print(f"Data Path: {data_path}")

        # Model info
        model_name = config.get("model_name", "unknown")
        print(f"Model Name: {model_name}")

        # SAC specific config
        if algorithm.lower() == "sac":
            sac_config = config.get("sac_hyperparameters", {})
            print(f"Learning Rate: {sac_config.get('learning_rate', 'default')}")
            buffer_size = sac_config.get("buffer_size", "default")
            if isinstance(buffer_size, (int, float)):
                print(f"Buffer Size: {buffer_size:,}")
            else:
                print(f"Buffer Size: {buffer_size}")
            print(f"Batch Size: {sac_config.get('batch_size', 'default')}")

        print("-" * 40)

    def start_training(self):
        """Initialize training UI."""
        self.start_time = time.time()
        print("\n🚀 Starting training...")

    def print_training_complete(
        self, success: bool, stats: Optional[Dict[str, Any]] = None
    ):
        """Print training completion status."""
        duration = time.time() - self.start_time if self.start_time else 0

        if success:
            print(f"\n✅ Training completed successfully in {duration:.1f}s")
            if stats:
                print("📊 Final Statistics:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        else:
            print(f"\n❌ Training failed after {duration:.1f}s")

    def print_training_progress_ensemble(
        self,
        step: int,
        total_steps: int,
        episode_reward: float,
        ensemble_stats: Dict[str, Any],
    ):
        """Print training progress with ensemble information."""
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        progress = (step / total_steps) * 100

        # Calculate ETA
        if step > 0:
            steps_per_sec = step / elapsed
            remaining_steps = total_steps - step
            eta_seconds = remaining_steps / steps_per_sec
            eta_str = (
                f"{eta_seconds/3600:.1f}h"
                if eta_seconds > 3600
                else f"{eta_seconds/60:.1f}m"
            )
        else:
            eta_str = "unknown"

        print(
            f"\r🔄 Step {step:,}/{total_steps:,} ({progress:.1f}%) | "
            f"Reward: {episode_reward:.2f} | "
            f"Ensemble Conf: {ensemble_stats.get('avg_confidence', 0):.2f} | "
            f"ETA: {eta_str}",
            end="",
            flush=True,
        )

    def update_ensemble_progress(self, step: int, ensemble_stats: Dict[str, Any]):
        """Update ensemble progress display."""
        avg_confidence = ensemble_stats.get("overall_stats", {}).get(
            "avg_confidence", 0
        )
        print(
            f"\r🎯 Ensemble | Confidence: {avg_confidence:.3f} | Members: {ensemble_stats.get('overall_stats', {}).get('total_members', 0)}",
            end="",
            flush=True,
        )

    def print_ensemble_adaptation(self, adaptation_info: Dict[str, Any]):
        """Print ensemble adaptation information."""
        print("\n🔄 ENSEMBLE ADAPTATION:")
        print("-" * 40)

        market_conditions = adaptation_info.get("market_conditions", {})
        if market_conditions:
            print("Market Conditions:")
            for key, value in market_conditions.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        changes = adaptation_info.get("changes", [])
        if changes:
            print("Adaptation Changes:")
            for change in changes:
                print(f"  • {change}")

        print(f"Adaptation completed in {adaptation_info.get('duration', 0):.2f}s")

    def print_error_with_suggestions(
        self, error: str, suggestions: Optional[List[str]] = None
    ):
        """Print error with helpful suggestions."""
        print(f"\n❌ ERROR: {error}")

        if suggestions:
            print("\n💡 SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")

    def print_success_with_metrics(
        self, message: str, metrics: Optional[Dict[str, Any]] = None
    ):
        """Print success message with key metrics."""
        print(f"\n✅ {message}")

        if metrics:
            print("\n📊 KEY METRICS:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    def create_progress_bar(
        self,
        current: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 50,
    ) -> str:
        """Create a visual progress bar."""
        percent = (current / total) * 100 if total > 0 else 0
        filled_length = int(length * current / total) if total > 0 else 0

        bar = "█" * filled_length + "░" * (length - filled_length)

        return f"\r{prefix} |{bar}| {percent:.1f}% {suffix}"

    def print_success(self, message: str):
        """Print success message."""
        print(f"✅ {message}")

    def print_error(self, error: str):
        """Print error message."""
        print(f"❌ {error}")

    def print_info(self, message: str):
        """Print info message."""
        print(f"ℹ️ {message}")

    def print_ensemble_status(self, ensemble_stats: Dict[str, Any]):
        """Print ensemble status information."""
        print("\n🤖 ENSEMBLE STATUS:")
        print("-" * 40)

        # Overall stats
        overall = ensemble_stats.get("overall_stats", {})
        print("Overall Statistics:")
        print(f"  Total Members: {overall.get('total_members', 0)}")
        print(f"  Avg Confidence: {overall.get('avg_confidence', 0):.3f}")
        print(f"  Avg Performance: {overall.get('avg_performance', 0):.3f}")
        print(f"  Avg Stability: {overall.get('avg_stability', 0):.3f}")
        print(f"  Decision Log Size: {overall.get('decision_log_size', 0)}")

        # Config
        config = ensemble_stats.get("config", {})
        print(f"  Voting Mechanism: {config.get('voting_mechanism', 'unknown')}")
        print(f"  Diversity Weight: {config.get('diversity_weight', 0):.2f}")
        print(f"  Consensus Enabled: {config.get('consensus_enabled', False)}")

        # Member stats
        member_stats = ensemble_stats.get("member_stats", {})
        if member_stats:
            print("\nMember Details:")
            for member_id, stats in member_stats.items():
                print(
                    f"  {member_id}: {stats.get('specialization', 'unknown')} "
                    f"(conf: {stats.get('confidence', 0):.2f}, "
                    f"perf: {stats.get('performance_score', 0):.2f})"
                )

        print("-" * 40)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate suffix."""
    if abs(num) >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"
