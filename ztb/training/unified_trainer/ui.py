#!/usr/bin/env python3
"""
UI and display utilities for Unified Trainer.
"""

import time
from typing import Any, Dict, Optional

from ztb.utils.logging_utils import get_logger


class TrainingUI:
    """Enhanced UI for training progress and statistics display."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.start_time = None

    def print_header(self, algorithm: str, config_name: str):
        """Print training header."""
        print("\n" + "="*80)
        print("🤖 ZAIF TRADE BOT - UNIFIED TRAINER")
        print("="*80)
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Config: {config_name}")
        print("="*80)

    def print_config_summary(self, config: Dict[str, Any]):
        """Print configuration summary."""
        print("\n📋 CONFIGURATION SUMMARY:")
        print("-" * 40)

        # Algorithm info
        algorithm = config.get('algorithm', 'unknown')
        print(f"Algorithm: {algorithm}")

        # Training parameters
        total_timesteps = config.get('total_timesteps', 'unknown')
        print(f"Total Timesteps: {total_timesteps:,}")

        # Data info
        data_path = config.get('data_path', 'unknown')
        print(f"Data Path: {data_path}")

        # Model info
        model_name = config.get('model_name', 'unknown')
        print(f"Model Name: {model_name}")

        # SAC specific config
        if algorithm.lower() == 'sac':
            sac_config = config.get('sac_hyperparameters', {})
            print(f"Learning Rate: {sac_config.get('learning_rate', 'default')}")
            print(f"Buffer Size: {sac_config.get('buffer_size', 'default'):,}")
            print(f"Batch Size: {sac_config.get('batch_size', 'default')}")

        print("-" * 40)

    def start_training(self):
        """Mark training start."""
        self.start_time = time.time()
        print("\n🚀 STARTING TRAINING...")
        print("-" * 40)

    def print_training_complete(self, success: bool, stats: Optional[Dict[str, Any]] = None):
        """Print training completion summary."""
        if self.start_time:
            total_time = time.time() - self.start_time
        else:
            total_time = 0

        print("\n" + "="*80)
        if success:
            print("✅ TRAINING COMPLETED SUCCESSFULLY")
        else:
            print("❌ TRAINING FAILED")
        print("="*80)

        print(f"Total Time: {total_time:.1f} seconds")

        if stats:
            print("\n📊 TRAINING STATISTICS:")
            print("-" * 40)

            for key, value in stats.items():
                if isinstance(value, float):
                    if 'rate' in key.lower() or 'ratio' in key.lower():
                        print(f"{key}: {value:.4f}")
                    elif 'time' in key.lower():
                        print(f"{key}: {value:.1f}s")
                    else:
                        print(f"{key}: {value:.2f}")
                elif isinstance(value, int):
                    print(f"{key}: {value:,}")
                else:
                    print(f"{key}: {value}")

        print("="*80)

    def print_validation_results(self, is_valid: bool, errors: Optional[list] = None):
        """Print configuration validation results."""
        print("\n🔍 CONFIGURATION VALIDATION:")
        print("-" * 40)

        if is_valid:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration validation failed:")
            if errors:
                for error in errors:
                    print(f"  - {error}")

        print("-" * 40)

    def print_error(self, message: str, details: Optional[str] = None):
        """Print error message."""
        print(f"\n❌ ERROR: {message}")
        if details:
            print(f"Details: {details}")

    def print_warning(self, message: str):
        """Print warning message."""
        print(f"\n⚠️  WARNING: {message}")

    def print_info(self, message: str):
        """Print info message."""
        print(f"ℹ️  {message}")

    def print_success(self, message: str):
        """Print success message."""
        print(f"✅ {message}")


class ProgressTracker:
    """Track and display training progress."""

    def __init__(self, total_steps: int, update_interval: int = 1000):
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update = self.start_time

    def update(self, current_step: int, stats: Optional[Dict[str, Any]] = None):
        """Update progress display."""
        current_time = time.time()

        # Only update at specified intervals
        if current_step % self.update_interval != 0 and current_step != self.total_steps:
            return

        elapsed = current_time - self.start_time
        progress = current_step / self.total_steps

        # Calculate ETA
        if progress > 0:
            eta = elapsed / progress - elapsed
        else:
            eta = 0

        # Calculate steps per second
        sps = current_step / elapsed if elapsed > 0 else 0

        # Create progress bar
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print progress
        print(f"\rStep {current_step:6d}/{self.total_steps:6d} |{bar}| "
              f"{progress:5.1%} | "
              f"Elapsed: {elapsed:6.1f}s | "
              f"ETA: {eta:6.1f}s | "
              f"SPS: {sps:5.1f}", end="", flush=True)

        # Print stats if provided
        if stats and current_step % (self.update_interval * 10) == 0:
            print()  # New line for stats
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}", end="")
                else:
                    print(f"  {key}: {value}", end="")
            print()  # New line after stats

        # Final newline when complete
        if current_step >= self.total_steps:
            print()


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