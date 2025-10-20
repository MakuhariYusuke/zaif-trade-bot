#!/usr/bin/env python3
"""
Test script for UnifiedTrainer with optimization features.
"""

import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.training.unified_trainer.trainer import UnifiedTrainer


def main():
    """Run a quick test training with optimization features."""

    # Load test configuration
    with open("test_sac_config.json", "r") as f:
        config = json.load(f)

    print("🚀 Testing UnifiedTrainer with optimization features...")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Total timesteps: {config['total_timesteps']}")

    # Create trainer with optimization enabled
    trainer = UnifiedTrainer(config=config, force=True, dry_run=False)  # Skip prompts

    # Run training
    success = trainer.run()

    if success:
        print("✅ Training completed successfully!")
        print("📊 Training stats:", trainer.get_training_stats())

        # Check optimization metrics
        if "optimization" in trainer.training_stats:
            opt_metrics = trainer.training_stats["optimization"]
            print("🔧 Optimization metrics:")
            print(f"  - Memory stats: {opt_metrics.get('memory_stats', 'N/A')}")
            print(
                f"  - Performance profile: {opt_metrics.get('performance_profile', 'N/A')}"
            )
            print(
                f"  - Parallel processing: {opt_metrics.get('parallel_processing_enabled', False)}"
            )
            print(f"  - Cache size: {opt_metrics.get('cache_size', 0)}")
            print(
                f"  - Data optimization applied: {opt_metrics.get('data_optimization_applied', False)}"
            )
    else:
        print("❌ Training failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
