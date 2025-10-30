#!/usr/bin/env python3
"""
SAC V443.1 Baseline Verification Training Script

Phase 1: V441 Baseline Verification
Objective: Replicate V441 results with identical configuration
Success Criteria: V441 reproduction within 5% variance

This script trains SAC with V441's proven balance mechanism to establish
a stable baseline for V443 incremental enhancements.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    """Main training function for V443.1 baseline verification."""

    # Configuration
    config_path = "config/sac_v443_1_baseline_verification_config.json"
    experiment_name = "v443_1_baseline_verification"

    print("=== SAC V443.1 Baseline Verification Training ===")
    print(f"Config: {config_path}")
    print(f"Experiment: {experiment_name}")
    print()

    # Load and validate configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return

    # Validate behavior_optimization section
    if "behavior_optimization" not in config:
        print("✗ Missing behavior_optimization section")
        return

    behavior_opt = config["behavior_optimization"]
    print("✓ Behavior optimization parameters:")
    print(
        f"  - action_balance_target: {behavior_opt.get('action_balance_target', 'N/A')}"
    )
    print(
        f"  - entropy_regularization: {behavior_opt.get('entropy_regularization', 'N/A')}"
    )
    print(f"  - action_smoothing: {behavior_opt.get('action_smoothing', 'N/A')}")
    print(f"  - consistency_penalty: {behavior_opt.get('consistency_penalty', 'N/A')}")
    print(f"  - balance_penalty: {behavior_opt.get('balance_penalty', 'N/A')}")
    print()

    # Initialize trainer
    try:
        trainer = V4XXUnifiedTrainer(config_path=config_path)
        print("✓ Trainer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize trainer: {e}")
        return

    # Start training
    try:
        print("Starting training...")
        trainer.train()
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return

    print()
    print("=== V443.1 Baseline Verification Complete ===")
    print("Next steps:")
    print("1. Analyze training results")
    print(
        "2. Compare action distribution with V441 (target: HOLD 32%, BUY 34%, SELL 34%)"
    )
    print("3. Validate performance metrics within 5% of V441 baseline")
    print("4. Proceed to Phase 2 if baseline verified, or debug if not")


if __name__ == "__main__":
    main()
