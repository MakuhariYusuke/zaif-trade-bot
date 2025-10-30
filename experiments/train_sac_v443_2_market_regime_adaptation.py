#!/usr/bin/env python3
"""
V443.2 Market Regime Adaptation Training Script

This script trains a PPO model with market regime detection and adaptive behavior optimization.
The model learns to adapt its trading strategy based on detected market conditions.

Key Features:
- Market regime detection (bull, bear, sideways, volatile)
- Regime-adaptive behavior optimization parameters
- Enhanced reward shaping based on market conditions
- Stability-optimized curriculum stage
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer

    print("✓ Successfully imported V4XXUnifiedTrainer")
except ImportError as e:
    print(f"✗ Failed to import V4XXUnifiedTrainer: {e}")
    sys.exit(1)


def main():
    """Main training function for v443.2 market regime adaptation."""

    # Configuration path
    config_path = "config/sac_v443_2_market_regime_adaptation_config.json"

    if not os.path.exists(config_path):
        print(f"✗ Configuration file not found: {config_path}")
        sys.exit(1)

    print("🚀 Starting V443.2 Market Regime Adaptation Training")
    print(f"📋 Configuration: {config_path}")

    try:
        # Load and validate configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        print("✓ Configuration loaded successfully")

        # Validate required sections
        required_sections = ["algorithm", "training", "behavior_optimization"]
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing required section: {section}")
                sys.exit(1)

        print("✓ Configuration validation passed")

        # Initialize trainer
        print("🔧 Initializing V4XXUnifiedTrainer...")
        trainer = V4XXUnifiedTrainer(config_path=config_path)

        # Execute training
        print("🎯 Starting training execution...")
        trainer.train()

        print("✅ Training completed successfully!")

        # Analyze results
        print("📊 Analyzing training results...")
        trainer.analyze_results()

        print("🎉 V443.2 Market Regime Adaptation training completed!")

    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
