#!/usr/bin/env python3
"""
SAC V443.2 Regime Specialization Training Script

Phase 2: Market Regime Specialization
Objective: Implement adaptive behavior based on market conditions
Enhancements:
- Market regime detection (trending, ranging, volatile)
- Regime-adaptive behavior optimization
- Enhanced risk management with position sizing
- Advanced feature engineering

This script trains PPO with regime-aware behavior optimization.
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
    """Main training function for V443.2 regime specialization."""

    # Configuration
    config_path = "config/sac_v443_2_regime_specialization_config.json"
    experiment_name = "v443_2_regime_specialization"

    print("=== SAC V443.2 Regime Specialization Training ===")
    print(f"Configuration: {config_path}")
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

    # Validate market_regime section
    env_config = config.get("training", {}).get("environment", {}).get("config", {})
    market_regime = env_config.get("market_regime", {})
    if market_regime.get("enabled"):
        print("✓ Market regime specialization enabled:")
        print(
            f"  - Regime detection window: {market_regime.get('regime_detection_window', 'N/A')}"
        )
        print(f"  - Trend threshold: {market_regime.get('trend_threshold', 'N/A')}")
        print(
            f"  - Volatility threshold: {market_regime.get('volatility_threshold', 'N/A')}"
        )
        print()

    # Validate risk_management section
    risk_mgmt = env_config.get("risk_management", {})
    if risk_mgmt.get("enabled"):
        print("✓ Risk management enabled:")
        print(f"  - Max drawdown limit: {risk_mgmt.get('max_drawdown_limit', 'N/A')}")
        print(
            f"  - Stop loss enabled: {risk_mgmt.get('stop_loss', {}).get('enabled', False)}"
        )
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
    print("=== V443.2 Regime Specialization Complete ===")
    print("Next steps:")
    print("1. Analyze training results")
    print("2. Compare action distribution with V443.1 baseline")
    print("3. Evaluate regime-adaptive behavior")
    print("4. Validate risk management improvements")
    print("5. Proceed to Phase 3 if enhancements successful")


if __name__ == "__main__":
    main()
