#!/usr/bin/env python3
"""
V443.2 Phase 3: Backtest Optimization Training
Final phase with optimized parameters based on Phase 2 learnings
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    print("🚀 Starting V443.2 Phase 3: Backtest Optimization Training")
    print("📋 Phase 3: 25k steps with backtest-optimized parameters")

    # Load Phase 3 configuration
    config_path = "config/v443_2_phase3_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"⚙️ Loaded configuration: {config_path}")
    print(f"🎯 Curriculum stage: {config.get('curriculum_stage', 'unknown')}")
    print(f"📊 Total timesteps: {config.get('total_timesteps', 0):,}")
    print(
        f"🏛️ Market regime adaptation: {config.get('market_regime', {}).get('enabled', False)}"
    )

    # Initialize trainer
    trainer = V4XXUnifiedTrainer(config_path=config_path)

    # Execute training
    print("\n🏃 Executing Phase 3 training...")
    results = trainer.train()

    # Analyze results
    print("\n📊 Analyzing Phase 3 training results...")
    analysis = trainer.analyze_training_results(results)

    print("\n✅ Phase 3 training completed successfully!")
    print("🎉 V443.2 Backtest Optimization Phase 3 training completed!")
    return results, analysis


if __name__ == "__main__":
    main()
