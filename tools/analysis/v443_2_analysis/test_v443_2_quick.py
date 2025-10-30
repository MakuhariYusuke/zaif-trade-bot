#!/usr/bin/env python3
"""
Quick test of v443.2 Phase 3 with Monitor wrapper fix
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def test_v443_2_phase3_quick():
    """Run a quick test of v443.2 Phase 3 to verify reward fix"""

    print("=== Testing v443.2 Phase 3 with Monitor Fix ===")

    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training_log.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Ensure all loggers inherit the root logger configuration
    logging.getLogger().setLevel(logging.INFO)

    config_path = "config/v443_2_phase3_config.json"

    try:
        # Create trainer with config file
        trainer = V4XXUnifiedTrainer(config_path=config_path)

        # Re-add file handler after trainer initialization (which may clear handlers)
        file_handler = logging.FileHandler("training_log.txt", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        # Override timesteps for quick test
        if hasattr(trainer, "config") and "training" in trainer.config:
            trainer.config["training"]["total_timesteps"] = 1000

        # Run training
        print("Starting quick training test...")
        trainer.train()

        print("✅ Training completed successfully")
        logging.info("Training completed successfully")

        # Check if trades_history.csv was created and rewards are not corrupted
        backtest_dir = Path("backtest_results") / "v443_2_phase3"
        if backtest_dir.exists():
            trades_file = backtest_dir / "trades_history.csv"
            if trades_file.exists():
                import pandas as pd

                trades_df = pd.read_csv(trades_file)
                rewards = trades_df["reward"].values

                negative_count = len([r for r in rewards if r < 0])
                positive_count = len([r for r in rewards if r > 0])

                print(
                    f"Rewards in test: {len(rewards)} total, {positive_count} positive, {negative_count} negative"
                )

                if positive_count > 0:
                    print("✅ REWARD FIX VERIFIED - Positive rewards detected!")
                    return True
                else:
                    print("❌ REWARD FIX FAILED - Still no positive rewards")
                    return False
            else:
                print("❌ No trades_history.csv found")
                return False
        else:
            print("❌ No backtest_results/v443_2_phase3 directory found")
            return False

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_v443_2_phase3_quick()
    sys.exit(0 if success else 1)
