#!/usr/bin/env python3
"""
SAC v442.10 Further Balance Refinement Training Script

Further refinement of balance parameters to reduce SELL bias:
- Lower consistency_penalty from 0.04 to 0.02
- Increase entropy_regularization from 0.01 to 0.02
- Adjust action_balance_target from 0.6 to 0.55

Previous v442.9 results: HOLD: 7.2%, BUY: 6.1%, SELL: 86.7%
Target: Reduce SELL bias through more aggressive entropy regularization and reduced consistency penalty
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer


def main():
    config_path = "config/sac_v442_10_further_balance_refinement_config.json"

    print("🚀 Starting SAC v442.10 Further Balance Refinement Training")
    print("Target: Reduce SELL bias through refined behavior optimization parameters")
    print(f"Config: {config_path}")

    trainer = V4XXUnifiedTrainer(config_path=config_path)
    trainer.train()

    print("✅ SAC v442.10 training completed successfully!")


if __name__ == "__main__":
    main()
