#!/usr/bin/env python3
"""
Train SAC v415 model with balanced trading reward (no BUY bias).
"""

import json
import os
from pathlib import Path

from ztb.training.algorithms.sac.sac_algorithm_trainer import SACAlgorithmTrainer


def main() -> None:
    """Train SAC v415 model."""
    config_path = Path("config/sac_v414_balanced_trading_config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update model name
    config["model_name"] = "sac_v415_balanced_trading"

    trainer = SACAlgorithmTrainer(config)
    trainer.train()

    print("SAC v415 training completed!")


if __name__ == "__main__":
    main()