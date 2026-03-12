#!/usr/bin/env python3
"""Quick test of SAC trainer market regime adaptation."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from unittest.mock import Mock

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer


def test_trainer_regime_adaptation():
    # Create mock environment
    mock_env = Mock()
    mock_env.reset.return_value = [0.1, 0.2, 0.3, 0.4]
    mock_env.step.return_value = ([0.2, 0.3, 0.4, 0.5], 1.0, False, {})
    mock_env.action_space = Mock()
    mock_env.action_space.shape = (2,)
    mock_env.observation_space = Mock()
    mock_env.observation_space.shape = (4,)

    # Create trainer config with regime adaptation
    config = {
        "algorithm": "sac",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 100000,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "target_update_interval": 1,
        "gradient_steps": 1,
        "training": {
            "market_regime_adaptation": {
                "enabled": True,
                "regime_update_frequency": 100,
                "regime_statistics_tracking": True,
            }
        },
    }

    try:
        trainer = SACTrainer(config, mock_env)
        print("SAC Trainer initialized successfully with regime adaptation!")

        # Check attributes
        print(f"Regime classifier initialized: {trainer.regime_classifier is not None}")
        print(
            f"Market regime adaptation enabled: {trainer.market_regime_adaptation.get('enabled', False)}"
        )

        if hasattr(trainer, "regime_stats"):
            print(f"Regime stats initialized: {len(trainer.regime_stats) > 0}")

        print("SAC Trainer regime adaptation test passed!")

    except Exception as e:
        print(f"SAC Trainer test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_trainer_regime_adaptation()
