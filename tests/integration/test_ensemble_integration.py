#!/usr/bin/env python3
"""
Test ensemble integration for SAC and PPO trainers.
"""

import json

import numpy as np

from ztb.training.core.config_manager import ConfigManager
from ztb.training.trainers.ppo_trainer import PPOAlgorithmTrainer
from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer


def test_sac_ensemble_integration():
    """Test SAC trainer ensemble integration."""
    print("Testing SAC trainer ensemble integration...")

    # Load config
    with open("configs/sac_v428_ensemble_test.json", "r") as f:
        config = json.load(f)

    # Create trainer
    config_manager = ConfigManager(config)
    trainer = SACAlgorithmTrainer(config_manager)

    # Test ensemble initialization
    trainer.initialize_ensemble(config)
    print(f"  Ensemble enabled: {trainer.ensemble_enabled}")
    print(f"  Ensemble system created: {trainer.ensemble_system is not None}")
    print(f"  Ensemble config created: {trainer.ensemble_config is not None}")

    # Test ensemble prediction (mock observation)
    obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = trainer.predict_with_ensemble(obs)
    print(f"  Ensemble prediction result: {result}")

    print("SAC trainer ensemble integration test passed!")
    return True


def test_ppo_ensemble_integration():
    """Test PPO trainer ensemble integration."""
    print("\nTesting PPO trainer ensemble integration...")

    # Load config
    with open("configs/ppo_v428_ensemble_test.json", "r") as f:
        config = json.load(f)

    # Create trainer
    config_manager = ConfigManager(config)
    trainer = PPOAlgorithmTrainer(config_manager)

    # Test ensemble initialization
    trainer.initialize_ensemble(config)
    print(f"  Ensemble enabled: {trainer.ensemble_enabled}")
    print(f"  Ensemble system created: {trainer.ensemble_system is not None}")
    print(f"  Ensemble config created: {trainer.ensemble_config is not None}")

    # Test ensemble prediction (mock observation)
    obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = trainer.predict_with_ensemble(obs)
    print(f"  Ensemble prediction result: {result}")

    print("PPO trainer ensemble integration test passed!")
    return True


if __name__ == "__main__":
    try:
        test_sac_ensemble_integration()
        test_ppo_ensemble_integration()
        print("\n🎉 All ensemble integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
