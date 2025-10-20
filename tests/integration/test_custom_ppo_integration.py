#!/usr/bin/env python3
"""
Integration test for CustomPPO with bias mitigation components.

Verifies that:
1. CustomPPO can be instantiated with all components
2. Components are correctly integrated into the training loop
3. Statistics are logged appropriately
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sb3_contrib.common.wrappers import ActionMasker

from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv
from ztb.training.adv_norm import PerActionAdvantageNormalizer
from ztb.training.custom_ppo import CustomPPO
from ztb.training.entropy_temperature import TargetEntropyController


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Create simple test dataframe."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="1h"),
            "open": 100.0 + np.random.randn(100) * 0.1,
            "high": 100.0 + np.random.randn(100) * 0.1 + 0.5,
            "low": 100.0 + np.random.randn(100) * 0.1 - 0.5,
            "close": 100.0 + np.random.randn(100) * 0.1,
            "volume": 1000.0 + np.random.randn(100) * 100,
        }
    )


@pytest.fixture
def env_config() -> EnvironmentConfig:
    """Create test environment configuration."""
    return EnvironmentConfig(
        commission_rate=0.001,
        slippage_rate=0.0005,
        initial_balance=100000.0,
        position_size_pct=0.1,
        allow_reverse=False,
    )


@pytest.fixture
def test_env(simple_df: pd.DataFrame, env_config: EnvironmentConfig) -> Any:
    """Create test environment with ActionMasker."""
    env = HeavyTradingEnv(df=simple_df, config=env_config)

    def mask_fn(env: Any) -> Any:
        return env.get_legal_actions().astype(bool)

    return ActionMasker(env, mask_fn)


class TestCustomPPOInstantiation:
    """Test CustomPPO instantiation with various configurations."""

    def test_create_with_all_mitigations(self, test_env: Any):
        """Test creating CustomPPO with all bias mitigations enabled."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=False,  # Complex, disabled for now
            verbose=0,
        )

        # Verify components are initialized
        assert model.enable_pan is True
        assert model.enable_target_entropy is True
        assert model.pan_normalizer is not None
        assert model.entropy_controller is not None
        assert isinstance(model.pan_normalizer, PerActionAdvantageNormalizer)
        assert isinstance(model.entropy_controller, TargetEntropyController)

    def test_create_with_pan_only(self, test_env: Any):
        """Test creating CustomPPO with only PAN enabled."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            enable_pan=True,
            enable_target_entropy=False,
            verbose=0,
        )

        assert model.enable_pan is True
        assert model.enable_target_entropy is False
        assert model.pan_normalizer is not None
        assert model.entropy_controller is None

    def test_create_with_target_entropy_only(self, test_env: Any):
        """Test creating CustomPPO with only Target Entropy enabled."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            enable_pan=False,
            enable_target_entropy=True,
            verbose=0,
        )

        assert model.enable_pan is False
        assert model.enable_target_entropy is True
        assert model.pan_normalizer is None
        assert model.entropy_controller is not None

    def test_create_without_mitigations(self, test_env: Any):
        """Test creating CustomPPO with no mitigations (standard PPO)."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            enable_pan=False,
            enable_target_entropy=False,
            enable_stratified_sampling=False,
            verbose=0,
        )

        assert model.enable_pan is False
        assert model.enable_target_entropy is False
        assert model.pan_normalizer is None
        assert model.entropy_controller is None


class TestCustomPPOTraining:
    """Test CustomPPO training with bias mitigations."""

    def test_short_training_run(self, test_env: Any):
        """Test a very short training run to verify integration."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,  # Small rollout
            batch_size=32,  # Small batch
            n_epochs=2,  # Few epochs
            enable_pan=True,
            enable_target_entropy=True,
            verbose=0,
        )

        # Collect initial rollout
        model.learn(total_timesteps=128, progress_bar=False)

        # Verify model trained (n_updates increased)
        assert model._n_updates > 0

    def test_pan_statistics_logging(self, test_env: Any):
        """Test that PAN statistics are logged during training."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            enable_pan=True,
            enable_target_entropy=False,
            verbose=0,
        )

        # Train briefly
        model.learn(total_timesteps=128, progress_bar=False)

        # Check PAN was used
        assert model.pan_normalizer is not None
        pan_stats = model.pan_normalizer.get_statistics()
        assert pan_stats["total_samples"] > 0, "PAN should have processed samples"

    def test_entropy_controller_updates(self, test_env: Any):
        """Test that Target Entropy Controller updates temperature."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            enable_pan=False,
            enable_target_entropy=True,
            verbose=0,
        )

        # Get initial alpha
        initial_alpha = model.entropy_controller.get_current_alpha()

        # Train briefly
        model.learn(total_timesteps=128, progress_bar=False)

        # Check controller was updated
        entropy_stats = model.entropy_controller.get_statistics()
        assert (
            entropy_stats["num_updates"] > 0
        ), "Entropy controller should have updated"

        # Alpha may or may not change depending on entropy, but updates should occur
        final_alpha = model.entropy_controller.get_current_alpha()
        # Just verify alpha is a valid number
        assert 0.0 <= final_alpha <= 1.0

    def test_both_components_active(self, test_env: Any):
        """Test that both PAN and Target Entropy work together."""
        model = CustomPPO(
            policy="MlpPolicy",
            env=test_env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            enable_pan=True,
            enable_target_entropy=True,
            verbose=0,
        )

        # Train briefly
        model.learn(total_timesteps=128, progress_bar=False)

        # Both components should be active
        pan_stats = model.pan_normalizer.get_statistics()
        entropy_stats = model.entropy_controller.get_statistics()

        assert pan_stats["total_samples"] > 0, "PAN should be active"
        assert entropy_stats["num_updates"] > 0, "Entropy controller should be active"


class TestCustomPPOWithSELLMitigationTrainer:
    """Test CustomPPO integration with SELLBiasMitigationPPOTrainer."""

    def test_trainer_uses_custom_ppo(self, simple_df: pd.DataFrame, tmp_path: Path):
        """Test that SELLBiasMitigationPPOTrainer creates CustomPPO."""
        from ztb.training.sell_mitigation_ppo_trainer import (
            SELLBiasMitigationPPOTrainer,
        )

        # Create config
        config = {
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "initial_balance": 100000.0,
            "position_size_pct": 0.1,
            "allow_reverse": False,
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "total_timesteps": 128,
            "verbose": 0,
        }

        # Save test data
        data_path = tmp_path / "test_data.csv"
        simple_df.to_csv(data_path, index=False)

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create trainer
        trainer = SELLBiasMitigationPPOTrainer(
            data_path=str(data_path),
            config=config,
            checkpoint_dir=str(checkpoint_dir),
            enable_lagrange=False,  # Disable Lagrange for this test
            enable_probes=False,  # Disable probes for this test
            enable_weights=False,  # Disable weights for this test
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=False,
        )

        # Train
        model = trainer.train(session_id="test_custom_ppo")

        # Verify CustomPPO was used
        assert isinstance(model, CustomPPO)
        assert model.enable_pan is True
        assert model.enable_target_entropy is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
