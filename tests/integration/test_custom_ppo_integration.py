#!/usr/bin/env python3
"""Focused PPO integration tests for the current compatibility layer."""
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sb3_contrib.common.wrappers import ActionMasker
from tests.helpers.environment import make_schema_feature_env_config
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.config.trainer_params import SELLMitigationParams
from ztb.training.experiments.sell_mitigation_ppo_trainer import (
    SELLBiasMitigationPPOTrainer,
)
from ztb.training.models.custom_ppo import CustomPPO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def simple_df() -> pd.DataFrame:
    """Deterministic OHLCV fixture small enough for fast PPO smoke runs."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=96, freq="1min"),
            "open": [100.0] * 96,
            "high": [101.0] * 96,
            "low": [99.0] * 96,
            "close": [100.5] * 96,
            "volume": [1000.0] * 96,
        }
    )


@pytest.fixture
def masked_env(simple_df: pd.DataFrame) -> Generator[ActionMasker, None, None]:
    """Gymnasium-compatible masked env using the current discrete PPO path."""
    env = HeavyTradingEnv(
        df=simple_df.copy(),
        config=make_schema_feature_env_config(
            simple_df,
            use_continuous_actions=False,
        ),
    )
    wrapped_env = ActionMasker(env, mask_fn=lambda inner: inner.get_action_masks())
    try:
        yield wrapped_env
    finally:
        env.close()


class TestActionMaskerCompatibility:
    """Validate the local ActionMasker shim against current PPO expectations."""

    def test_action_masker_preserves_env_contract(self, masked_env: ActionMasker) -> None:
        assert masked_env.action_space.n == 3
        assert masked_env.observation_space.shape[0] > 0
        masks = masked_env.get_action_masks()
        assert masks.dtype.name == "bool"
        assert masks.shape == (3,)

    def test_action_masker_accepts_legacy_keyword(self, simple_df: pd.DataFrame) -> None:
        env = HeavyTradingEnv(
            df=simple_df.copy(),
            config=make_schema_feature_env_config(
                simple_df,
                use_continuous_actions=False,
            ),
        )
        try:
            wrapped = ActionMasker(env, action_mask_fn=lambda inner: inner.get_action_masks())
            assert wrapped.action_space.n == 3
            assert wrapped.action_masks().shape == (3,)
        finally:
            env.close()


class TestCustomPPOIntegration:
    """Current CustomPPO integration smoke tests."""

    def test_create_with_current_masked_env(self, masked_env: ActionMasker) -> None:
        model = CustomPPO(
            policy="MlpPolicy",
            env=masked_env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=False,
            verbose=0,
        )

        assert model.enable_pan is True
        assert model.enable_target_entropy is True
        assert model.pan_normalizer is not None
        assert model.entropy_controller is not None

    def test_short_training_run(self, masked_env: ActionMasker) -> None:
        model = CustomPPO(
            policy="MlpPolicy",
            env=masked_env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=False,
            verbose=0,
        )

        learned = model.learn(total_timesteps=64, progress_bar=False)

        assert learned is model
        assert model.pan_normalizer is not None
        assert model.entropy_controller is not None


class TestSellMitigationTrainerIntegration:
    """SELL mitigation trainer should still create a current CustomPPO model."""

    def test_trainer_uses_current_params_interface(
        self,
        simple_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        data_path = tmp_path / "ppo_smoke.csv"
        simple_df.to_csv(data_path, index=False)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        params = SELLMitigationParams(
            data_path=str(data_path),
            config={
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "n_steps": 32,
                "batch_size": 16,
                "n_epochs": 1,
                "total_timesteps": 64,
                "verbose": 0,
                "use_continuous_actions": False,
            },
            checkpoint_dir=str(checkpoint_dir),
            enable_lagrange=False,
            enable_probes=False,
            enable_weights=False,
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=False,
        )
        trainer = SELLBiasMitigationPPOTrainer(params)

        with (
            patch.object(CustomPPO, "learn", autospec=True, return_value=None) as mock_learn,
            patch.object(trainer, "_final_validation", return_value=None),
            patch.object(trainer, "start_training", return_value=None),
        ):
            model = trainer.train(session_id="ppo_integration_smoke")

        assert isinstance(model, CustomPPO)
        mock_learn.assert_called_once()
        assert model.enable_pan is True
        assert model.enable_target_entropy is True
