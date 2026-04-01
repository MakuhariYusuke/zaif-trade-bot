#!/usr/bin/env python3
"""Focused PPO integration tests for the current compatibility layer."""
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from sb3_contrib.common.wrappers import ActionMasker
from ztb.training.config.trainer_params import SELLMitigationParams
from ztb.training.experiments.sell_mitigation_ppo_trainer import (
    SELLBiasMitigationPPOTrainer,
)
from ztb.training.models.custom_ppo import CustomPPO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


class _TinyMaskedEnv:
    """Minimal discrete env for PPO trainer integration smoke."""

    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32,
    )

    def reset(self, *_args: object, **_kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros(4, dtype=np.float32), {}

    def step(
        self, _action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        return None

    def get_action_masks(self) -> NDArray[np.bool_]:
        return np.asarray([True, True, True], dtype=np.bool_)

    def get_legal_actions(self) -> NDArray[np.int_]:
        raise AssertionError("legacy get_legal_actions mask path should not be used")


@pytest.fixture(scope="module")
def tiny_masked_env() -> Generator[ActionMasker, None, None]:
    """Fast masked env for PPO compatibility smoke without HeavyTradingEnv setup."""
    env = _TinyMaskedEnv()
    wrapped_env = ActionMasker(env, mask_fn=lambda inner: inner.get_action_masks())
    yield wrapped_env


@pytest.fixture(scope="module")
def simple_df() -> pd.DataFrame:
    """Deterministic OHLCV fixture small enough for fast PPO smoke runs."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=48, freq="1min"),
            "open": [100.0] * 48,
            "high": [101.0] * 48,
            "low": [99.0] * 48,
            "close": [100.5] * 48,
            "volume": [1000.0] * 48,
        }
    )


class TestActionMaskerCompatibility:
    """Validate the local ActionMasker shim against current PPO expectations."""

    def test_action_masker_preserves_env_contract(self, tiny_masked_env: ActionMasker) -> None:
        assert tiny_masked_env.action_space.n == 3
        assert tiny_masked_env.observation_space.shape[0] > 0
        masks = tiny_masked_env.get_action_masks()
        assert masks.dtype.name == "bool"
        assert masks.shape == (3,)

    def test_action_masker_accepts_legacy_keyword(self) -> None:
        env = _TinyMaskedEnv()
        wrapped = ActionMasker(env, action_mask_fn=lambda inner: inner.get_action_masks())
        assert wrapped.action_space.n == 3
        assert wrapped.action_masks().shape == (3,)


class TestCustomPPOIntegration:
    """Current CustomPPO integration smoke tests."""

    def test_create_with_current_masked_env(self, tiny_masked_env: ActionMasker) -> None:
        model = CustomPPO(
            policy="MlpPolicy",
            env=tiny_masked_env,
            n_steps=16,
            batch_size=8,
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

    def test_short_training_run(self, tiny_masked_env: ActionMasker) -> None:
        model = CustomPPO(
            policy="MlpPolicy",
            env=tiny_masked_env,
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            enable_pan=True,
            enable_target_entropy=True,
            enable_stratified_sampling=False,
            verbose=0,
        )

        learned = model.learn(total_timesteps=32, progress_bar=False)

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
                "n_steps": 16,
                "batch_size": 8,
                "n_epochs": 1,
                "total_timesteps": 32,
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
        tiny_env = _TinyMaskedEnv()

        with (
            patch(
                "ztb.training.experiments.sell_mitigation_ppo_trainer.DataLoader.load_csv_strict",
                return_value=simple_df.copy(),
            ),
            patch(
                "ztb.training.experiments.sell_mitigation_ppo_trainer.HeavyTradingEnv",
                return_value=tiny_env,
            ),
            patch.object(CustomPPO, "learn", autospec=True, return_value=None) as mock_learn,
            patch.object(trainer, "_final_validation", return_value=None),
            patch.object(trainer, "start_training", return_value=None),
        ):
            model = trainer.train(session_id="ppo_integration_smoke")

        assert isinstance(model, CustomPPO)
        mock_learn.assert_called_once()
        assert model.enable_pan is True
        assert model.enable_target_entropy is True

    def test_sell_mitigation_load_and_continue_uses_current_warm_start_path(
        self,
        simple_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        data_path = tmp_path / "ppo_smoke.csv"
        simple_df.to_csv(data_path, index=False)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        model_path = checkpoint_dir / "ppo_sidecar.zip"
        model_path.write_bytes(b"fake-model")

        params = SELLMitigationParams(
            data_path=str(data_path),
            config={
                "policy": "MlpPolicy",
                "learning_rate": 3e-4,
                "n_steps": 16,
                "batch_size": 8,
                "n_epochs": 1,
                "total_timesteps": 32,
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
        loaded_model = CustomPPO(
            policy="MlpPolicy",
            env=ActionMasker(_TinyMaskedEnv(), mask_fn=lambda inner: inner.get_action_masks()),
            n_steps=16,
            batch_size=8,
            n_epochs=1,
            verbose=0,
        )

        with (
            patch.object(trainer, "_create_training_env", return_value=ActionMasker(_TinyMaskedEnv(), mask_fn=lambda inner: inner.get_action_masks())),
            patch(
                "ztb.training.experiments.sell_mitigation_ppo_trainer.load_ppo_model_for_env",
                return_value=loaded_model,
            ) as mock_load,
            patch.object(CustomPPO, "learn", autospec=True, return_value=None) as mock_learn,
            patch.object(trainer, "_final_validation", return_value=None),
            patch.object(trainer, "start_training", return_value=None),
        ):
            model = trainer.load_and_continue(
                model_path=model_path,
                total_timesteps=12,
                session_id="ppo_warm_resume",
            )

        assert model is loaded_model
        mock_load.assert_called_once()
        mock_learn.assert_called_once()
