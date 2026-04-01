"""Focused PPO warm-start contract tests."""
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ztb.training.config.trainer_params import TrainerParams
from ztb.training.core.ppo_trainer import PPOTrainerAutoHalt


class _WarmStartModel:
    def __init__(self) -> None:
        self.env: object | None = None
        self.learn_calls: list[int] = []
        self.reset_num_timesteps_calls: list[bool] = []
        self.saved_paths: list[str] = []

    def set_env(self, env: object) -> None:
        self.env = env

    def learn(
        self,
        total_timesteps: int,
        callback: object,
        tb_log_name: str,
        progress_bar: bool,
        reset_num_timesteps: bool = True,
    ) -> "_WarmStartModel":
        del callback, tb_log_name, progress_bar
        self.learn_calls.append(total_timesteps)
        self.reset_num_timesteps_calls.append(reset_num_timesteps)
        return self

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"ppo-model")
        self.saved_paths.append(path)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=16, freq="1min"),
            "open": [100.0] * 16,
            "high": [101.0] * 16,
            "low": [99.0] * 16,
            "close": [100.5] * 16,
            "volume": [1000.0] * 16,
        }
    )


@pytest.fixture
def trainer_params(tmp_path: Path) -> TrainerParams:
    return TrainerParams(
        data_path=str(tmp_path / "ppo.csv"),
        config={
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 16,
                "batch_size": 8,
                "n_epochs": 1,
                "total_timesteps": 32,
                "use_custom_ppo": True,
            },
            "eval_gates_enabled": False,
        },
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


class TestPPOWarmStart:
    def test_load_and_continue_roundtrip(self, trainer_params: TrainerParams) -> None:
        trainer = PPOTrainerAutoHalt(trainer_params)
        cold_model = _WarmStartModel()
        warm_model = _WarmStartModel()
        model_path = Path(trainer_params.checkpoint_dir) / "ppo_sidecar.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            patch.object(trainer, "_create_environment", return_value=MagicMock()),
            patch.object(trainer, "_create_model", return_value=cold_model),
            patch.object(trainer, "_create_callback", return_value=MagicMock()),
            patch(
                "ztb.training.core.ppo_trainer.MaskablePPO",
                MagicMock(load=MagicMock(return_value=warm_model)),
            ),
        ):
            trained = trainer.train(session_id="cold_start")
            assert trained is cold_model
            assert cold_model.learn_calls == [32]
            assert cold_model.reset_num_timesteps_calls == [True]

            cold_model.save(str(model_path))
            warmed = trainer.load_and_continue(
                model_path=model_path,
                total_timesteps=12,
                session_id="warm_start",
            )

        assert warmed is warm_model
        assert warm_model.env is not None
        assert warm_model.learn_calls == [12]
        assert warm_model.reset_num_timesteps_calls == [False]
        warm_model.save(str(model_path))
        assert model_path.exists()

    def test_missing_model_falls_back_to_cold_start(
        self,
        trainer_params: TrainerParams,
    ) -> None:
        trainer = PPOTrainerAutoHalt(trainer_params)
        fallback_model = _WarmStartModel()

        with patch.object(trainer, "train", return_value=fallback_model) as mock_train:
            loaded = trainer.load_and_continue(
                model_path=Path(trainer_params.checkpoint_dir) / "missing.zip",
                total_timesteps=12,
                session_id="fallback",
            )

        assert loaded is fallback_model
        mock_train.assert_called_once_with(session_id="fallback")

    def test_load_failure_falls_back_to_cold_start(
        self,
        trainer_params: TrainerParams,
    ) -> None:
        trainer = PPOTrainerAutoHalt(trainer_params)
        model_path = Path(trainer_params.checkpoint_dir) / "broken.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"broken")
        fallback_model = _WarmStartModel()

        with (
            patch.object(trainer, "_create_environment", return_value=MagicMock()),
            patch(
                "ztb.training.core.ppo_trainer.MaskablePPO",
                MagicMock(load=MagicMock(side_effect=RuntimeError("load failed"))),
            ),
            patch.object(trainer, "train", return_value=fallback_model) as mock_train,
        ):
            loaded = trainer.load_and_continue(
                model_path=model_path,
                total_timesteps=12,
                session_id="fallback_after_error",
            )

        assert loaded is fallback_model
        mock_train.assert_called_once_with(session_id="fallback_after_error")
