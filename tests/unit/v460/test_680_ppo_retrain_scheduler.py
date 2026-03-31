"""680# PPO retrain scheduler foundation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from scripts.v460.lib.sidecar_signal_io import read_ppo_sidecar_signal
from scripts.v460.lib.sidecar_types import PPOSidecarSignal
from scripts.v460.ml.ppo_retrain_scheduler import (
    PPORetrainResult,
    PPORetrainTrigger,
    _push_neutral_fallback,
    _update_ppo_sidecar_signal,
    load_config,
    retrain_once,
)
from scripts.v460.ml.ppo_sidecar_config import PPOSidecarConfig


class TestPPOSidecarConfig:
    def test_build_trainer_config_flattens_hyperparameters(self) -> None:
        cfg = PPOSidecarConfig(
            data_path="data/ppo.csv",
            ppo_hyperparameters={
                "learning_rate": 1e-4,
                "n_steps": 128,
                "use_custom_ppo": True,
            },
        )

        trainer_config = cfg.build_trainer_config(total_timesteps=12_345)

        assert trainer_config["data_path"] == "data/ppo.csv"
        assert trainer_config["total_timesteps"] == 12_345
        assert trainer_config["learning_rate"] == pytest.approx(1e-4)
        assert trainer_config["n_steps"] == 128
        assert isinstance(trainer_config["ppo"], dict)
        assert trainer_config["ppo"]["use_custom_ppo"] is True

    def test_load_config_parses_scheduler_fields(self, tmp_path: Path) -> None:
        config_path = tmp_path / "ppo.yaml"
        config_path.write_text(
            """
data:
  data_path: data/ppo_training.csv
output:
  model_dir: models/v461
ppo_sidecar:
  retrain_interval_sec: 600
  retrain_interval_max_sec: 1800
  history_path: logs/custom_ppo_history.jsonl
  signal_path: cache/custom_ppo_signal.json
  min_override_confidence: 0.61
  min_action_probability_gap: 0.17
ppo_hyperparameters:
  learning_rate: 0.0002
""".strip(),
            encoding="utf-8",
        )

        cfg = load_config(config_path)

        assert cfg.data_path == "data/ppo_training.csv"
        assert cfg.retrain_interval_sec == 600
        assert cfg.retrain_interval_max_sec == 1800
        assert cfg.history_path == Path("logs/custom_ppo_history.jsonl")
        assert cfg.signal_path == Path("cache/custom_ppo_signal.json")
        assert cfg.min_override_confidence == pytest.approx(0.61)
        assert cfg.min_action_probability_gap == pytest.approx(0.17)
        assert cfg.ppo_hyperparameters["learning_rate"] == pytest.approx(0.0002)


class TestPPORetrainTrigger:
    def _make_trigger(self, tmp_path: Path) -> tuple[PPORetrainTrigger, Path]:
        data_file = tmp_path / "ppo.csv"
        data_file.write_text("timestamp,close\n1,100\n", encoding="utf-8")
        cfg = PPOSidecarConfig(
            data_path=str(data_file),
            check_interval_sec=10,
            retrain_interval_sec=60,
            retrain_interval_max_sec=600,
        )
        return PPORetrainTrigger(cfg), data_file

    def test_first_run_should_retrain(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        should, reason = trigger.should_retrain()
        assert should is True
        assert "data_updated" in reason

    def test_backoff_on_error(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        trigger.record_result("error")
        trigger.record_result("error")
        assert trigger.effective_interval == 240.0


class TestPPORetrainResult:
    def test_to_dict(self) -> None:
        result = PPORetrainResult(
            status="deployed",
            timestamp="2026-04-01T00:00:00+00:00",
            model_version="ppo_sidecar_20260401_0000",
            training_time_sec=12.345,
            total_timesteps=50_000,
            warm_start=True,
            action="buy",
            confidence=0.812345,
            action_margin=0.423456,
        )
        payload = result.to_dict()
        assert payload["status"] == "deployed"
        assert payload["training_time_sec"] == 12.3
        assert payload["confidence"] == pytest.approx(0.812345)
        assert payload["action_margin"] == pytest.approx(0.423456)


class _TinyInferenceEnv:
    def reset(self) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros(4, dtype=np.float32), {}

    def get_action_masks(self) -> NDArray[np.bool_]:
        return np.array([True, True, True], dtype=np.bool_)

    def close(self) -> None:
        return None


class _FakePolicy:
    def obs_to_tensor(self, observation: object) -> tuple[torch.Tensor, None]:
        del observation
        return torch.zeros((1, 4), dtype=torch.float32), None

    def get_distribution(self, observation: object) -> object:
        del observation
        probs = torch.tensor([[0.10, 0.70, 0.20]], dtype=torch.float32)
        return SimpleNamespace(distribution=SimpleNamespace(probs=probs))


class _FakePPOModel:
    def __init__(self) -> None:
        self._policy = _FakePolicy()

    @property
    def policy(self) -> object:
        return self._policy

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"ppo-model")

    def predict(
        self,
        observation: object,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return np.asarray([1], dtype=np.int64), None


class TestPPOSidecarSignalUpdate:
    def test_push_neutral_fallback_writes_skip_signal(self, tmp_path: Path) -> None:
        signal_path = tmp_path / "ppo_signal.json"

        assert _push_neutral_fallback(signal_path) is True

        loaded = read_ppo_sidecar_signal(signal_path)
        assert loaded is not None
        assert loaded.action == "skip"
        assert loaded.action_probabilities["skip"] == pytest.approx(1.0)

    def test_update_sidecar_signal_writes_probabilities(self, tmp_path: Path) -> None:
        cfg = PPOSidecarConfig(
            data_path="data/ppo.csv",
            signal_path=tmp_path / "ppo_signal.json",
        )
        model = _FakePPOModel()

        with patch(
            "scripts.v460.ml.ppo_retrain_scheduler._build_inference_env",
            return_value=_TinyInferenceEnv(),
        ):
            signal_obj = _update_ppo_sidecar_signal(model, cfg, "ppo_v1")

        assert signal_obj.action == "buy"
        assert signal_obj.confidence == pytest.approx(0.70)
        assert signal_obj.action_margin == pytest.approx(0.50)
        assert signal_obj.training_metrics["min_override_confidence"] == pytest.approx(0.55)
        loaded = read_ppo_sidecar_signal(cfg.signal_path)
        assert loaded is not None
        assert loaded.action == "buy"


class TestPPORetrainOnce:
    def test_success_deploys_model_and_updates_signal(self, tmp_path: Path) -> None:
        cfg = PPOSidecarConfig(
            data_path=str(tmp_path / "ppo.csv"),
            model_path=tmp_path / "ppo_sidecar.zip",
            checkpoint_dir=tmp_path / "ckpt",
            signal_path=tmp_path / "ppo_signal.json",
        )
        Path(cfg.data_path).write_text("timestamp,close\n1,100\n", encoding="utf-8")
        fake_model = _FakePPOModel()
        fake_signal = PPOSidecarSignal.from_probabilities(
            timestamp="2026-04-01T00:00:00+00:00",
            action_probabilities={"buy": 0.61, "sell": 0.22, "skip": 0.17},
            model_version="ppo_v1",
        )
        fake_trainer = MagicMock()
        fake_trainer.train.return_value = fake_model

        with (
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler.SELLBiasMitigationPPOTrainer",
                return_value=fake_trainer,
            ),
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._update_ppo_sidecar_signal",
                return_value=fake_signal,
            ) as mock_update_signal,
        ):
            result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.action == "buy"
        assert result.confidence == pytest.approx(0.61)
        assert cfg.model_path.exists()
        mock_update_signal.assert_called_once()

    def test_error_pushes_neutral_fallback(self, tmp_path: Path) -> None:
        cfg = PPOSidecarConfig(
            data_path=str(tmp_path / "ppo.csv"),
            signal_path=tmp_path / "ppo_signal.json",
        )
        Path(cfg.data_path).write_text("timestamp,close\n1,100\n", encoding="utf-8")
        fake_trainer = MagicMock()
        fake_trainer.train.side_effect = RuntimeError("boom")

        with patch(
            "scripts.v460.ml.ppo_retrain_scheduler.SELLBiasMitigationPPOTrainer",
            return_value=fake_trainer,
        ):
            result = retrain_once(cfg)

        assert result.status == "error"
        loaded = read_ppo_sidecar_signal(cfg.signal_path)
        assert loaded is not None
        assert loaded.action == "skip"
