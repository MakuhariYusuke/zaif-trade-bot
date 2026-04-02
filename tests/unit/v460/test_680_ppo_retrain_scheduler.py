"""680# PPO retrain scheduler foundation tests."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Iterator
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
    _TRAINING_TIMEOUT_SEC,
    _shutdown_event,
    _extract_action_probabilities,
    _one_hot_ppo_probabilities,
    _push_neutral_fallback,
    _train_with_timeout,
    _update_ppo_sidecar_signal,
    load_config,
    retrain_once,
    run_scheduler,
)
from scripts.v460.ml.ppo_sidecar_config import PPOSidecarConfig
from tests.unit.v460._sidecar_scheduler_test_helpers import (
    make_shutdown_wait,
    patch_noop_paths,
)


@contextmanager
def _patch_scheduler_runtime_overheads() -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._build_trainer_params",
                return_value=object(),
            )
        )
        stack.enter_context(
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler.current_iso_timestamp",
                return_value="2026-04-01T00:00:00+00:00",
            )
        )
        stack.enter_context(
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler.current_compact_timestamp",
                return_value="20260401_0000",
            )
        )
        stack.enter_context(
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._cleanup_training_cycle",
                return_value=None,
            )
        )
        stack.enter_context(
            patch_noop_paths(
                "scripts.v460.ml.ppo_retrain_scheduler.append_history_best_effort",
                "scripts.v460.ml.ppo_retrain_scheduler.record_trigger_result_best_effort",
                "scripts.v460.ml.ppo_retrain_scheduler.logger.error",
                "scripts.v460.ml.ppo_retrain_scheduler.logger.warning",
                "scripts.v460.ml.ppo_retrain_scheduler.logger.info",
            )
        )
        yield


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

    def test_deployed_resets_backoff(self, tmp_path: Path) -> None:
        trigger, _ = self._make_trigger(tmp_path)
        trigger.record_result("error")
        trigger.record_result("error")

        trigger.record_result("deployed")

        assert trigger.effective_interval == 60.0

    def test_time_forced_retrain_when_mtime_is_unchanged(self, tmp_path: Path) -> None:
        trigger, data_file = self._make_trigger(tmp_path)
        trigger.record_result("deployed")
        trigger._last_data_mtime = data_file.stat().st_mtime
        trigger._last_retrain_time = time.time() - (trigger.effective_interval * 3.0 + 1.0)

        should, reason = trigger.should_retrain()

        assert should is True
        assert "time_forced" in reason


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


class _PolicylessPPOModel(_FakePPOModel):
    @property
    def policy(self) -> object:
        return None


class _LogitOnlyPolicy(_FakePolicy):
    def get_distribution(self, observation: object) -> object:
        del observation
        logits = torch.tensor([[0.2, 1.0, -0.2]], dtype=torch.float32)
        return SimpleNamespace(distribution=SimpleNamespace(logits=logits))


class _NoProbPolicy(_FakePolicy):
    def get_distribution(self, observation: object) -> object:
        del observation
        return SimpleNamespace(distribution=SimpleNamespace())


class _NoProbPPOModel(_FakePPOModel):
    def __init__(self) -> None:
        self._policy = _NoProbPolicy()


class _LogitOnlyPPOModel(_FakePPOModel):
    def __init__(self) -> None:
        self._policy = _LogitOnlyPolicy()


class TestPPOProbabilityHelpers:
    def test_one_hot_probabilities_clamp_out_of_range(self) -> None:
        assert _one_hot_ppo_probabilities(-1) == {
            "skip": 1.0,
            "buy": 0.0,
            "sell": 0.0,
        }
        assert _one_hot_ppo_probabilities(9) == {
            "skip": 1.0,
            "buy": 0.0,
            "sell": 0.0,
        }

    def test_extract_probabilities_falls_back_when_policy_is_missing(self) -> None:
        probs = _extract_action_probabilities(
            _PolicylessPPOModel(),
            observation=np.zeros(4, dtype=np.float32),
        )

        assert probs == {"skip": 0.0, "buy": 1.0, "sell": 0.0}

    def test_extract_probabilities_falls_back_when_probs_are_missing(self) -> None:
        probs = _extract_action_probabilities(
            _NoProbPPOModel(),
            observation=np.zeros(4, dtype=np.float32),
        )

        assert probs == {"skip": 0.0, "buy": 1.0, "sell": 0.0}

    def test_extract_probabilities_uses_logits_when_probs_absent(self) -> None:
        probs = _extract_action_probabilities(
            _LogitOnlyPPOModel(),
            observation=np.zeros(4, dtype=np.float32),
            action_masks=np.array([True, True, True], dtype=np.bool_),
        )

        assert probs["buy"] > probs["skip"]
        assert probs["buy"] > probs["sell"]


class TestPPONeutralFallback:
    def test_push_neutral_fallback_writes_skip_signal(self, tmp_path: Path) -> None:
        signal_path = tmp_path / "ppo_signal.json"

        assert _push_neutral_fallback(signal_path) is True

        loaded = read_ppo_sidecar_signal(signal_path)
        assert loaded is not None
        assert loaded.action == "skip"
        assert loaded.action_probabilities["skip"] == pytest.approx(1.0)

    def test_push_neutral_fallback_write_failure_is_suppressed(
        self, tmp_path: Path
    ) -> None:
        signal_path = tmp_path / "ppo_signal.json"
        with patch(
            "scripts.v460.ml.ppo_retrain_scheduler.write_ppo_sidecar_signal",
            side_effect=PermissionError("locked"),
        ):
            assert _push_neutral_fallback(signal_path) is False


class TestPPOSidecarSignalUpdate:
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
            _patch_scheduler_runtime_overheads(),
        ):
            result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.action == "buy"
        assert result.confidence == pytest.approx(0.61)
        assert cfg.model_path.exists()
        mock_update_signal.assert_called_once()

    def test_warm_start_prefers_load_and_continue(self, tmp_path: Path) -> None:
        cfg = PPOSidecarConfig(
            data_path=str(tmp_path / "ppo.csv"),
            model_path=tmp_path / "ppo_sidecar.zip",
            checkpoint_dir=tmp_path / "ckpt",
            signal_path=tmp_path / "ppo_signal.json",
        )
        Path(cfg.data_path).write_text("timestamp,close\n1,100\n", encoding="utf-8")
        cfg.model_path.write_bytes(b"existing-model")
        fake_model = _FakePPOModel()
        fake_signal = PPOSidecarSignal.from_probabilities(
            timestamp="2026-04-01T00:00:00+00:00",
            action_probabilities={"buy": 0.66, "sell": 0.20, "skip": 0.14},
            model_version="ppo_v2",
        )
        fake_trainer = MagicMock()
        fake_trainer.load_and_continue.return_value = fake_model

        with (
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler.SELLBiasMitigationPPOTrainer",
                return_value=fake_trainer,
            ),
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._update_ppo_sidecar_signal",
                return_value=fake_signal,
            ),
            _patch_scheduler_runtime_overheads(),
        ):
            result = retrain_once(cfg)

        assert result.status == "deployed"
        assert result.warm_start is True
        fake_trainer.load_and_continue.assert_called_once()
        fake_trainer.train.assert_not_called()

    def test_error_pushes_neutral_fallback(self, tmp_path: Path) -> None:
        cfg = PPOSidecarConfig(
            data_path=str(tmp_path / "ppo.csv"),
            signal_path=tmp_path / "ppo_signal.json",
        )
        Path(cfg.data_path).write_text("timestamp,close\n1,100\n", encoding="utf-8")
        fake_trainer = MagicMock()

        with (
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler.SELLBiasMitigationPPOTrainer",
                return_value=fake_trainer,
            ),
            patch("scripts.v460.ml.ppo_retrain_scheduler._train_with_timeout", side_effect=RuntimeError("boom")),
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._push_neutral_fallback",
                return_value=True,
            ) as mock_fallback,
            _patch_scheduler_runtime_overheads(),
        ):
            result = retrain_once(cfg)

        assert result.status == "error"
        mock_fallback.assert_called_once_with(cfg.signal_path)


class TestPPORunScheduler:
    @patch("scripts.v460.ml.ppo_retrain_scheduler._install_signal_handlers")
    @patch("scripts.v460.ml.ppo_retrain_scheduler.retrain_once")
    def test_single_iteration_then_shutdown(
        self,
        mock_retrain: MagicMock,
        _mock_signals: MagicMock,
        tmp_path: Path,
    ) -> None:
        data_file = tmp_path / "ppo.csv"
        data_file.write_text("timestamp,close\n1,100\n", encoding="utf-8")
        cfg = PPOSidecarConfig(
            data_path=str(data_file),
            check_interval_sec=1,
            retrain_interval_sec=1,
            history_path=tmp_path / "history.jsonl",
        )
        mock_retrain.return_value = PPORetrainResult(status="deployed", action="buy")

        _shutdown_event.clear()
        with (
            patch.object(_shutdown_event, "wait", side_effect=make_shutdown_wait(shutdown_event=_shutdown_event)),
            patch.object(_shutdown_event, "is_set", side_effect=[False, False, True]),
            patch.object(
                PPORetrainTrigger,
                "should_retrain",
                side_effect=[(True, "manual"), (False, "done")],
            ),
        ):
            run_scheduler(cfg)

        mock_retrain.assert_called_once()
        _shutdown_event.clear()

    @patch("scripts.v460.ml.ppo_retrain_scheduler._install_signal_handlers")
    @patch("scripts.v460.ml.ppo_retrain_scheduler.retrain_once")
    def test_crash_resilience(
        self,
        mock_retrain: MagicMock,
        _mock_signals: MagicMock,
        tmp_path: Path,
    ) -> None:
        data_file = tmp_path / "ppo.csv"
        data_file.write_text("timestamp,close\n1,100\n", encoding="utf-8")
        cfg = PPOSidecarConfig(
            data_path=str(data_file),
            check_interval_sec=1,
            retrain_interval_sec=1,
            signal_path=tmp_path / "ppo_signal.json",
            history_path=tmp_path / "history.jsonl",
        )
        mock_retrain.side_effect = [
            RuntimeError("boom"),
            PPORetrainResult(status="deployed", action="buy"),
        ]

        _shutdown_event.clear()
        with (
            patch.object(_shutdown_event, "wait", side_effect=make_shutdown_wait(set_after=3, shutdown_event=_shutdown_event)),
            patch.object(_shutdown_event, "is_set", side_effect=[False, False, False, True]),
            patch.object(
                PPORetrainTrigger,
                "should_retrain",
                side_effect=[(True, "first"), (True, "second"), (False, "done")],
            ),
        ):
            run_scheduler(cfg)

        assert mock_retrain.call_count >= 2
        loaded = read_ppo_sidecar_signal(cfg.signal_path)
        assert loaded is not None
        assert loaded.action == "skip"
        _shutdown_event.clear()

    @patch("scripts.v460.ml.ppo_retrain_scheduler._install_signal_handlers")
    @patch("scripts.v460.ml.ppo_retrain_scheduler.retrain_once")
    def test_record_result_exception_does_not_kill_loop(
        self,
        mock_retrain: MagicMock,
        _mock_signals: MagicMock,
        tmp_path: Path,
    ) -> None:
        data_file = tmp_path / "ppo.csv"
        data_file.write_text("timestamp,close\n1,100\n", encoding="utf-8")
        cfg = PPOSidecarConfig(
            data_path=str(data_file),
            check_interval_sec=1,
            retrain_interval_sec=1,
            history_path=tmp_path / "history.jsonl",
        )
        mock_retrain.return_value = PPORetrainResult(status="deployed", action="buy")

        _shutdown_event.clear()
        with (
            patch.object(_shutdown_event, "wait", side_effect=make_shutdown_wait(shutdown_event=_shutdown_event)),
            patch.object(_shutdown_event, "is_set", side_effect=[False, False, True]),
            patch.object(
                PPORetrainTrigger,
                "should_retrain",
                side_effect=[(True, "manual"), (False, "done")],
            ),
            patch.object(
                PPORetrainTrigger,
                "record_result",
                side_effect=RuntimeError("record error"),
            ),
        ):
            run_scheduler(cfg)

        assert mock_retrain.call_count >= 1
        _shutdown_event.clear()

    def test_update_signal_failure_pushes_neutral_fallback(self, tmp_path: Path) -> None:
        cfg = PPOSidecarConfig(
            data_path=str(tmp_path / "ppo.csv"),
            signal_path=tmp_path / "ppo_signal.json",
        )
        Path(cfg.data_path).write_text("timestamp,close\n1,100\n", encoding="utf-8")
        fake_model = _FakePPOModel()
        fake_trainer = MagicMock()
        fake_trainer.train.return_value = fake_model

        with (
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler.SELLBiasMitigationPPOTrainer",
                return_value=fake_trainer,
            ),
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._update_ppo_sidecar_signal",
                side_effect=RuntimeError("signal boom"),
            ),
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._push_neutral_fallback",
                return_value=True,
            ) as mock_fallback,
            patch(
                "scripts.v460.ml.ppo_retrain_scheduler._cleanup_training_cycle",
                return_value=None,
            ),
        ):
            result = retrain_once(cfg)

        assert result.status == "error"
        mock_fallback.assert_called_once_with(cfg.signal_path)


class TestPPOTrainingTimeout:
    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trainer = MagicMock()

        def _slow_train(*_args: object, **_kwargs: object) -> object:
            time.sleep(0.05)
            return _FakePPOModel()

        trainer.train.side_effect = _slow_train
        monkeypatch.setattr(
            "scripts.v460.ml.ppo_retrain_scheduler._TRAINING_TIMEOUT_SEC",
            0.01,
        )

        with pytest.raises(TimeoutError, match="exceeded"):
            _train_with_timeout(trainer, session_id="ppo_timeout")

        assert _TRAINING_TIMEOUT_SEC >= 1
