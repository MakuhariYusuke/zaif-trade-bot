"""Focused PPO sidecar foundation tests.

PPO sidecar はまだ live wiring 前段だが、
signal 契約と config foundation はここで固定しておく。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.v460.lib.sidecar_signal_io import (
    clear_ppo_sidecar_signal_cache,
    create_neutral_ppo_signal,
    read_ppo_sidecar_signal,
    read_ppo_sidecar_signal_with_status,
    write_ppo_sidecar_signal,
)
from scripts.v460.lib.sidecar_types import (
    PPOSidecarSignal,
    normalize_ppo_action_probabilities,
    resolve_ppo_sidecar_action,
    should_activate_ppo_sidecar,
)
from scripts.v460.ml.ppo_sidecar_config import PPOSidecarConfig


class TestPPOSidecarSignal:
    def test_from_probabilities_resolves_action_and_confidence(self) -> None:
        signal = PPOSidecarSignal.from_probabilities(
            timestamp="2026-04-01T00:00:00+00:00",
            action_probabilities={"buy": 0.65, "sell": 0.20, "skip": 0.15},
            model_version="ppo_sidecar_v1",
        )

        assert signal.action == "buy"
        assert signal.selected_side == "buy"
        assert signal.confidence == pytest.approx(0.65)
        assert signal.action_margin == pytest.approx(0.45)

    def test_normalize_action_probabilities(self) -> None:
        normalized = normalize_ppo_action_probabilities(
            {"buy": 2.0, "sell": 1.0, "skip": 1.0}
        )

        assert normalized["buy"] == pytest.approx(0.5)
        assert normalized["sell"] == pytest.approx(0.25)
        assert normalized["skip"] == pytest.approx(0.25)
        assert resolve_ppo_sidecar_action(normalized) == "buy"

    def test_neutral_signal_uses_skip(self) -> None:
        signal = create_neutral_ppo_signal("2026-04-01T00:00:00+00:00")
        assert signal.action == "skip"
        assert signal.selected_side is None
        assert signal.confidence == 0.0

    def test_override_activation_requires_confidence_and_margin(self) -> None:
        strong = PPOSidecarSignal.from_probabilities(
            timestamp="2026-04-01T00:00:00+00:00",
            action_probabilities={"buy": 0.72, "sell": 0.18, "skip": 0.10},
            model_version="ppo_sidecar_v1",
        )
        weak = PPOSidecarSignal.from_probabilities(
            timestamp="2026-04-01T00:00:00+00:00",
            action_probabilities={"buy": 0.52, "sell": 0.38, "skip": 0.10},
            model_version="ppo_sidecar_v1",
        )

        assert should_activate_ppo_sidecar(strong) is True
        assert should_activate_ppo_sidecar(weak) is False


class TestPPOSidecarSignalIo:
    def test_round_trip(self, tmp_path: Path) -> None:
        signal = PPOSidecarSignal.from_probabilities(
            timestamp="2026-04-01T00:00:00+00:00",
            action_probabilities={"buy": 0.10, "sell": 0.15, "skip": 0.75},
            model_version="ppo_sidecar_v1",
            regime_hint="ranging",
            training_metrics={"gross_roi": 0.012},
        )
        path = tmp_path / "ppo_sidecar_signal.json"

        write_ppo_sidecar_signal(signal, path)
        loaded = read_ppo_sidecar_signal(path, ttl_sec=0)

        assert loaded is not None
        assert loaded.action == "skip"
        assert loaded.action_probabilities["skip"] == pytest.approx(0.75)
        assert loaded.training_metrics["gross_roi"] == pytest.approx(0.012)

    def test_stale_status(self, tmp_path: Path) -> None:
        clear_ppo_sidecar_signal_cache()
        stale_signal = create_neutral_ppo_signal(
            (
                datetime.now(timezone.utc) - timedelta(hours=3)
            ).isoformat()
        )
        path = tmp_path / "ppo_sidecar_signal.json"
        write_ppo_sidecar_signal(stale_signal, path)

        loaded, status = read_ppo_sidecar_signal_with_status(path, ttl_sec=60)

        assert loaded is None
        assert status == "stale"


class TestPPOSidecarConfig:
    def test_from_yaml_dict_builds_discrete_runtime_contract(self) -> None:
        cfg = PPOSidecarConfig.from_yaml_dict(
            {
                "data": {"data_path": "data/ppo_training.csv"},
                "output": {"model_dir": "models/v461"},
                "training": {"total_timesteps": 12345},
                "ppo_hyperparameters": {"learning_rate": 1e-4, "n_steps": 256},
                "ppo_sidecar": {
                    "signal_path": "cache/ppo_sidecar_signal.json",
                    "incremental_timesteps": 6789,
                    "min_override_confidence": 0.61,
                    "min_action_probability_gap": 0.12,
                    "enable_target_entropy": False,
                },
            }
        )

        trainer_config = cfg.build_trainer_config()

        assert cfg.data_path == "data/ppo_training.csv"
        assert cfg.total_timesteps == 12345
        assert cfg.incremental_timesteps == 6789
        assert cfg.min_override_confidence == pytest.approx(0.61)
        assert cfg.min_action_probability_gap == pytest.approx(0.12)
        assert cfg.use_continuous_actions is False
        assert cfg.action_space_type == "discrete"
        assert trainer_config["use_continuous_actions"] is False
        assert trainer_config["action_space_type"] == "discrete"
        assert trainer_config["algorithm"] == "ppo"

    def test_continuous_actions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="discrete actions"):
            PPOSidecarConfig(
                data_path="data/ppo_training.csv",
                use_continuous_actions=True,
            )
