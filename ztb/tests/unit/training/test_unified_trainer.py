"""Unit tests for the unified trainer facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ztb.training.core.config_builder import ConfigBuilder
from ztb.training.unified_trainer import (
    UnifiedAlgorithm,
    UnifiedTrainer,
    UnifiedTrainerConfig,
    configure_progress_bar,
)


@pytest.fixture()
def sample_config() -> dict[str, object]:
    return {
        "algorithm": "ppo",
        "model_name": "test_model",
        "total_timesteps": 100_000,
        "ppo": {"verbose": 0},
    }


class TestUnifiedAlgorithm:
    def test_algorithm_values(self) -> None:
        assert [item.value for item in UnifiedAlgorithm] == [
            "ppo",
            "base_ml",
            "iterative",
            "ensemble",
            "curriculum",
        ]


class TestUnifiedTrainerConfig:
    def test_defaults(self) -> None:
        cfg = UnifiedTrainerConfig(algorithm=UnifiedAlgorithm.PPO)
        assert cfg.force is False
        assert cfg.enable_streaming is False
        assert cfg.stream_batch_size == 256
        assert cfg.total_timesteps is None

    def test_custom_values(self) -> None:
        cfg = UnifiedTrainerConfig(
            algorithm=UnifiedAlgorithm.ENSEMBLE,
            force=True,
            enable_streaming=True,
            stream_batch_size=128,
            max_features=42,
            total_timesteps=5000,
        )
        assert cfg.algorithm == UnifiedAlgorithm.ENSEMBLE
        assert cfg.force is True
        assert cfg.enable_streaming is True
        assert cfg.stream_batch_size == 128
        assert cfg.max_features == 42
        assert cfg.total_timesteps == 5000


class TestConfigureProgressBar:
    def test_cli_override(self, sample_config: dict[str, object]) -> None:
        enabled = configure_progress_bar(sample_config, cli_override=True)
        assert enabled is True
        assert sample_config["progress_bar"] is True
        assert sample_config["_progress_configured"] is True
        assert sample_config["ppo"]["verbose"] == 1  # type: ignore[index]

    def test_infer_from_config(self) -> None:
        config: dict[str, object] = {"progress_bar": False, "ppo": {"verbose": 1}}
        enabled = configure_progress_bar(config)
        assert enabled is True
        assert config["ppo"]["verbose"] == 1  # type: ignore[index]


class TestUnifiedTrainer:
    @patch("ztb.training.unified_trainer.safe_operation")
    def test_train_fallback(
        self, mock_safe_operation: MagicMock, sample_config: dict[str, object]
    ) -> None:
        mock_safe_operation.return_value = "fallback"
        trainer = UnifiedTrainer(sample_config)
        assert trainer.train() == "fallback"
        mock_safe_operation.assert_called_once()

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [("ppo", "ppo"), ("PPO", "ppo"), ("ensemble", "ensemble")],
    )
    @patch("ztb.training.unified_trainer.AlgorithmTrainer")
    @patch("ztb.training.unified_trainer.safe_operation")
    def test_train_delegates(
        self,
        mock_safe_operation: MagicMock,
        mock_algorithm_trainer: MagicMock,
        declared: str,
        expected: str,
        sample_config: dict[str, object],
    ) -> None:
        mock_safe_operation.side_effect = lambda **kw: kw["operation"]()
        mock_algorithm_trainer.return_value.train.return_value = {"ok": True}

        config = dict(sample_config)
        config["algorithm"] = declared
        trainer = UnifiedTrainer(config)

        result = trainer.train()

        mock_algorithm_trainer.assert_called_once()
        args, kwargs = mock_algorithm_trainer.return_value.train.call_args
        assert args[0] == expected
        assert isinstance(args[1], dict)
        assert result == {"ok": True}

    @patch.object(ConfigBuilder, "build_unified_config")
    def test_build_unified_config_cached(
        self, mock_build: MagicMock, sample_config: dict[str, object]
    ) -> None:
        mock_build.return_value = {"foo": "bar"}
        trainer = UnifiedTrainer(sample_config)

        first = trainer.build_unified_config()
        second = trainer.build_unified_config()

        assert first == {"foo": "bar"}
        assert second == {"foo": "bar"}
        mock_build.assert_called_once()

    @patch("ztb.training.unified_trainer.AlgorithmTrainer")
    @patch.object(ConfigBuilder, "build_unified_config")
    @patch("ztb.training.unified_trainer.safe_operation")
    def test_total_timesteps_override(
        self,
        mock_safe_operation: MagicMock,
        mock_build: MagicMock,
        mock_algorithm_trainer: MagicMock,
        sample_config: dict[str, object],
    ) -> None:
        mock_safe_operation.side_effect = lambda **kw: kw["operation"]()
        mock_build.return_value = {
            "total_timesteps": 100,
            "ppo": {"total_timesteps": 100},
        }

        trainer = UnifiedTrainer(sample_config, total_timesteps=512)
        trainer.train()

        unified_config = mock_algorithm_trainer.return_value.train.call_args.args[1]
        assert unified_config["total_timesteps"] == 512
        assert unified_config["ppo"]["total_timesteps"] == 512  # type: ignore[index]
