"""Compatibility tests for legacy PPO training import paths."""

from pathlib import Path

from ztb.training.config.ppo_config import (
    DEFAULT_PPO_CONFIG as CANONICAL_DEFAULT_PPO_CONFIG,
)
from ztb.training.config.ppo_config import get_ppo_config as canonical_get_ppo_config
from ztb.training.core.ppo_trainer import PPOTrainer as CorePPOTrainer
from ztb.training.core.ppo_trainer import PPOTrainingConfig
from ztb.training.custom_ppo import CustomPPO as CanonicalCustomPPO
from ztb.training.ppo_config import DEFAULT_PPO_CONFIG, PPOConfig, get_ppo_config
from ztb.training.ppo_trainer import (
    CustomPPO,
    MaskablePPO,
    PPOTrainer,
    TrainingConfig,
)
from sb3_contrib import MaskablePPO as CanonicalMaskablePPO


class TestPPOTrainerCompatibilityShims:
    """Verify that legacy PPO imports resolve to the current implementation."""

    def test_trainer_shim_reexports_core_trainer(self) -> None:
        assert PPOTrainer is CorePPOTrainer

    def test_training_config_alias_matches_core(self) -> None:
        assert TrainingConfig is PPOTrainingConfig

    def test_model_shim_exports_current_symbols(self) -> None:
        assert CustomPPO is CanonicalCustomPPO
        assert MaskablePPO is CanonicalMaskablePPO

    def test_config_shim_reexports_canonical_defaults(self) -> None:
        assert DEFAULT_PPO_CONFIG == CANONICAL_DEFAULT_PPO_CONFIG
        assert PPOConfig.__name__ == "PPOConfig"

    def test_config_shim_reexports_get_ppo_config(self) -> None:
        assert get_ppo_config({"learning_rate": 1e-4}) == canonical_get_ppo_config(
            {"learning_rate": 1e-4}
        )

    def test_trainer_shim_initializes_current_trainer(
        self, tmp_path: Path
    ) -> None:
        trainer = PPOTrainer(
            data_path="dummy_path.csv",
            config={"ppo": {"use_custom_ppo": False}},
            checkpoint_dir=str(tmp_path),
        )

        assert trainer.data_path == "dummy_path.csv"
        assert trainer.training_config.use_custom_ppo is False
