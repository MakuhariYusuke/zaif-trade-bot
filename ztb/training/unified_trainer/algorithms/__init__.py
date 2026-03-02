"""
Algorithm-specific training implementations.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from ztb.training.unified_trainer.base.base_trainer import BaseAlgorithmTrainer

_LAZY_TRAINERS = {
    "PPOTrainer": (".ppo_trainer", "PPOTrainer"),
    "SACTrainer": (".sac_trainer", "SACTrainer"),
    "SelfSupervisedTrainer": (".self_supervised_trainer", "SelfSupervisedTrainer"),
}

def _load_trainer(name: str):
    mod_name, attr = _LAZY_TRAINERS[name]
    module = __import__(f"{__name__}{mod_name}", fromlist=[attr])
    return getattr(module, attr)

def __getattr__(name: str):
    if name in _LAZY_TRAINERS:
        cls = _load_trainer(name)
        globals()[name] = cls
        return cls
    raise AttributeError(name)

def create_algorithm_trainer(
    algorithm: str,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
    gradient_accumulation_steps: int = 1,
    system_optimizer: Any | None = None,
    optimizer_tracker: OptimizerFeatureTracker | None = None,
) -> BaseAlgorithmTrainer:
    """Factory function to create algorithm-specific trainer."""
    algorithm = algorithm.lower()

    if algorithm == "sac":
        SACTrainer = _load_trainer("SACTrainer")
        return SACTrainer(
            config,
            None,  # env
            logger,
            gradient_accumulation_steps,
            system_optimizer,
            optimizer_tracker,
        )
    elif algorithm == "ppo":
        PPOTrainer = _load_trainer("PPOTrainer")
        return PPOTrainer(
            config,
            logger,
            gradient_accumulation_steps,
            optimizer_tracker=optimizer_tracker,
        )
    elif algorithm == "self_supervised":
        SelfSupervisedTrainer = _load_trainer("SelfSupervisedTrainer")
        return SelfSupervisedTrainer(
            config,
            logger,
            gradient_accumulation_steps,
            system_optimizer,
            optimizer_tracker,
        )
    elif algorithm == "multimodal":
        # Create multimodal config from unified config
        # Temporarily disabled due to circular import issues
        # multimodal_config = MultimodalConfig(
        #     price_feature_dim=config.get('price_feature_dim', 156),
        #     text_embedding_dim=config.get('text_embedding_dim', 768),
        #     economic_feature_dim=config.get('economic_feature_dim', 10),
        #     action_dim=config.get('action_dim', 3),
        #     hidden_dim=config.get('multimodal_hidden_dim', 256),
        #     num_heads=config.get('multimodal_num_heads', 8)
        # )
        # return MultimodalSACTrainer(multimodal_config, config, config.get('env_config', {}))
        raise NotImplementedError(
            "Multimodal training temporarily disabled due to circular import issues"
        )
    elif algorithm == "online_learning":
        # Create online learning config from unified config
        # Temporarily disabled due to circular import issues
        # online_config = OnlineLearningConfig(
        #     learning_mode=config.get('online_learning_mode', 'incremental'),
        #     batch_size=config.get('online_batch_size', 32),
        #     max_memory_samples=config.get('online_memory_samples', 10000),
        #     adaptation_trigger_threshold=config.get('online_adaptation_threshold', 0.1)
        # )
        # return OnlineLearningSACTrainer(online_config, config, config.get('env_config', {}))
        raise NotImplementedError(
            "Online learning training temporarily disabled due to circular import issues"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

__all__ = [
    "SACTrainer",
    "PPOTrainer",
    "SelfSupervisedTrainer",
    "create_algorithm_trainer",
]
if TYPE_CHECKING:
    # Import for forward type references only. Avoid importing heavy modules at runtime
    # to prevent test collection/import-time errors where these dependencies may be
    # unavailable.
    from ztb.features.processors.optimization.features import (
        OptimizerFeatureTracker,
    )
