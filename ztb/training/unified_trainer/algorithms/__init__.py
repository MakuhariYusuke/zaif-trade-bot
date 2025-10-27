"""
Algorithm-specific training implementations.
"""

import logging
from typing import Any, Dict, Optional

from ztb.training.unified_trainer.base.base_trainer import BaseAlgorithmTrainer

from .ppo_trainer import PPOTrainer
from .sac_trainer import SACTrainer
from .self_supervised_trainer import SelfSupervisedTrainer


def create_algorithm_trainer(
    algorithm: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    gradient_accumulation_steps: int = 1,
    system_optimizer: Optional[Any] = None,
) -> BaseAlgorithmTrainer:
    """Factory function to create algorithm-specific trainer."""
    algorithm = algorithm.lower()

    if algorithm == "sac":
        return SACTrainer(config, logger, gradient_accumulation_steps, system_optimizer)
    elif algorithm == "ppo":
        return PPOTrainer(config, logger, gradient_accumulation_steps)
    elif algorithm == "self_supervised":
        return SelfSupervisedTrainer(config, logger)
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
