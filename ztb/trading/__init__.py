# Conditional import for PPO trainer
try:
    from ztb.training.archive.ppo_trainer_old import (  # type: ignore[attr-defined]
        CheckpointCallback,
        PPOTrainer,
        SafetyCallback,
        TensorBoardCallback,
    )

    _ppo_available = True
except ImportError:
    _ppo_available = False
    # Create dummy classes to avoid import errors
    PPOTrainer = None  # type: ignore[misc,assignment]
    SafetyCallback = None
    TensorBoardCallback = None

from .environment.environment import HeavyTradingEnv

__all__ = [
    "HeavyTradingEnv",
]

if _ppo_available:
    __all__.extend(
        [
            "PPOTrainer",
            "TensorBoardCallback",
            "CheckpointCallback",
            "SafetyCallback",
        ]
    )
