from .environment import HeavyTradingEnv

# Conditional import for PPO trainer
try:
    from .ppo_trainer import (
        CheckpointCallback,
        PPOTrainer,
        SafetyCallback,
        TensorBoardCallback,
    )

    _ppo_available = True
except ImportError:
    _ppo_available = False
    # Create dummy classes to avoid import errors
    PPOTrainer = None
    SafetyCallback = None
    TensorBoardCallback = None

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
