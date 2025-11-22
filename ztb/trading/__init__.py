# Conditional import for PPO trainer - LAZY LOAD to avoid heavy torch imports
_ppo_available = False
PPOTrainer = None
CheckpointCallback = None
SafetyCallback = None
TensorBoardCallback = None


def _load_ppo_trainer():
    """Lazy load PPO trainer components."""
    global _ppo_available, PPOTrainer, CheckpointCallback, SafetyCallback, TensorBoardCallback
    if not _ppo_available:
        try:
            from ztb.training.archive.ppo_trainer_old import (  # type: ignore[attr-defined]
                CheckpointCallback,
                PPOTrainer,
                SafetyCallback,
                TensorBoardCallback,
            )
            _ppo_available = True
        except (ImportError, OSError):
            _ppo_available = False

try:
    from .environment.environment import HeavyTradingEnv
except (ImportError, OSError):
    HeavyTradingEnv = None  # type: ignore[assignment]

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
