# Conditional import for PPO trainer - LAZY LOAD to avoid heavy torch imports
_ppo_available = False
PPOTrainer = None
CheckpointCallback = None
SafetyCallback = None
TensorBoardCallback = None


def _load_ppo_trainer():
    """Lazy load PPO trainer components."""
    global \
        _ppo_available, \
        PPOTrainer, \
        CheckpointCallback, \
        SafetyCallback, \
        TensorBoardCallback
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


# Avoid importing environment at package import time to prevent heavy imports / side effects
# Provide lazy access via module-level __getattr__ (PEP 562) to keep backwards compatibility
HeavyTradingEnv = None


def _get_HeavyTradingEnv():
    """Lazy loader for HeavyTradingEnv to avoid import-time side effects."""
    global HeavyTradingEnv
    if HeavyTradingEnv is not None:
        return HeavyTradingEnv
    try:
        from .environment.environment import (
            HeavyTradingEnv as _HTE,
        )  # local import to avoid side-effects

        HeavyTradingEnv = _HTE
        return HeavyTradingEnv
    except (ImportError, OSError):
        HeavyTradingEnv = None
        return None


def __getattr__(name: str):
    if name == "HeavyTradingEnv":
        return _get_HeavyTradingEnv()
    if name == "PPOTrainer":
        _load_ppo_trainer()
        return PPOTrainer
    # Try to resolve submodules lazily (allows tests to patch ztb.trading.<module>)
    try:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    except Exception:
        raise AttributeError(f"module {__name__} has no attribute {name}")


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
