"""Compatibility forwarding module for ppo_trainer.

Some parts of the codebase import `ztb.trading.training.ppo_trainer` while the
real implementation lives at `ztb.trading.ppo_trainer`. To make static type
checking and imports robust across environments we provide a small forwarding
module which re-exports the real implementation when available, and otherwise
exposes a minimal typed stub used only to satisfy mypy or light-weight tooling.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Provide type info to mypy by importing the real symbol if possible
    try:
        from ztb.trading.ppo_trainer import PPOTrainer  # type: ignore
    except Exception:  # pragma: no cover - best-effort for static analysis

        class PPOTrainer:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                ...

            def train(self, *args: Any, **kwargs: Any) -> object:
                ...

else:  # Runtime: prefer to use the real implementation when available
    try:
        from ztb.trading.ppo_trainer import PPOTrainer  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback

        class PPOTrainer:
            def __init__(self, *args, **kwargs) -> None:
                raise RuntimeError("PPOTrainer implementation not available")


__all__ = ["PPOTrainer"]
