"""Training package for trading-related trainers.

This package intentionally provides light typed forwarders/stubs so that mypy
and other static checkers can import trainer symbols without requiring heavy
runtime dependencies during static analysis.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # During static type checking, expose the real symbols by importing the
    # implementation modules if available in the environment used by mypy.
    try:
        from ztb.trading import ppo_trainer as ppo_impl
    except Exception:  # pragma: no cover - best-effort for static analysis
        ppo_impl = None  # type: ignore[assignment]

    if ppo_impl is not None:
        from ztb.trading.ppo_trainer import PPOTrainer  # type: ignore[attr-defined]
    else:  # pragma: no cover - fallback for type checkers

        class PPOTrainer:  # type: ignore
            def __init__(self, *args, **kwargs) -> None:
                ...

            def train(self, *args, **kwargs) -> object:
                ...

else:
    # At runtime prefer importing the real implementation where present. If
    # not present keep attributes minimal but present so runtime imports do not
    # crash accidental imports during packaging or light-weight tasks.
    try:
        from ztb.trading.ppo_trainer import PPOTrainer  # type: ignore
    except Exception:  # pragma: no cover - runtime fallback

        class PPOTrainer:
            """Minimal runtime stub used only when the full trainer isn't
            installed. This stub raises at train time to avoid silent
            misbehavior.
            """

            def __init__(self, *args, **kwargs) -> None:
                raise RuntimeError("PPOTrainer implementation not available")

            def train(self, *args, **kwargs) -> object:  # pragma: no cover
                raise RuntimeError("PPOTrainer implementation not available")

__all__ = ["PPOTrainer"]
