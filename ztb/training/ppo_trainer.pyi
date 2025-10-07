"""
Type stubs for PPO trainer implementations.

This module provides type annotations for complex PPO trainer classes
to improve IDE support and type checking.
"""

from typing import Any, Dict, Optional, Protocol
from sb3_contrib import MaskablePPO

# Type stubs for PPO trainer classes
class PPOTrainerProtocol(Protocol):
    """Protocol for PPO Trainer implementations."""

    def train(self, session_id: str) -> Optional[MaskablePPO]:
        """Train the model and return it."""
        ...

    def get_reward_stats(self) -> Dict[str, float]:
        """Get training reward statistics."""
        ...

    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias."""
        ...

class PPOTrainerAutoHalt:
    """Auto-halt capable PPO trainer."""

    def __init__(
        self,
        data_path: str,
        config: Dict[str, Any],
        checkpoint_dir: str,
        eval_gates: Optional[Any] = None,
        halt_callback: Optional[Any] = None,
        checkpoint_interval: int = 10000,
    ) -> None:
        ...

    def train(self, session_id: str) -> Optional[MaskablePPO]:
        """Train the model with auto-halt capability."""
        ...

    def get_reward_stats(self) -> Dict[str, float]:
        """Get training reward statistics."""
        ...

    def neutralize_policy_bias(self) -> None:
        """Neutralize policy head bias."""
        ...