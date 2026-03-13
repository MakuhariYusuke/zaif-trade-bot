"""
Core base classes for configuration modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class BaseConfigLoader(ABC):
    """Abstract base class for configuration loaders."""

    @abstractmethod
    def load_config(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
