import importlib
import warnings
from collections.abc import Callable
from types import ModuleType

from .components import (
    ActionValidator,
    DataProcessor,
    MemoryManager,
    ObservationBuilder,
    PositionManager,
    RewardCalculator,
    StreamingHandler,
)
from .types import EPSILON, StatisticsDict
from .utils.config import EnvironmentConfig, RewardSettings


def _load_environment_exports(
    import_module: Callable[[str, str], ModuleType] | None = None,
) -> tuple[object | None, object | None]:
    """Load heavy environment exports with narrow optional-import handling."""
    loader = importlib.import_module if import_module is None else import_module
    try:
        env_module = loader(".environment", __name__)
    except ImportError as exc:
        warnings.warn(
            f"HeavyTradingEnv could not be imported: {exc}",
            ImportWarning,
            stacklevel=2,
        )
        return None, None
    return (
        getattr(env_module, "FlipHeavyTradingEnv"),
        getattr(env_module, "HeavyTradingEnv"),
    )


FlipHeavyTradingEnv, HeavyTradingEnv = _load_environment_exports()

__all__ = [
    "HeavyTradingEnv",
    "FlipHeavyTradingEnv",
    "EnvironmentConfig",
    "RewardSettings",
    "PositionManager",
    "RewardCalculator",
    "DataProcessor",
    "MemoryManager",
    "StreamingHandler",
    "ObservationBuilder",
    "ActionValidator",
    "EPSILON",
    "StatisticsDict",
]
