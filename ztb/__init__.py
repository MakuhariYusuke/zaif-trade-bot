"""
ZAIF Trade Bot - Advanced Trading System with Reinforcement Learning

This package provides a comprehensive trading bot framework featuring:
- SAC (Soft Actor-Critic) reinforcement learning algorithms
- Advanced backtesting and analysis tools
- Risk management and portfolio optimization
- Real-time trading execution
- Comprehensive monitoring and logging

Main Components:
- trading: Core trading logic and strategies
- analysis: Backtesting and performance analysis tools
- evaluation: Trading performance evaluation
- config: Configuration management system
- utils: Utility functions and helpers
- data: Data processing and augmentation tools
"""

from typing import TYPE_CHECKING

__version__ = "4.2.0"
__author__ = "MakuhariYusuke"
__description__ = "Advanced trading bot with reinforcement learning"

if TYPE_CHECKING:
    from .config.schema import GlobalConfig

# Import main components for easy access
from .analysis import BacktestAnalyzer
from .config import ConfigManager
from .data import BTCBiasDetector, BTCDataAugmentor

# Define public API
__all__ = [
    # Core components
    "ConfigManager",
    "BacktestAnalyzer",
    "BTCDataAugmentor",
    "BTCBiasDetector",
    # Metadata
    "__version__",
    "__author__",
    "__description__",
]


def get_version() -> str:
    """Get the current version of the package."""
    return __version__


def get_config() -> "GlobalConfig":
    """Get the current configuration."""
    return ConfigManager.get_instance().get_config()
