#!/usr/bin/env python3
"""
Type definitions and protocols for ZTB system.

This module provides type hints and protocols used across the codebase.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from typing_extensions import TypedDict
import numpy as np
import os
import pandas as pd


# Basic data types
NumericType = Union[int, float, np.number[Any]]
ArrayLike = Union[np.ndarray[Any, np.dtype[np.floating[Any]]], pd.Series, List[NumericType]]

# Trading action types
ActionType = int  # 0: HOLD, 1: BUY, 2: SELL
ActionMask = np.ndarray[Any, np.dtype[np.bool_]]  # Boolean array for valid actions

# Market data types
PriceData = TypedDict('PriceData', {
    'open': ArrayLike,
    'high': ArrayLike,
    'low': ArrayLike,
    'close': ArrayLike,
    'volume': Optional[ArrayLike]
}, total=False)

OHLCData = TypedDict('OHLCData', {
    'open': ArrayLike,
    'high': ArrayLike,
    'low': ArrayLike,
    'close': ArrayLike
})

# Configuration types
class TrainingConfig(TypedDict, total=False):
    """Training configuration dictionary."""
    total_timesteps: int
    learning_rate: float
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: Optional[float]
    verbose: int
    seed: Optional[int]

class EnvironmentConfig(TypedDict, total=False):
    """Environment configuration dictionary."""
    max_steps: int
    initial_balance: float
    transaction_fee: float
    slippage: float
    feature_columns: List[str]
    reward_scaling: float
    position_penalty: float

class ModelConfig(TypedDict, total=False):
    """Model configuration dictionary."""
    policy: str
    features_dim: int
    action_space_size: int
    hidden_layers: List[int]
    activation_fn: str

# Protocol definitions
class TradingEnvironment(Protocol):
    """Protocol for trading environments."""

    def reset(self: "TradingEnvironment", **kwargs: Any) -> Tuple[np.ndarray[Any, np.dtype[np.floating[Any]]], Dict[str, Any]]:
        """Reset environment and return initial observation and info."""
        ...

    def step(self: "TradingEnvironment", action: ActionType) -> Tuple[np.ndarray[Any, np.dtype[np.floating[Any]]], float, bool, bool, Dict[str, Any]]:
        """Execute action and return next observation, reward, terminated, truncated, info."""
        ...

    def render(self) -> Optional[str]:
        """Render environment state."""
        ...

    @property
    def action_space(self) -> Any:
        """Get action space."""
        ...

    @property
    def observation_space(self) -> Any:
        """Get observation space."""
        ...

class FeatureCalculator(Protocol):
    """Protocol for feature calculators."""

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from input data."""
        ...

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        ...

class CallbackProtocol(Protocol):
    """Protocol for training callbacks."""

    def __call__(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """Callback function."""
        ...

# Result types
class TrainingResult(TypedDict):
    """Training result dictionary."""
    model_path: str
    total_timesteps: int
    final_reward: float
    best_reward: float
    training_time: float
    config: TrainingConfig

class BacktestResult(TypedDict):
    """Backtest result dictionary."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    final_portfolio_value: float
    action_distribution: Dict[str, int]

class ValidationResult(TypedDict):
    """Validation result dictionary."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Optional[Dict[str, float]]

# Utility type aliases
PathLike = Union[str, 'os.PathLike[str]']
JSONSerializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]