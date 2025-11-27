#!/usr/bin/env python3
"""
Type definitions and protocols for ZTB system.

This module provides type hints and protocols used across the codebase.
"""

import os
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

import numpy as np
from numpy import typing as npt

import numpy as np
import pandas as pd
from numpy import typing as npt
from typing_extensions import TypedDict

# Type variables for generic types
T = TypeVar("T", bound=np.floating)
U = TypeVar("U", bound=np.integer)

# Generic array types
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int64]
NDArrayBool = npt.NDArray[np.bool_]


# Basic data types
NumericType = Union[int, float, np.number]
ArrayLike = Union[NDArrayFloat, pd.Series, List[NumericType]]

# Trading action types
ActionType = int  # 0: HOLD, 1: BUY, 2: SELL
ActionMask = NDArrayBool  # Boolean array for valid actions

# Market data types
PriceData = TypedDict(
    "PriceData",
    {
        "open": ArrayLike,
        "high": ArrayLike,
        "low": ArrayLike,
        "close": ArrayLike,
        "volume": Optional[ArrayLike],
    },
    total=False,
)

OHLCData = TypedDict(
    "OHLCData",
    {"open": ArrayLike, "high": ArrayLike, "low": ArrayLike, "close": ArrayLike},
)

InfoDict = TypedDict(
    "InfoDict",
    {
        "portfolio_value": float,
        "position": float,
        "reward": float,
        "step": int,
        "episode": int,
    },
    total=False,
)


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
class FeatureCalculator(Protocol):
    """Protocol for feature calculators."""

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from input data."""
        ...

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        ...


class TrainerProtocol(Protocol):
    """Protocol for SAC trainers."""

    def train(self) -> Dict[str, Any]:
        """Train the model."""
        ...

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the trained model."""
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


class IndicatorInfo(TypedDict):
    """Indicator information dictionary."""

    description: str
    talib_available: bool
    parameters: List[str]
    inputs: Optional[List[str]]
    output_range: Optional[Tuple[Optional[float], Optional[float]]]
    interpretation: Optional[str]


class StatsResult(TypedDict):
    """Statistics result dictionary."""

    mean: float
    std: float
    ci95: List[float]


class FeatureMetrics(TypedDict):
    """Feature evaluation metrics dictionary."""

    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sample_count: int


# Utility type aliases
PathLike = Union[str, "os.PathLike[str]"]
JSONSerializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@runtime_checkable
class PerformanceMonitorProtocol(Protocol):
    """Protocol for performance monitors."""

    def record_decision(self, decision: Any) -> None:
        """Record a decision."""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        ...


@runtime_checkable
class ThresholdManagerProtocol(Protocol):
    """Protocol for threshold managers."""

    def get_adaptive_gates(self) -> Dict[str, float]:
        """Get adaptive threshold gates."""
        ...

    def update_thresholds(self, evaluation_results: Dict[str, Any]) -> None:
        """Update thresholds based on evaluation results."""
        ...


@runtime_checkable
class FeeModelProtocol(Protocol):
    """Protocol for fee models."""

    def calculate_fee(self, trade_value: float, trade_type: str = "buy") -> float:
        """Calculate transaction fee."""
        ...

    def get_fee_rate(self, trade_type: str = "buy") -> float:
        """Get fee rate."""
        ...


@runtime_checkable
class NormalizerProtocol(Protocol):
    """Protocol for data normalizers."""

    def fit(self, data: npt.NDArray[np.float64]) -> None:
        """Fit normalizer to data."""
        ...

    def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform data."""
        ...

    def inverse_transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Inverse transform data."""
        ...


@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for loggers."""

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        ...
