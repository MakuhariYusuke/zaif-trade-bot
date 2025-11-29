#!/usr/bin/env python3
"""
Type definitions for common analysis components.

Provides type-safe interfaces for data loading, analysis, and path management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

T = TypeVar("T", covariant=True)


# Risk Management Types
class RiskProfile(Enum):
    """Risk profile levels."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskProfileLimits:
    """Risk limit configuration."""

    # Position limits
    max_position_notional: float  # Maximum position size in JPY
    max_single_trade_pct: float  # Max % of capital per trade

    # Daily loss limits
    daily_loss_limit_pct: float  # Max daily loss as % of starting capital
    max_drawdown_pct: float  # Max drawdown before stopping

    # Trade frequency
    max_trades_per_hour: int  # Maximum trades per hour
    min_trade_interval_sec: int  # Minimum seconds between trades

    # Risk metrics
    max_volatility_pct: float  # Max portfolio volatility
    required_sharpe_ratio: float  # Minimum Sharpe ratio threshold

    # Stop loss settings
    stop_loss_pct: float  # Stop loss percentage
    take_profit_pct: float  # Take profit percentage


# Protocol definitions
class FeatureCalculator(Protocol):
    """Protocol for feature calculators."""

    def calculate(self, data: Any) -> Any:
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

    def fit(self, data: Any) -> None:
        """Fit normalizer to data."""
        ...

    def transform(self, data: Any) -> Any:
        """Transform data."""
        ...

    def inverse_transform(self, data: Any) -> Any:
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


# Data Loading Types
class DataSource(Protocol):
    """Protocol for data sources that can be loaded."""

    def exists(self) -> bool:
        """Check if the data source exists."""
        ...

    def get_path(self) -> Path:
        """Get the path to the data source."""
        ...


# Analysis Types
AnalysisResult = Dict[str, Any]
AnalysisConfig = Dict[str, Any]


class AnalysisInput(TypedDict, total=False):
    """Type for analysis input data."""

    data: Union[Any, Dict[str, Any], List[Dict[str, Any]]]
    config: AnalysisConfig
    metadata: Dict[str, Any]


class AnalysisOutput(TypedDict, total=False):
    """Type for analysis output data."""

    results: AnalysisResult
    summary: Dict[str, Any]
    plots: List[str]  # File paths to generated plots
    reports: List[str]  # File paths to generated reports
    metadata: Dict[str, Any]


# Backtest Data Types
class BacktestResult(TypedDict, total=False):
    """Type for backtest result data."""

    total_return_pct: float
    final_portfolio_value: float
    initial_balance: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_trade_return_pct: float
    portfolio_values: List[float]
    trades_history: List[Dict[str, Any]]


class TrainingResult(TypedDict, total=False):
    """Type for training result data."""

    training_stats: Dict[str, Any]
    final_reward: float
    total_timesteps: int
    training_time: float
    action_distribution: Dict[str, float]
    performance_metrics: Dict[str, Any]


# Path Management Types
class PathConfig(TypedDict, total=False):
    """Configuration for path management."""

    base_dir: Path
    create_dirs: bool
    standard_paths: Dict[str, str]


# Validation Types
ValidationResult = Dict[str, Union[bool, List[str]]]


class ValidationRule(TypedDict, total=False):
    """Type for validation rules."""

    field: str
    required: bool
    type_check: Optional[type]
    range_check: Optional[Dict[str, Union[int, float]]]
    custom_validator: Optional[str]  # Function name for custom validation


# Error Types
class AnalysisErrorInfo(TypedDict, total=False):
    """Type for error information in analysis."""

    component: str
    error_type: str
    message: str
    context: Dict[str, Any]
    timestamp: str


# Component Configuration Types
class DataLoaderConfig(TypedDict, total=False):
    """Configuration for data loaders."""

    base_path: Path
    required_files: List[str]
    file_patterns: Dict[str, str]
    error_handling: str  # 'strict', 'warn', 'ignore'


class AnalyzerConfig(TypedDict, total=False):
    """Configuration for analyzers."""

    name: str
    input_validation: List[ValidationRule]
    output_validation: List[ValidationRule]
    error_handling: str
    logging_level: str


class PathManagerConfig(TypedDict, total=False):
    """Configuration for path managers."""

    base_dir: Path
    create_dirs: bool
    standard_paths: Dict[str, str]
    permissions_check: bool


# Factory Types
class ComponentFactory(Protocol[T]):
    """Protocol for component factories."""

    def create(self, config: Dict[str, Any]) -> T:
        """Create a component instance from configuration."""
        ...


# Registry Types
ComponentRegistry = Dict[str, Any]
FactoryRegistry = Dict[str, ComponentFactory[Any]]


# Utility Types
FileInfo = Dict[str, Union[str, int, float]]
DirectoryInfo = Dict[str, Union[List[FileInfo], int]]


class SystemInfo(TypedDict, total=False):
    """System information for analysis components."""

    python_version: str
    platform: str
    available_memory: int
    cpu_count: int
    component_versions: Dict[str, str]


# Risk Management Types
class RiskStatus(TypedDict):
    """Risk status summary."""

    daily_loss: float
    daily_loss_limit: float
    portfolio_value: float
    portfolio_volatility: float
    trades_this_hour: int
    max_trades_per_hour: int
    trailing_stop_level: Optional[float]
    cooldown_period: int


class ExtendedRiskStatus(TypedDict, total=False):
    """Extended risk status with additional metrics."""

    daily_pnl: float
    daily_trades: int
    daily_trade_limit: int
    hourly_trades: int
    hourly_trade_limit: int
    emergency_stop_loss: float
    position: float
    entry_price: float
    statistics: Dict[str, Any]  # 統計情報


class TriggerStatus(TypedDict):
    """Trigger status."""

    triggered: bool
    reason: str


class PositionMonitorResult(TypedDict):
    """Position monitoring result."""

    trailing_stop: TriggerStatus
    take_profit: TriggerStatus


class RiskStatusReport(TypedDict):
    """Comprehensive risk status report."""

    profile: RiskProfileLimits
    current_status: RiskStatus
    limits: RiskProfileLimits
