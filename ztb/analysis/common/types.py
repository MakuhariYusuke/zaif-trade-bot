#!/usr/bin/env python3
"""
Type definitions for common analysis components.

Provides type-safe interfaces for data loading, analysis, and path management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

from ztb.types.common import ConfigSection, MetricsDict, ObjectMap, ObjectRecords, StringMap

T = TypeVar("T", covariant=True)

# Risk Management Types
class RiskProfile(Enum):
    """Risk profile levels."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    DYNAMIC = "dynamic"

@dataclass
class RiskProfileLimits:
    """Risk limit configuration."""

    # Position limits
    max_position_notional: float = 1000000.0  # Maximum position size in JPY
    max_single_trade_pct: float = 0.1  # Max % of capital per trade

    # Daily loss limits
    daily_loss_limit_pct: float = 0.05  # Max daily loss as % of starting capital
    max_drawdown_pct: float = 0.1  # Max drawdown before stopping

    # Trade frequency
    max_trades_per_hour: int = 10  # Maximum trades per hour
    min_trade_interval_sec: int = 60  # Minimum seconds between trades

    # Risk metrics
    max_volatility_pct: float = 0.2  # Max portfolio volatility
    required_sharpe_ratio: float = 1.0  # Minimum Sharpe ratio threshold

    # Stop loss settings
    stop_loss_pct: float = 0.02  # Stop loss percentage
    take_profit_pct: float = 0.04  # Take profit percentage

    def to_dict(self) -> ObjectMap:
        """Convert to dictionary."""
        return {
            "max_position_notional": self.max_position_notional,
            "max_single_trade_pct": self.max_single_trade_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_trades_per_hour": self.max_trades_per_hour,
            "min_trade_interval_sec": self.min_trade_interval_sec,
            "max_volatility_pct": self.max_volatility_pct,
            "required_sharpe_ratio": self.required_sharpe_ratio,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
        }

# Protocol definitions
class FeatureCalculator(Protocol):
    """Protocol for feature calculators."""

    def calculate(self, data: object) -> object:
        """Calculate features from input data."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        ...

class TrainerProtocol(Protocol):
    """Protocol for SAC trainers."""

    def train(self) -> bool | ObjectMap:
        """Train the model."""
        ...

class CallbackProtocol(Protocol):
    """Protocol for training callbacks."""

    def __call__(self, locals_: ObjectMap, globals_: ObjectMap) -> None:
        """Callback function."""
        ...

@runtime_checkable
class PerformanceMonitorProtocol(Protocol):
    """Protocol for performance monitors."""

    def record_decision(self, decision: object) -> None:
        """Record a decision."""
        ...

    def get_metrics(self) -> ObjectMap:
        """Get current metrics."""
        ...

@runtime_checkable
class ThresholdManagerProtocol(Protocol):
    """Protocol for threshold managers."""

    def get_adaptive_gates(self) -> MetricsDict:
        """Get adaptive threshold gates."""
        ...

    def update_thresholds(self, evaluation_results: ObjectMap) -> None:
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

    def fit(self, data: object) -> None:
        """Fit normalizer to data."""
        ...

    def transform(self, data: object) -> object:
        """Transform data."""
        ...

    def inverse_transform(self, data: object) -> object:
        """Inverse transform data."""
        ...

@runtime_checkable
class LoggerProtocol(Protocol):
    """Protocol for loggers."""

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        """Log info message."""
        ...

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        """Log error message."""
        ...

# Data Loading Types
@dataclass
class DataSource:
    """Concrete data source definition used in tests and simple loaders.

    This class replaces the protocol-only definition for test convenience.
    For more advanced usage implementers can still provide objects matching
    the earlier protocol shape.
    """

    name: str
    url: str
    data_format: str
    update_frequency: str
    reliability_score: float

    def is_reliable(self, threshold: float = 0.8) -> bool:
        """Return whether the data source is considered reliable based on a threshold."""
        return float(self.reliability_score) >= float(threshold)

# Analysis Types
AnalysisResult = ObjectMap
AnalysisConfig = ConfigSection

class AnalysisInput(TypedDict, total=False):
    """Type for analysis input data."""

    data: object | ObjectMap | ObjectRecords
    config: AnalysisConfig
    metadata: ObjectMap

class AnalysisOutput(TypedDict, total=False):
    """Type for analysis output data."""

    results: AnalysisResult
    summary: ObjectMap
    plots: list[str]  # File paths to generated plots
    reports: list[str]  # File paths to generated reports
    metadata: ObjectMap

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
    portfolio_values: list[float]
    trades_history: ObjectRecords
    performance_metrics: MetricsDict
    trade_log: ObjectRecords
    regime_analysis: ObjectMap
    risk_metrics: MetricsDict
    benchmark_comparison: ObjectMap

class TrainingResult(TypedDict, total=False):
    """Type for training result data."""

    training_stats: ObjectMap
    final_reward: float
    total_timesteps: int
    training_time: float
    action_distribution: MetricsDict
    performance_metrics: ObjectMap

class EvaluationResult(TypedDict, total=False):
    """Type for evaluation result data."""

    metric: str
    value: float
    confidence_interval: tuple[float, float] | None
    benchmark_comparison: float | None
    metadata: ObjectMap

class ComprehensiveEvaluation(TypedDict, total=False):
    """Type for comprehensive evaluation data."""

    model_name: str
    evaluation_type: str
    timestamp: str
    results: dict[str, EvaluationResult]
    summary_stats: ObjectMap
    risk_metrics: ObjectMap
    performance_metrics: ObjectMap
    market_regime_analysis: ObjectMap
    robustness_tests: ObjectMap

# NOTE: This class is a dataclass version that includes methods.
# The TypedDict versions above are kept for backward compatibility with type checking.
@dataclass
class ComprehensiveEvaluationClass:
    """Comprehensive evaluation result with methods.
    
    Designed to work with both:
    - EvaluationMetric (Enum) and EvaluationType (Enum) from unified_evaluation.py
    - str values for TypedDict compatibility
    """

    model_name: str
    evaluation_type: str | Enum  # accepts both str and EvaluationType enum
    timestamp: str | datetime  # accepts both str and datetime
    roi_out_of_sample: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    results: dict[object, object] = field(default_factory=dict)  # Flexible for EvaluationMetric keys
    summary_stats: ObjectMap = field(default_factory=dict)
    risk_metrics: ObjectMap = field(default_factory=dict)
    performance_metrics: ObjectMap = field(default_factory=dict)
    market_regime_analysis: ObjectMap = field(default_factory=dict)
    robustness_tests: ObjectMap = field(default_factory=dict)

    def get_metric_value(self, metric: object, default: float | None = None) -> float | None:
        """指定した指標の値を取得

        Handles results stored with either string keys or Enum keys and supports
        both dict-style and `EvaluationResult` dataclass-style values.
        """
        # Prefer string key lookup (handles Enum input too)
        metric_key = metric.value if hasattr(metric, "value") else str(metric)

        result = self.results.get(metric_key)
        if result is None:
            # Try enum-keyed lookup (tests sometimes use EvaluationMetric as keys)
            result = self.results.get(metric)
            # If still None, try to find an enum key whose value matches the string
            if result is None:
                for k in self.results.keys():
                    if hasattr(k, "value") and k.value == metric_key:
                        result = self.results[k]
                        break

        if result is None:
            return default

        if isinstance(result, dict):
            return result.get("value", default)
        # dataclass or object with `value` attribute
        return getattr(result, "value", default)

    def get_summary_score(self) -> float:
        """総合評価スコアを計算"""
        # Sharpe ratio, Sortino ratio, Calmar ratioの加重平均
        sharpe = self.get_metric_value("sharpe_ratio") or 0
        sortino = self.get_metric_value("sortino_ratio") or 0
        calmar = self.get_metric_value("calmar_ratio") or 0

        # 重み付け: Sharpe 40%, Sortino 30%, Calmar 30%
        score = sharpe * 0.4 + sortino * 0.3 + calmar * 0.3

        # Max drawdown penalty
        max_dd = self.get_metric_value("max_drawdown") or 0
        if max_dd > 0.2:  # 20%以上のドローダウンはペナルティ
            penalty = (max_dd - 0.2) * 2
            score -= penalty

        return max(0, score)  # 負のスコアは0にクリップ

    def to_dict(self) -> ObjectMap:
        """辞書形式に変換

        Converts result keys to strings and serializes `EvaluationResult` objects
        into plain dictionaries for JSON compatibility.
        """
        results_serialized: ObjectMap = {}

        for k, v in self.results.items():
            key_str = k.value if hasattr(k, "value") else str(k)

            if isinstance(v, dict):
                val = v
            else:
                # Support dataclass-like EvaluationResult objects
                val = {
                    "metric": v.metric.value if hasattr(v.metric, "value") else str(v.metric),
                    "value": v.value,
                    "confidence_interval": getattr(v, "confidence_interval", None),
                    "benchmark_comparison": getattr(v, "benchmark_comparison", None),
                    "metadata": getattr(v, "metadata", {}),
                }

            results_serialized[key_str] = val

        return {
            "model_name": self.model_name,
            "evaluation_type": (
                self.evaluation_type.value
                if hasattr(self.evaluation_type, "value")
                else str(self.evaluation_type)
            ),
            "timestamp": (
                self.timestamp.isoformat()
                if hasattr(self.timestamp, "isoformat")
                else str(self.timestamp)
            ),
            "results": results_serialized,
            "summary_stats": self.summary_stats,
            "risk_metrics": self.risk_metrics,
            "performance_metrics": self.performance_metrics,
            "market_regime_analysis": self.market_regime_analysis,
            "robustness_tests": self.robustness_tests,
            "summary_score": float(self.get_summary_score()),
        }

# Path Management Types
class PathConfig(TypedDict, total=False):
    """Configuration for path management."""

    base_dir: Path
    create_dirs: bool
    standard_paths: StringMap

# Validation Types
ValidationResult = dict[str, bool | list[str]]

class ValidationRule(TypedDict, total=False):
    """Type for validation rules."""

    field: str
    required: bool
    type_check: type | None
    range_check: dict[str, int | float] | None
    custom_validator: str | None  # Function name for custom validation

# Error Types
class AnalysisErrorInfo(TypedDict, total=False):
    """Type for error information in analysis."""

    component: str
    error_type: str
    message: str
    context: ObjectMap
    timestamp: str

# Component Configuration Types
class DataLoaderConfig(TypedDict, total=False):
    """Configuration for data loaders."""

    base_path: Path
    required_files: list[str]
    file_patterns: StringMap
    error_handling: str  # 'strict', 'warn', 'ignore'

class AnalyzerConfig(TypedDict, total=False):
    """Configuration for analyzers."""

    name: str
    input_validation: list[ValidationRule]
    output_validation: list[ValidationRule]
    error_handling: str
    logging_level: str

class PathManagerConfig(TypedDict, total=False):
    """Configuration for path managers."""

    base_dir: Path
    create_dirs: bool
    standard_paths: StringMap
    permissions_check: bool

# Factory Types
class ComponentFactory(Protocol[T]):
    """Protocol for component factories."""

    def create(self, config: ConfigSection) -> T:
        """Create a component instance from configuration."""
        ...

# Registry Types
ComponentRegistry = ObjectMap
FactoryRegistry = dict[str, ComponentFactory[object]]

# Utility Types
FileInfo = dict[str, str | int | float]
DirectoryInfo = dict[str, list[FileInfo] | int]

class SystemInfo(TypedDict, total=False):
    """System information for analysis components."""

    python_version: str
    platform: str
    available_memory: int
    cpu_count: int
    component_versions: StringMap

# Risk Management Types
class RiskStatus(TypedDict):
    """Risk status summary."""

    daily_loss: float
    daily_loss_limit: float
    portfolio_value: float
    portfolio_volatility: float
    trades_this_hour: int
    max_trades_per_hour: int
    trailing_stop_level: float | None
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
    statistics: ObjectMap  # 統計情報

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

class PortfolioAnalysisResult(TypedDict, total=False):
    """Portfolio performance analysis result."""

    initial_value: float
    final_value: float
    total_return_pct: float
    total_steps: int
    avg_daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_positive_return: float
    total_negative_return: float

class EpisodeAnalysisResult(TypedDict, total=False):
    """Episode performance analysis result."""

    total_episodes: int
    positive_episodes: int
    negative_episodes: int
    episode_win_rate: float
    avg_episode_reward: float
    best_episode_reward: float
    worst_episode_reward: float
    episode_reward_std: float
    avg_final_portfolio: float
    best_final_portfolio: float
    worst_final_portfolio: float
