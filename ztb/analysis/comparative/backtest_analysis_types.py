"""
Type definitions for backtest analysis.

This module contains shared type definitions used across different
backtest analysis implementations.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union


class NormalityTestResult(TypedDict, total=False):
    """Result of normality statistical tests."""

    shapiro_wilk: Optional[Dict[str, Optional[float]]]
    kolmogorov_smirnov: Dict[str, float]
    jarque_bera: Dict[str, float]
    error: str


class AutocorrelationResult(TypedDict, total=False):
    """Result of autocorrelation analysis."""

    autocorrelations: Dict[str, float]
    ljung_box_test: Union[Dict[str, float], Dict[str, str]]
    error: str


class VolatilityClusteringResult(TypedDict, total=False):
    """Result of volatility clustering analysis."""

    absolute_return_autocorrelation: Dict[str, float]
    rolling_volatility: List[float]
    volatility_persistence: float
    error: str


class RiskAdjustedMetricsResult(TypedDict, total=False):
    """Result of risk-adjusted performance metrics."""

    calmar_ratio: float
    omega_ratio: float
    kappa_ratio: float
    annual_return: float
    max_drawdown: float
    error: str


class StatisticalTestResult(TypedDict, total=False):
    """Result of statistical significance tests."""

    t_test: Dict[str, Union[float, bool]]
    mann_whitney_u: Optional[Dict[str, Optional[Union[float, bool]]]]
    bartlett_test: Optional[Dict[str, Optional[Union[float, bool]]]]
    error: str


class TemporalPatternsResult(TypedDict, total=False):
    """Result of temporal pattern analysis."""

    hourly_returns: Dict[str, float]
    weekday_returns: Dict[str, str]


class MarketConditionResult(TypedDict):
    """Result of market condition analysis."""

    uptrend: Optional[Dict[str, Union[str, float]]]
    downtrend: Optional[Dict[str, Union[str, float]]]
    sideways: Optional[Dict[str, Union[str, float]]]


class TradingFrequencyResult(TypedDict, total=False):
    """Result of trading frequency analysis."""

    action_distribution: Dict[int, int]
    trade_frequency: float
    avg_trade_interval: float
    min_trade_interval: float
    max_trade_interval: float
    total_trades: int


class ActionAveragesResult(TypedDict, total=False):
    """Result of action averages analysis."""

    action_mean: float
    action_std: float
    action_median: float
    action_mode: float
    action_trend: float
    action_cv: float
    action_skewness: float
    action_kurtosis: float
    most_common_transition: str
    transition_frequency: int
    total_transitions: int


class PerformanceMetricsResult(TypedDict):
    """Standard performance metrics result."""

    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int


class RegimeAnalysisResult(TypedDict):
    """Result of regime-based performance analysis."""

    high_volatility_performance: PerformanceMetricsResult
    low_volatility_performance: PerformanceMetricsResult
    volatility_regime_consistency: float


class SeasonalAnalysisResult(TypedDict):
    """Result of seasonal performance analysis."""

    hourly_performance: Dict[str, Optional[PerformanceMetricsResult]]
    weekday_performance: Dict[str, Optional[PerformanceMetricsResult]]
    monthly_performance: Dict[str, Optional[PerformanceMetricsResult]]
    seasonal_consistency_score: float


class RobustnessAnalysisResult(TypedDict):
    """Comprehensive robustness analysis result."""

    overall_performance: PerformanceMetricsResult
    volatility_analysis: Optional[RegimeAnalysisResult]
    trend_analysis: Optional[RegimeAnalysisResult]
    drawdown_analysis: Optional[RegimeAnalysisResult]
    seasonal_analysis: Optional[SeasonalAnalysisResult]
    robustness_score: float


class CorrelationAnalysisResult(TypedDict):
    """Result of correlation and dependency analysis."""

    price_portfolio_correlation: float
    lag_correlations: Dict[str, float]
    beta: float
    action_price_relationships: Dict[str, Dict[str, float]]


class TransactionCostAnalysisResult(TypedDict):
    """Result of transaction cost impact analysis."""

    total_transaction_cost: float
    average_cost_per_trade: float
    cost_to_return_ratio: float
    trades_per_step: float
    cost_efficiency_score: float


class StressTestResult(TypedDict):
    """Result of stress testing under various market conditions."""

    price_drop_10pct: Dict[str, Union[float, int]]
    price_drop_20pct: Dict[str, Union[float, int]]
    price_drop_30pct: Dict[str, Union[float, int]]
    high_volatility: Dict[str, Union[float, int]]


class WalkForwardAnalysisResult(TypedDict):
    """Result of walk-forward efficiency analysis."""

    window_analysis: Optional[Dict[str, List[PerformanceMetricsResult]]]
    adaptation_analysis: Optional[Dict[str, Union[float, int]]]


class MicrostructureAnalysisResult(TypedDict):
    """Result of market microstructure analysis."""

    price_impact: Optional[Dict[str, float]]
    market_depth: Optional[Dict[str, float]]
    spread_analysis: Optional[Dict[str, Union[float, str]]]
    behavioral_patterns: Optional[Dict[str, float]]


class AnalysisResult(TypedDict, total=False):
    """Comprehensive backtest analysis result."""

    risk_metrics: Dict[str, float]
    temporal_patterns: TemporalPatternsResult
    market_conditions: MarketConditionResult
    trading_frequency: TradingFrequencyResult
    temporal_analysis: Union[
        Dict[str, Any], TemporalPatternsResult
    ]  # Keep for enhanced stats
    market_condition_analysis: Union[
        Dict[str, Any], MarketConditionResult
    ]  # Keep for enhanced stats
    trading_frequency_analysis: Union[
        Dict[str, Any], TradingFrequencyResult
    ]  # Keep for enhanced stats
    robustness_analysis: RobustnessAnalysisResult
    correlation_analysis: CorrelationAnalysisResult
    transaction_cost_analysis: TransactionCostAnalysisResult
    walk_forward_analysis: WalkForwardAnalysisResult
    microstructure_analysis: MicrostructureAnalysisResult
