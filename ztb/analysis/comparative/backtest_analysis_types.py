"""
Type definitions for backtest analysis.

This module contains shared type definitions used across different
backtest analysis implementations.
"""

from typing import Dict, List, Optional, TypedDict, Union


class RiskMetricsResult(TypedDict, total=False):
    """Aggregated risk metric results."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float


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
    weekday_returns: Dict[str, float]


class MarketConditionResult(TypedDict, total=False):
    """Result of market condition analysis."""

    uptrend: Optional[Dict[str, Union[str, float]]]
    downtrend: Optional[Dict[str, Union[str, float]]]
    sideways: Optional[Dict[str, Union[str, float]]]


class TradingFrequencyResult(TypedDict, total=False):
    """Result of trading frequency analysis."""

    action_distribution: Dict[str, float]
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


class WalkForwardAnalysisResult(TypedDict, total=False):
    """Result of walk-forward efficiency analysis."""

    window_metrics: Dict[str, Dict[str, float]]
    adaptation_analysis: Dict[str, Union[float, int]]


class MicrostructureAnalysisResult(TypedDict):
    """Result of market microstructure analysis."""

    price_impact: Optional[Dict[str, float]]
    market_depth: Optional[Dict[str, float]]
    spread_analysis: Optional[Dict[str, Union[float, str]]]
    behavioral_patterns: Optional[Dict[str, float]]


class SignalGuidanceAnalysisResult(TypedDict, total=False):
    """Analysis result for signal guidance episodes."""

    number_of_signals: int
    average_score: float
    score_std: float
    min_score: float
    max_score: float
    original_hold: int
    original_buy: int
    original_sell: int
    guidance_hold: int
    guidance_buy: int
    guidance_sell: int
    differences: int
    total_actions: int
    difference_pct: float
    correlation: float


class BTCPerformanceResult(TypedDict, total=False):
    """Performs BTC-related tracking metrics."""

    initial_btc: float
    final_btc: float
    net_btc_gained: float
    btc_return_pct: float
    usd_return_pct: float
    btc_vs_usd_performance_ratio: float
    btc_mean_holding: float
    btc_max_holding: float
    btc_min_holding: float
    btc_holding_volatility: float
    btc_positive_changes: int
    btc_negative_changes: int
    btc_trade_frequency: float
    btc_position_stability: float


class AnalysisResult(TypedDict, total=False):
    """Comprehensive backtest analysis result."""

    risk_metrics: RiskMetricsResult
    temporal_patterns: TemporalPatternsResult
    market_conditions: MarketConditionResult
    trading_frequency: TradingFrequencyResult
    btc_analysis: BTCPerformanceResult
    signal_guidance_analysis: SignalGuidanceAnalysisResult
    temporal_analysis: TemporalPatternsResult
    market_condition_analysis: MarketConditionResult
    trading_frequency_analysis: TradingFrequencyResult
    robustness_analysis: RobustnessAnalysisResult
    correlation_analysis: CorrelationAnalysisResult
    transaction_cost_analysis: TransactionCostAnalysisResult
    walk_forward_analysis: WalkForwardAnalysisResult
    microstructure_analysis: MicrostructureAnalysisResult


