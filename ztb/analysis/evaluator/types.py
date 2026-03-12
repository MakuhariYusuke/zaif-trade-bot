#!/usr/bin/env python3
"""
Type definitions for evaluation module.
"""

from typing import Any, TypedDict

class SingleEpisodeResultDict(TypedDict):
    """Single episode evaluation result."""

    rewards: list[float]
    positions: list[float]
    pnls: list[float]
    actions: list[int]
    states: list[Any]
    portfolio_history: list[float]
    price_history: list[float]
    timestamps: list[Any]

class EvaluationResult(TypedDict, total=False):
    """Type definition for comprehensive evaluation results.

    Contains all metrics and statistics from model evaluation including
    performance metrics, risk metrics, and trading statistics.
    """

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float

    # Risk metrics
    volatility: float
    sortino_ratio: float
    calmar_ratio: float
    expected_value: float
    recovery_factor: float
    var_95: float
    cvar_95: float

    # Trading statistics
    total_pnl: float
    gross_profit: float
    gross_loss: float
    largest_win: float
    largest_loss: float
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int

    # Advanced analysis
    seasonality_analysis: dict[str, Any]
    market_regime_analysis: dict[str, Any]
    walkforward_analysis: dict[str, Any]
    stress_test_analysis: dict[str, Any]

    # Episode data
    rewards: list[float]
    positions: list[float]
    pnls: list[float]
    actions: list[int]
    states: list[Any]
    action_history: list[int]
    portfolio_history: list[float]
    price_history: list[float]
    timestamps: list[Any]
    trade_pnls: list[float]
    continuous_action_stats: dict[str, Any]

    # Model info
    model_path: str
    evaluation_config: dict[str, Any]
