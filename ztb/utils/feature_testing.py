"""
Feature testing utilities for ZTB.

This module provides functions for evaluating and testing trading features
against various strategies and calculating performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def calculate_trading_metrics(
    signals: pd.Series,
    returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate comprehensive trading metrics from signals and returns.

    Args:
        signals: Trading signals (-1, 0, 1)
        returns: Asset returns

    Returns:
        Dictionary of trading metrics
    """
    valid_idx = signals.notna() & returns.notna() & (signals != 0)
    if valid_idx.sum() == 0:
        return {
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sample_count': 0
        }

    strategy_returns = signals[valid_idx] * returns[valid_idx]
    cumulative = (1 + strategy_returns).cumprod()

    # Win rate
    win_rate = (strategy_returns > 0).mean()

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe ratio
    if strategy_returns.std() > 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Sortino ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = strategy_returns.mean() / downside_returns.std() * np.sqrt(252)
    else:
        sortino_ratio = 0.0

    # Calmar ratio
    if abs(max_drawdown) > 0:
        calmar_ratio = strategy_returns.mean() * 252 / abs(max_drawdown)
    else:
        calmar_ratio = 0.0

    return {
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'sample_count': int(valid_idx.sum())
    }


def generate_feature_signals(
    feature_data: pd.Series,
    feature_name: str
) -> pd.Series:
    """
    Generate trading signals based on feature values using feature-specific strategies.

    Args:
        feature_data: Feature values
        feature_name: Name of the feature

    Returns:
        Trading signals (-1, 0, 1)
    """
    signals = pd.Series(0, index=feature_data.index)

    # Feature-specific strategies
    if feature_name == 'RSI':
        # RSI strategy: buy when RSI < 30, sell when RSI > 70
        signals[feature_data < 30] = 1  # Buy signal
        signals[feature_data > 70] = -1  # Sell signal
    elif feature_name == 'ROC':
        # ROC strategy: buy when ROC > 5, sell when ROC < -5
        signals[feature_data > 5] = 1
        signals[feature_data < -5] = -1
    elif feature_name == 'OBV':
        # OBV strategy: buy when OBV increasing, sell when decreasing
        obv_change = feature_data.diff()
        # Type ignore for pandas boolean indexing
        signals[obv_change > 0] = 1  # type: ignore
        signals[obv_change < 0] = -1  # type: ignore
    elif feature_name == 'ZScore':
        # ZScore strategy: buy when ZScore < -1, sell when ZScore > 1 (mean reversion)
        signals[feature_data < -1] = 1
        signals[feature_data > 1] = -1
    elif 'MACD' in feature_name:
        # MACD strategy: buy when MACD > signal, sell when MACD < signal
        signals[feature_data > 0] = 1  # Simplified: assume signal is 0
        signals[feature_data < 0] = -1
    elif 'Stochastic' in feature_name:
        # Stochastic strategy: buy when %K < 20, sell when %K > 80
        signals[feature_data < 20] = 1
        signals[feature_data > 80] = -1
    elif 'CCI' in feature_name:
        # CCI strategy: buy when CCI < -100, sell when CCI > 100
        signals[feature_data < -100] = 1
        signals[feature_data > 100] = -1
    elif 'Bollinger' in feature_name:
        # Bollinger strategy: buy when price < lower band, sell when price > upper band
        signals[feature_data < -1] = 1  # Simplified band position
        signals[feature_data > 1] = -1
    else:
        # Default strategy: buy when feature > 0, sell when feature < 0
        signals = (feature_data > 0).astype(int) - (feature_data < 0).astype(int)

    return signals


def evaluate_feature_performance(
    feature_data: pd.Series,
    price_data: pd.Series,
    feature_name: str
) -> Dict[str, Any]:
    """
    Evaluate feature performance using appropriate trading strategy.

    Args:
        feature_data: Feature values
        price_data: Price data for calculating returns
        feature_name: Name of the feature

    Returns:
        Dictionary containing signals and performance metrics
    """
    # Generate signals
    signals = generate_feature_signals(feature_data, feature_name)

    # Calculate returns
    returns = price_data.pct_change().shift(-1)  # Next period returns

    # Calculate metrics
    metrics = calculate_trading_metrics(signals, returns)

    return {
        'feature_name': feature_name,
        'signals': signals,
        'metrics': metrics
    }