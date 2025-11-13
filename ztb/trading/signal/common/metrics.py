"""
Common Metrics for Signal Processing

This module provides standardized metric calculations used across
signal processing components.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_trend_metrics(data: pd.DataFrame,
                          window: int = 20) -> Dict[str, float]:
    """
    Calculate comprehensive trend metrics

    Args:
        data: Market data DataFrame
        window: Analysis window

    Returns:
        Dictionary of trend metrics
    """
    if len(data) < window:
        return {
            'trend_strength': 0.0,
            'trend_direction': 0.0,
            'trend_slope': 0.0,
            'trend_consistency': 0.0
        }

    close = data['close'].tail(window)

    # Linear regression for trend
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close.values, 1)
    trend_line = slope * x + intercept

    # Trend strength (R-squared)
    ss_res = np.sum((close.values - trend_line) ** 2)
    ss_tot = np.sum((close.values - np.mean(close.values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Trend direction (-1 to 1)
    trend_direction = np.sign(slope)

    # Trend consistency (how well data follows trend)
    trend_consistency = r_squared * trend_direction

    # Normalized trend strength
    price_range = close.max() - close.min()
    if price_range > 0:
        normalized_slope = slope / (price_range / window)
        trend_strength = abs(normalized_slope)
    else:
        trend_strength = 0.0

    # Bull/Bear strength calculations
    if slope > 0:
        bull_strength = trend_strength * r_squared
        bear_strength = 0.0
    else:
        bull_strength = 0.0
        bear_strength = trend_strength * r_squared

    return {
        'trend_strength': float(trend_strength),
        'trend_direction': float(trend_direction),
        'trend_slope': float(slope),
        'trend_consistency': float(trend_consistency),
        'r_squared': float(r_squared),
        'bull_strength': float(bull_strength),
        'bear_strength': float(bear_strength)
    }


def calculate_momentum_metrics(data: pd.DataFrame,
                             short_window: int = 10,
                             long_window: int = 20) -> Dict[str, float]:
    """
    Calculate momentum-based metrics

    Args:
        data: Market data DataFrame
        short_window: Short-term momentum window
        long_window: Long-term momentum window

    Returns:
        Dictionary of momentum metrics
    """
    if len(data) < long_window:
        return {
            'momentum': 0.0,
            'momentum_acceleration': 0.0,
            'momentum_divergence': 0.0
        }

    close = data['close']

    # Short-term momentum
    short_momentum = (close - close.shift(short_window)) / close.shift(short_window)

    # Long-term momentum
    long_momentum = (close - close.shift(long_window)) / close.shift(long_window)

    # Current momentum
    momentum = short_momentum.iloc[-1] if not short_momentum.empty else 0.0

    # Momentum acceleration (rate of change of momentum)
    momentum_acceleration = short_momentum.diff().iloc[-1] if len(short_momentum) > 1 else 0.0

    # Momentum divergence (short vs long term)
    if not long_momentum.empty and long_momentum.iloc[-1] != 0:
        momentum_divergence = momentum / long_momentum.iloc[-1]
    else:
        momentum_divergence = 0.0

    return {
        'momentum': float(momentum),
        'momentum_acceleration': float(momentum_acceleration),
        'momentum_divergence': float(momentum_divergence),
        'short_momentum': float(short_momentum.iloc[-1]) if not short_momentum.empty else 0.0,
        'long_momentum': float(long_momentum.iloc[-1]) if not long_momentum.empty else 0.0
    }


def calculate_volatility_metrics(data: pd.DataFrame,
                               window: int = 20) -> Dict[str, float]:
    """
    Calculate volatility-based metrics

    Args:
        data: Market data DataFrame
        window: Analysis window

    Returns:
        Dictionary of volatility metrics
    """
    if len(data) < window:
        return {
            'volatility': 0.0,
            'volatility_trend': 0.0,
            'volatility_regime': 'unknown'
        }

    # Price returns
    returns = data['close'].pct_change().fillna(0)

    # Rolling volatility
    volatility = returns.rolling(window).std()

    # Volatility trend
    vol_trend = volatility.diff().iloc[-1] if len(volatility) > 1 else 0.0

    # Current volatility
    current_vol = volatility.iloc[-1] if not volatility.empty else 0.0

    # Volatility regime classification
    vol_percentile = volatility.rank(pct=True).iloc[-1]
    if vol_percentile > 0.8:
        vol_regime = 'high'
    elif vol_percentile < 0.2:
        vol_regime = 'low'
    else:
        vol_regime = 'normal'

    # ATR (Average True Range) approximation
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift(1)).abs()
    low_close = (data['low'] - data['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()

    return {
        'volatility': float(current_vol),
        'volatility_trend': float(vol_trend),
        'volatility_regime': vol_regime,
        'atr': float(atr.iloc[-1]) if not atr.empty else 0.0,
        'volatility_percentile': float(vol_percentile)
    }


def calculate_volume_metrics(data: pd.DataFrame,
                           window: int = 20) -> Dict[str, float]:
    """
    Calculate volume-based metrics

    Args:
        data: Market data DataFrame
        window: Analysis window

    Returns:
        Dictionary of volume metrics
    """
    if 'volume' not in data.columns or len(data) < window:
        return {
            'volume_trend': 0.0,
            'volume_price_trend': 0.0,
            'volume_intensity': 0.0
        }

    volume = data['volume']
    close = data['close']

    # Volume trend
    volume_ma = volume.rolling(window).mean()
    volume_trend = (volume / volume_ma).iloc[-1] if not volume_ma.empty else 1.0

    # Volume-price trend (correlation)
    volume_returns = volume.pct_change().fillna(0)
    price_returns = close.pct_change().fillna(0)

    correlation = volume_returns.rolling(window).corr(price_returns)
    volume_price_trend = correlation.iloc[-1] if not correlation.empty else 0.0

    # Volume intensity (relative to price movement)
    price_volatility = price_returns.rolling(window).std()
    volume_intensity = volume_trend * price_volatility.iloc[-1] if not price_volatility.empty else 0.0

    return {
        'volume_trend': float(volume_trend),
        'volume_price_trend': float(volume_price_trend),
        'volume_intensity': float(volume_intensity),
        'volume_ma_ratio': float(volume_trend)
    }


def calculate_support_resistance_metrics(data: pd.DataFrame,
                                       window: int = 20) -> Dict[str, float]:
    """
    Calculate support and resistance metrics

    Args:
        data: Market data DataFrame
        window: Analysis window

    Returns:
        Dictionary of support/resistance metrics
    """
    if len(data) < window:
        return {
            'support_strength': 0.0,
            'resistance_strength': 0.0,
            'sr_position': 0.5
        }

    high = data['high'].tail(window)
    low = data['low'].tail(window)
    close = data['close'].tail(window)

    # Rolling support and resistance levels
    support_level = low.rolling(window).min()
    resistance_level = high.rolling(window).max()

    # Current position relative to S/R levels
    current_price = close.iloc[-1]
    support = support_level.iloc[-1]
    resistance = resistance_level.iloc[-1]

    if resistance > support:
        sr_position = (current_price - support) / (resistance - support)
    else:
        sr_position = 0.5

    # Strength based on how long levels have held
    support_touches = sum(low == support_level)
    resistance_touches = sum(high == resistance_level)

    support_strength = min(1.0, support_touches / window)
    resistance_strength = min(1.0, resistance_touches / window)

    return {
        'support_strength': float(support_strength),
        'resistance_strength': float(resistance_strength),
        'sr_position': float(sr_position),
        'support_level': float(support),
        'resistance_level': float(resistance)
    }


def calculate_composite_score(indicator_scores: Dict[str, float],
                            weights: Dict[str, float]) -> float:
    """
    Calculate composite score from multiple indicators

    Args:
        indicator_scores: Dictionary of individual indicator scores
        weights: Dictionary of weights for each indicator

    Returns:
        Composite score (0-100)
    """
    if not indicator_scores:
        return 50.0

    weighted_sum = 0.0
    total_weight = 0.0

    for indicator, score in indicator_scores.items():
        weight = weights.get(indicator, 1.0)
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 50.0

    composite = weighted_sum / total_weight

    # Ensure result is in 0-100 range
    return max(0.0, min(100.0, composite))


def calculate_signal_quality_metrics(predictions: List[int],
                                   actual_returns: List[float],
                                   confidence_scores: List[float] = None) -> Dict[str, float]:
    """
    Calculate signal quality metrics

    Args:
        predictions: List of predicted actions (-1, 0, 1)
        actual_returns: List of actual returns
        confidence_scores: List of confidence scores (optional)

    Returns:
        Dictionary of quality metrics
    """
    if len(predictions) != len(actual_returns):
        raise ValueError("Predictions and returns must have same length")

    # Basic accuracy metrics
    correct_predictions = 0
    total_predictions = len(predictions)

    for pred, ret in zip(predictions, actual_returns):
        if pred == 0:  # HOLD
            continue
        elif (pred > 0 and ret > 0) or (pred < 0 and ret < 0):
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Directional accuracy (excluding HOLD)
    directional_preds = [p for p in predictions if p != 0]
    directional_returns = [r for p, r in zip(predictions, actual_returns) if p != 0]

    directional_correct = 0
    for pred, ret in zip(directional_preds, directional_returns):
        if (pred > 0 and ret > 0) or (pred < 0 and ret < 0):
            directional_correct += 1

    directional_accuracy = directional_correct / len(directional_preds) if directional_preds else 0.0

    # Profitability metrics
    profitable_trades = sum(1 for p, r in zip(predictions, actual_returns)
                          if (p > 0 and r > 0) or (p < 0 and r < 0))
    profitability = profitable_trades / total_predictions if total_predictions > 0 else 0.0

    # Sharpe-like ratio for signals
    returns_array = np.array(actual_returns)
    if len(returns_array) > 1:
        signal_sharpe = returns_array.mean() / returns_array.std() if returns_array.std() > 0 else 0.0
    else:
        signal_sharpe = 0.0

    metrics = {
        'accuracy': accuracy,
        'directional_accuracy': directional_accuracy,
        'profitability': profitability,
        'signal_sharpe_ratio': signal_sharpe,
        'total_signals': total_predictions,
        'directional_signals': len(directional_preds)
    }

    # Confidence-weighted metrics
    if confidence_scores:
        if len(confidence_scores) == len(predictions):
            # Confidence-weighted accuracy
            weighted_correct = sum(c for p, r, c in zip(predictions, actual_returns, confidence_scores)
                                 if ((p > 0 and r > 0) or (p < 0 and r < 0)))
            total_confidence = sum(confidence_scores)
            confidence_weighted_accuracy = weighted_correct / total_confidence if total_confidence > 0 else 0.0

            metrics['confidence_weighted_accuracy'] = confidence_weighted_accuracy

    return metrics