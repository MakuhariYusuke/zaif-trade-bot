"""
Common Utilities for Signal Processing

This module provides shared utility functions to reduce code duplication
across signal processing components.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_market_data(data: pd.DataFrame,
                        required_columns: List[str] = None) -> bool:
    """
    Validate market data DataFrame

    Args:
        data: Market data DataFrame
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if data is None or data.empty:
        logger.error("Market data is None or empty")
        return False

    default_required = ['open', 'high', 'low', 'close']
    required_cols = required_columns or default_required

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # Check for NaN values in required columns
    for col in required_cols:
        if data[col].isna().any():
            logger.warning(f"NaN values found in column: {col}")

    return True


def calculate_returns(data: pd.DataFrame,
                     method: str = 'pct_change') -> pd.Series:
    """
    Calculate price returns

    Args:
        data: Market data DataFrame
        method: Return calculation method ('pct_change', 'log')

    Returns:
        Returns series
    """
    if method == 'log':
        returns = np.log(data['close'] / data['close'].shift(1))
    else:
        returns = data['close'].pct_change()

    return returns.fillna(0)


def calculate_volatility(returns: pd.Series,
                        window: int = 20,
                        method: str = 'std') -> float:
    """
    Calculate volatility from returns

    Args:
        returns: Returns series
        window: Rolling window size
        method: Volatility calculation method

    Returns:
        Current volatility value
    """
    if method == 'std':
        vol = returns.rolling(window).std()
    elif method == 'var':
        vol = returns.rolling(window).var()
    elif method == 'atr':
        # Simplified ATR calculation
        high_low = returns.rolling(window).max() - returns.rolling(window).min()
        vol = high_low.rolling(window).mean()
    else:
        raise ValueError(f"Unknown volatility method: {method}")

    return vol.iloc[-1] if not vol.empty else 0.0


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0

    Args:
        weights: Weight dictionary

    Returns:
        Normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        # Equal weights if all are zero
        num_weights = len(weights)
        return {k: 1.0/num_weights for k in weights.keys()}

    return {k: v/total for k, v in weights.items()}


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to specified range

    Args:
        value: Input value
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def calculate_confidence_score(score: float,
                              thresholds: Dict[str, float],
                              method: str = 'distance') -> float:
    """
    Calculate confidence score based on distance from thresholds

    Args:
        score: Quality score (0-100)
        thresholds: Threshold dictionary
        method: Confidence calculation method

    Returns:
        Confidence score (0-1)
    """
    buy_threshold = thresholds.get('buy', 75)
    sell_threshold = thresholds.get('sell', 25)

    if method == 'distance':
        if score >= buy_threshold:
            distance = score - buy_threshold
            max_distance = 100 - buy_threshold
        elif score <= sell_threshold:
            distance = sell_threshold - score
            max_distance = sell_threshold
        else:
            # In hold zone
            dist_to_buy = abs(score - buy_threshold)
            dist_to_sell = abs(score - sell_threshold)
            distance = min(dist_to_buy, dist_to_sell)
            max_distance = (buy_threshold - sell_threshold) / 2

        confidence = clamp_value(distance / max_distance, 0.0, 1.0)

    elif method == 'probability':
        # Simple sigmoid-based confidence
        if score >= buy_threshold:
            confidence = 1 / (1 + np.exp(-(score - buy_threshold) / 10))
        elif score <= sell_threshold:
            confidence = 1 / (1 + np.exp(-(sell_threshold - score) / 10))
        else:
            confidence = 0.5

    else:
        raise ValueError(f"Unknown confidence method: {method}")

    return confidence


def smooth_series(series: pd.Series,
                 method: str = 'ema',
                 span: int = 5) -> pd.Series:
    """
    Smooth time series data

    Args:
        series: Input series
        method: Smoothing method ('ema', 'sma', 'median')
        span: Smoothing window/span

    Returns:
        Smoothed series
    """
    if method == 'ema':
        return series.ewm(span=span).mean()
    elif method == 'sma':
        return series.rolling(span).mean()
    elif method == 'median':
        return series.rolling(span).median()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def detect_outliers(series: pd.Series,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers in series

    Args:
        series: Input series
        method: Outlier detection method
        threshold: Outlier threshold

    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        mean_val = series.mean()
        std_val = series.std()
        z_scores = abs((series - mean_val) / std_val)
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def resample_data(data: pd.DataFrame,
                 target_freq: str,
                 method: str = 'last') -> pd.DataFrame:
    """
    Resample time series data

    Args:
        data: Input DataFrame
        target_freq: Target frequency (e.g., '1H', '1D')
        method: Resampling method ('last', 'first', 'mean')

    Returns:
        Resampled DataFrame
    """
    if method == 'last':
        resampled = data.resample(target_freq).last()
    elif method == 'first':
        resampled = data.resample(target_freq).first()
    elif method == 'mean':
        resampled = data.resample(target_freq).mean()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return resampled.dropna()


def calculate_correlation_matrix(data: pd.DataFrame,
                               columns: List[str] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns

    Args:
        data: Input DataFrame
        columns: Columns to include (default: numeric columns)

    Returns:
        Correlation matrix
    """
    if columns is None:
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = data[columns]

    return numeric_data.corr()


def find_optimal_weights(returns: pd.DataFrame,
                        target_return: float = None,
                        method: str = 'equal') -> Dict[str, float]:
    """
    Find optimal weights for portfolio/ensemble

    Args:
        returns: Returns DataFrame for different components
        target_return: Target return (for optimization)
        method: Weight optimization method

    Returns:
        Optimal weights dictionary
    """
    if method == 'equal':
        n_assets = len(returns.columns)
        weights = {col: 1.0/n_assets for col in returns.columns}

    elif method == 'risk_parity':
        # Simplified risk parity
        vols = returns.std()
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        weights = weights.to_dict()

    elif method == 'mean_variance':
        # Simple mean-variance optimization
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Minimize variance (simplified)
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones(len(returns.columns))
            weights_array = inv_cov @ ones
            weights_array = weights_array / weights_array.sum()
            weights = dict(zip(returns.columns, weights_array))
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            n_assets = len(returns.columns)
            weights = {col: 1.0/n_assets for col in returns.columns}

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return weights