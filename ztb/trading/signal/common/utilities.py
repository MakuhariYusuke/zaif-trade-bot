"""
Common Utilities for Signal Processing

This module provides shared utility functions to reduce code duplication
across signal processing components.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def validate_market_data(
    data: pd.DataFrame, required_columns: List[str] = None
) -> bool:
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

    default_required = ["open", "high", "low", "close"]
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


def calculate_returns(data: pd.DataFrame, method: str = "pct_change") -> pd.Series:
    """
    Calculate price returns

    Args:
        data: Market data DataFrame
        method: Return calculation method ('pct_change', 'log')

    Returns:
        Returns series
    """
    if method == "log":
        # np.log returns a Series if input is Series, but mypy might infer ndarray
        log_returns = np.log(data["close"] / data["close"].shift(1))
        return pd.Series(log_returns).fillna(0)
    else:
        pct_returns = data["close"].pct_change()
        return pct_returns.fillna(0)


def calculate_volatility(
    returns: pd.Series, window: int = 20, method: str = "std"
) -> float:
    """
    Calculate volatility from returns

    Args:
        returns: Returns series
        window: Rolling window size
        method: Volatility calculation method

    Returns:
        Current volatility value
    """
    if method == "std":
        vol = returns.rolling(window).std()
    elif method == "var":
        vol = returns.rolling(window).var()
    elif method == "atr":
        # Simplified ATR calculation
        high_low = returns.rolling(window).max() - returns.rolling(window).min()
        vol = high_low.rolling(window).mean()
    else:
        raise ValueError(f"Unknown volatility method: {method}")

    return vol.iloc[-1] if not vol.empty else 0.0


def calculate_volatility_from_prices(
    prices: pd.Series,
    window: int = 20,
    returns_method: str = "pct_change",
    annualize: bool = False,
) -> float:
    """
    Convenience wrapper: accepts a series of prices and calculates volatility.

    Args:
        prices: Series of price values
        window: Rolling window size (default 20)
        returns_method: 'pct_change' or 'log'
        annualize: If True, convert to annualized volatility (assumes daily data ~252 trading days)

    Returns:
        Volatility value (float). For annualize=True, volatility is annualized (e.g., sigma * sqrt(252)).
    """
    if prices is None or prices.empty:
        return 0.0

    if returns_method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()

    vol = calculate_volatility(returns, window=window)
    if annualize:
        # Standard daily scaling by sqrt(252)
        try:
            vol = float(vol) * (252**0.5)
        except Exception:
            pass
    return float(vol)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0

    Args:
        weights: Weight dictionary

    Returns:
        Normalized weights
    """
    # Treat negative weights as zero
    clipped = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clipped.values())
    if total == 0:
        # If all weights are zero, return zeros with same keys
        return {k: 0.0 for k in weights.keys()}
    return {k: v / total for k, v in clipped.items()}


def update_weights_with_dynamic_adjustment(
    weights: Dict[str, float], adjustment: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Apply optional adjustments to weights before normalizing.

    Args:
        weights: Base weight dict
        adjustment: Optional adjustment dict with same keys mapping to multiplier

    Returns:
        Normalized, adjusted weights
    """
    if adjustment is None:
        return normalize_weights(weights)

    adjusted = {
        k: float(weights.get(k, 0.0)) * float(adjustment.get(k, 1.0))
        for k in weights.keys()
    }
    return normalize_weights(adjusted)


def confidence_to_score_thresholds(
    confidence: float,
    default_buy: float = 75.0,
    default_sell: float = 25.0,
    min_gap: float = 10.0,
    buy_min: float = None,
    buy_max: float = None,
    sell_min: float = None,
    sell_max: float = None,
) -> Tuple[float, float]:
    """
    Map a confidence threshold (0..1) to buy/sell score thresholds (0..100) with clamping
    and a minimum gap to preserve a HOLD zone.

    Args:
        confidence: Confidence threshold in 0..1
        default_buy: Default BUY threshold (0..100)
        default_sell: Default SELL threshold (0..100)
        min_gap: Minimum difference between buy and sell thresholds (in score points)
        buy_min/buy_max/sell_min/sell_max: optional clamp overrides

    Returns:
        Tuple (buy_threshold, sell_threshold)
    """
    # Safety: ensure 0..1
    c = max(0.0, min(1.0, float(confidence)))

    # Base mapping
    buy = c * 100.0
    sell = (1.0 - c) * 100.0

    # Apply default clamping ranges if not provided
    buy_min_val = 0.0 if buy_min is None else max(0.0, float(buy_min))
    buy_max_val = 100.0 if buy_max is None else min(100.0, float(buy_max))
    sell_min_val = 0.0 if sell_min is None else max(0.0, float(sell_min))
    sell_max_val = 100.0 if sell_max is None else min(100.0, float(sell_max))

    # Fallback default ranges centered around existing defaults
    if buy_min is None:
        buy_min_val = max(0.0, default_buy - 20.0)
    if buy_max is None:
        buy_max_val = min(100.0, default_buy + 20.0)
    if sell_min is None:
        sell_min_val = max(0.0, default_sell - 20.0)
    if sell_max is None:
        sell_max_val = min(100.0, default_sell + 20.0)

    # Clamp to ranges
    buy = clamp_value(buy, buy_min_val, buy_max_val)
    sell = clamp_value(sell, sell_min_val, sell_max_val)

    # Ensure min_gap between buy and sell
    try:
        mg = float(min_gap)
    except Exception:
        mg = 10.0

    if buy < sell + mg:
        # Move them apart around the midpoint
        mid = (buy + sell) / 2.0
        buy = min(buy_max_val, max(buy_min_val, mid + mg / 2.0))
        sell = min(sell_max_val, max(sell_min_val, mid - mg / 2.0))

    # Final clamp to 0..100
    buy = clamp_value(buy, 0.0, 100.0)
    sell = clamp_value(sell, 0.0, 100.0)

    return float(buy), float(sell)


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


def calculate_confidence_score(
    score: float, thresholds: Dict[str, float] = None, method: str = "linear"
) -> float:
    """
    Calculate confidence score based on distance from thresholds

    Args:
        score: Quality score (0-100)
        thresholds: Threshold dictionary
        method: Confidence calculation method

    Returns:
        Confidence score (0-1)
    """
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = {"buy": 75, "sell": 25}
    buy_threshold = thresholds.get("buy", 75)
    sell_threshold = thresholds.get("sell", 25)

    if method == "linear":
        # Map 0-100 score linearly to 0-1 confidence
        score_clamped = max(0.0, min(100.0, float(score)))
        return clamp_value(score_clamped / 100.0, 0.0, 1.0)

    if method == "distance":
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

    elif method == "probability":
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


def score_to_discrete_action(
    score: float,
    buy_threshold: float = 75.0,
    sell_threshold: float = 25.0,
    high_score_is_buy: bool = True,
) -> int:
    """
    Convert score (0-100) to discrete action with consistent parity support

    Args:
        score: Score 0-100
        buy_threshold: Score threshold for BUY
        sell_threshold: Score threshold for SELL
        high_score_is_buy: If True, high score implies BUY. If False, high score implies SELL.

    Returns:
        int: 1 (BUY), -1 (SELL), 0 (HOLD)
    """
    # Ensure score is within 0-100
    s = clamp_value(float(score), 0.0, 100.0)

    if high_score_is_buy:
        if s >= buy_threshold:
            return 1
        if s <= sell_threshold:
            return -1
        return 0
    else:
        # Inverted parity: high scores mean SELL, low scores mean BUY
        if s >= buy_threshold:
            return -1
        if s <= sell_threshold:
            return 1
        return 0


def get_dynamic_thresholds(
    confidence: float = 0.7,
    threshold_manager: Optional[Any] = None,
    market_data: Optional[pd.DataFrame] = None,
    min_gap: float = 10.0,
    default_buy: float = 75.0,
    default_sell: float = 25.0,
) -> Tuple[float, float]:
    """
    Obtain buy/sell score thresholds dynamically.

    Behavior:
        - If `threshold_manager` is provided and market_data is available, use it to compute
            adaptive confidence/strength thresholds and map them to scores.
        - Otherwise, map the provided `confidence` directly into 0..100 scaled buy/sell thresholds
            while preserving a hold region using min_gap.

    Returns:
        Tuple[buy_threshold, sell_threshold]
    """
    # Best-effort use the ThresholdManager when available
    try:
        if threshold_manager is not None and market_data is not None:
            # Expect threshold_manager.calculate_adaptive_signal_thresholds to return confidence-like threshold
            adaptive = threshold_manager.calculate_adaptive_signal_thresholds(
                market_data,
                base_confidence=confidence,
                base_strength=0.5,
            )
            conf_thresh = float(adaptive.get("confidence_threshold", confidence))
            buy, sell = confidence_to_score_thresholds(
                conf_thresh,
                default_buy=default_buy,
                default_sell=default_sell,
                min_gap=min_gap,
            )
            return buy, sell
    except Exception:
        logger.warning(
            "ThresholdManager failed to calculate adaptive thresholds; falling back to static mapping"
        )

    # Fallback: static mapping
    return confidence_to_score_thresholds(
        confidence, default_buy=default_buy, default_sell=default_sell, min_gap=min_gap
    )


def smooth_series(series: pd.Series, method: str = "ema", span: int = 5) -> pd.Series:
    """
    Smooth time series data

    Args:
        series: Input series
        method: Smoothing method ('ema', 'sma', 'median')
        span: Smoothing window/span

    Returns:
        Smoothed series
    """
    if method == "ema":
        return series.ewm(span=span).mean()
    elif method == "sma":
        return series.rolling(span).mean()
    elif method == "median":
        return series.rolling(span).median()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def detect_outliers(
    series: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in series

    Args:
        series: Input series
        method: Outlier detection method
        threshold: Outlier threshold

    Returns:
        Boolean series indicating outliers
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == "zscore":
        mean_val = series.mean()
        std_val = series.std()
        z_scores = abs((series - mean_val) / std_val)
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def resample_data(
    data: pd.DataFrame, target_freq: str, method: str = "last"
) -> pd.DataFrame:
    """
    Resample time series data

    Args:
        data: Input DataFrame
        target_freq: Target frequency (e.g., '1H', '1D')
        method: Resampling method ('last', 'first', 'mean')

    Returns:
        Resampled DataFrame
    """
    if method == "last":
        resampled = data.resample(target_freq).last()
    elif method == "first":
        resampled = data.resample(target_freq).first()
    elif method == "mean":
        resampled = data.resample(target_freq).mean()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return resampled.dropna()


def calculate_correlation_matrix(
    data: pd.DataFrame, columns: List[str] = None
) -> pd.DataFrame:
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


def find_optimal_weights(
    returns: pd.DataFrame, target_return: float = None, method: str = "equal"
) -> Dict[str, float]:
    """
    Find optimal weights for portfolio/ensemble

    Args:
        returns: Returns DataFrame for different components
        target_return: Target return (for optimization)
        method: Weight optimization method

    Returns:
        Optimal weights dictionary
    """
    if method == "equal":
        n_assets = len(returns.columns)
        weights = {col: 1.0 / n_assets for col in returns.columns}

    elif method == "risk_parity":
        # Simplified risk parity
        vols = returns.std()
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        weights = weights.to_dict()

    elif method == "mean_variance":
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
            weights = {col: 1.0 / n_assets for col in returns.columns}

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    return weights
