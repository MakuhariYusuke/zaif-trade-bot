#!/usr/bin/env python3
"""
kalman_ext.py
Extended Kalman Filter analysis with residuals and autocorrelation
"""

from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.registry import FeatureRegistry


class SimpleKalmanFilter:
    """
    Simple Kalman Filter implementation for price estimation
    """

    def __init__(
        self, process_variance: float = 0.01, measurement_variance: float = 0.1
    ):
        super().__init__()
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        # Declare instance variables with type annotations
        self.posterior_estimate: Optional[float] = None
        self.posterior_error_estimate: Optional[float] = None
        self.estimates: List[float] = []
        self.error_estimates: List[float] = []
        self.reset()

    def reset(self) -> None:
        """Reset filter state"""
        self.posterior_estimate = None
        self.posterior_error_estimate = None
        self.estimates = []
        self.error_estimates = []

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update filter with new measurement

        Args:
            measurement: New observed value

        Returns:
            Tuple of (estimate, error_estimate)
        """
        if self.posterior_estimate is None or self.posterior_error_estimate is None:
            # First measurement: initialize without prediction step
            self.posterior_estimate = measurement
            self.posterior_error_estimate = 1.0
        else:
            # Prediction step
            prior_estimate = float(self.posterior_estimate)
            prior_error_estimate = float(
                self.posterior_error_estimate + self.process_variance
            )

            # Update step
            kalman_gain = prior_error_estimate / (
                prior_error_estimate + self.measurement_variance
            )
            self.posterior_estimate = prior_estimate + kalman_gain * (
                measurement - prior_estimate
            )
            self.posterior_error_estimate = (1 - kalman_gain) * prior_error_estimate

        self.estimates.append(float(self.posterior_estimate))
        self.error_estimates.append(float(self.posterior_error_estimate))

        return float(self.posterior_estimate), float(self.posterior_error_estimate)


@FeatureRegistry.register("Kalman_Estimate")
def compute_kalman_estimate(df: pd.DataFrame) -> pd.Series:
    """Kalman Filter Price Estimate"""
    extended_features = calculate_kalman_extended(df)
    return (
        extended_features["kalman_estimate"]
        if "kalman_estimate" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


@FeatureRegistry.register("Kalman_Residual")
def compute_kalman_residual(df: pd.DataFrame) -> pd.Series:
    """Kalman Filter Residual (Price - Estimate)"""
    extended_features = calculate_kalman_extended(df)
    return (
        extended_features["kalman_residual"]
        if "kalman_residual" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


@FeatureRegistry.register("Kalman_Residual_Norm")
def compute_kalman_residual_norm(df: pd.DataFrame) -> pd.Series:
    """Kalman Filter Normalized Residual"""
    extended_features = calculate_kalman_extended(df)
    return (
        extended_features["kalman_residual_normalized"]
        if "kalman_residual_normalized" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


def calculate_kalman_extended(
    data: pd.DataFrame,
    process_variance: float = 0.01,
    measurement_variance: float = 0.1,
    residual_window: int = 20,
) -> pd.DataFrame:
    """
    Calculate extended Kalman Filter features including residuals and autocorrelation

    Args:
        data: DataFrame with OHLCV data
        process_variance: Process noise variance
        measurement_variance: Measurement noise variance
        residual_window: Window for residual analysis

    Returns:
        DataFrame with extended Kalman features
    """
    if len(data) < 10:
        return pd.DataFrame(index=data.index)

    close_prices = cast(NDArray[np.float64], data["close"].values)
    n = len(close_prices)

    # Vectorized Kalman filter implementation
    estimates = np.zeros(n, dtype=np.float64)
    error_estimates = np.zeros(n, dtype=np.float64)

    # Initialize first values
    estimates[0] = close_prices[0]
    error_estimates[0] = 1.0

    # Vectorized Kalman filter update
    for i in range(1, n):
        # Prediction step
        prior_estimate = estimates[i - 1]
        prior_error_estimate = error_estimates[i - 1] + process_variance

        # Update step
        kalman_gain = prior_error_estimate / (
            prior_error_estimate + measurement_variance
        )
        estimates[i] = prior_estimate + kalman_gain * (close_prices[i] - prior_estimate)
        error_estimates[i] = (1 - kalman_gain) * prior_error_estimate

    # Calculate residuals
    residuals = close_prices - estimates

    features = {}

    # Basic Kalman features
    features["kalman_estimate"] = estimates
    features["kalman_error_estimate"] = error_estimates
    features["kalman_residual"] = residuals

    # Residual analysis
    features["kalman_residual_abs"] = np.abs(residuals)
    features["kalman_residual_squared"] = residuals**2

    # Rolling residual statistics - vectorized
    residual_series = pd.Series(residuals, index=data.index)
    features["kalman_residual_mean"] = np.asarray(
        residual_series.rolling(window=residual_window).mean()
    )
    features["kalman_residual_std"] = np.asarray(
        residual_series.rolling(window=residual_window).std()
    )
    features["kalman_residual_skew"] = np.asarray(
        residual_series.rolling(window=residual_window).skew()
    )
    features["kalman_residual_kurt"] = np.asarray(
        residual_series.rolling(window=residual_window).kurt()
    )

    # Residual percentiles - vectorized
    features["kalman_residual_percentile"] = np.asarray(
        residual_series.rolling(window=residual_window * 2)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # Normalized residuals - vectorized
    residual_std = np.asarray(residual_series.rolling(window=residual_window).std())
    epsilon = 1e-8  # Small value to prevent division by zero
    features["kalman_residual_normalized"] = residuals / (residual_std + epsilon)

    # Residual autocorrelation - optimized vectorized implementation
    autocorr_lags = [1, 2, 3, 5, 10]
    for lag in autocorr_lags:
        if len(residual_series) > lag + residual_window:
            # Vectorized autocorrelation calculation
            autocorr_values = np.zeros(len(residual_series), dtype=np.float64)

            # Calculate autocorrelation for each window position
            for i in range(lag + residual_window, len(residual_series)):
                window_start = i - residual_window
                window_end = i

                if window_end - window_start >= lag + 5:
                    window_data = residuals[window_start:window_end]
                    # Use numpy's correlate for autocorrelation
                    autocorr = np.corrcoef(window_data[:-lag], window_data[lag:])[0, 1]
                    if np.isfinite(autocorr):
                        autocorr_values[i] = autocorr

            features[f"kalman_autocorr_lag{lag}"] = autocorr_values

    # Filter confidence - vectorized
    confidence = 1.0 / (1.0 + error_estimates)
    features["kalman_confidence"] = confidence

    # Tracking error (deviation from estimate)
    close_values = np.asarray(data["close"])
    tracking_error = np.abs(close_values - estimates) / close_values
    features["kalman_tracking_error"] = tracking_error

    # Innovation (prediction error)
    innovation = np.diff(residuals, prepend=residuals[0])
    features["kalman_innovation"] = innovation
    features["kalman_innovation_abs"] = np.abs(innovation)

    # Filter adaptation signals
    adaptation_signal = error_estimates / np.mean(error_estimates)
    features["kalman_adaptation_signal"] = adaptation_signal

    # High residual periods (potential regime changes)
    residual_threshold = np.percentile(np.abs(residuals), 90)
    features["kalman_high_residual"] = (
        np.abs(residuals) > residual_threshold
    ).astype(int)

    # Trend consistency with filter
    price_values = np.asarray(data["close"])
    price_trend = np.sign(np.diff(price_values, prepend=price_values[0]))
    filter_trend = np.sign(np.diff(estimates, prepend=np.array([estimates[0]])))
    # Ensure both arrays are the same length as data
    min_len = min(len(price_trend), len(filter_trend), len(data))
    price_trend = price_trend[:min_len]
    filter_trend = filter_trend[:min_len]
    trend_agreement = (price_trend == filter_trend).astype(int)
    features["kalman_trend_agreement"] = trend_agreement

    # Filter divergence
    price_change = data["close"].pct_change()
    estimate_change = pd.Series(estimates, index=data.index).pct_change()
    divergence = price_change - estimate_change
    features["kalman_divergence"] = np.asarray(divergence)
    features["kalman_divergence_abs"] = np.abs(np.asarray(divergence))

    # Multi-step ahead prediction error
    if len(data) > 5:
        # Simple 5-step ahead prediction using current estimate + trend
        trend_5step = np.asarray(pd.Series(estimates, index=data.index).diff(5).values)
        prediction_5step = pd.Series(estimates, index=data.index) + trend_5step
        actual_5step = data["close"].shift(-5)
        prediction_error_5step = actual_5step - prediction_5step
        features["kalman_prediction_error_5step"] = np.asarray(prediction_error_5step)

    # Volatility-adjusted residuals
    volatility = data["close"].rolling(window=residual_window).std()
    features["kalman_residual_vol_adj"] = residual_series / np.asarray(volatility)

    # Residual momentum
    residual_momentum = residual_series.diff(5)
    features["kalman_residual_momentum"] = np.asarray(residual_momentum)

    # Create result DataFrame
    result_df = pd.DataFrame(features, index=data.index)

    # Fill NaN values (forward fill only to avoid masking data issues)
    result_df = result_df.ffill()

    return result_df


def calculate_kalman_signals(
    extended_features: pd.DataFrame,
    confidence_threshold: float = 0.7,
    residual_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Generate trading signals from extended Kalman features

    Args:
        extended_features: Output from calculate_kalman_extended
        confidence_threshold: Minimum confidence for signal generation
        residual_threshold: Threshold for high residual detection

    Returns:
        DataFrame with trading signals
    """
    signals = {}

    # High confidence trend following
    high_confidence = extended_features["kalman_confidence"] > confidence_threshold
    trend_up = extended_features["kalman_divergence"] > 0.005
    trend_down = extended_features["kalman_divergence"] < -0.005

    signals["kalman_confident_bullish"] = (high_confidence & trend_up).astype(int)
    signals["kalman_confident_bearish"] = (high_confidence & trend_down).astype(int)

    # Mean reversion based on residuals
    high_positive_residual = (
        extended_features["kalman_residual_normalized"] > residual_threshold
    )
    high_negative_residual = (
        extended_features["kalman_residual_normalized"] < -residual_threshold
    )

    signals["kalman_mean_revert_sell"] = high_positive_residual.astype(int)
    signals["kalman_mean_revert_buy"] = high_negative_residual.astype(int)

    # Regime change detection
    regime_change = (extended_features["kalman_high_residual"] == 1) & (
        extended_features["kalman_trend_agreement"] == 0
    )
    signals["kalman_regime_change"] = regime_change.astype(int)

    # Filter breakout (when price consistently deviates from filter)
    consistent_deviation = (extended_features["kalman_tracking_error"] > 0.02) & (
        extended_features["kalman_trend_agreement"] == 1
    )
    signals["kalman_breakout"] = consistent_deviation.astype(int)

    # Autocorrelation-based signals
    if "kalman_autocorr_lag1" in extended_features.columns:
        low_autocorr = (
            extended_features["kalman_autocorr_lag1"].abs() < 0.1
        )  # White noise residuals
        signals["kalman_low_autocorr"] = low_autocorr.astype(int)

    return pd.DataFrame(signals, index=extended_features.index)


def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic OHLCV data with regime changes."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    base_price = 100
    trend1 = np.cumsum(np.random.normal(0.001, 0.01, 100))  # Low vol
    trend2 = np.cumsum(np.random.normal(0.002, 0.03, 100))  # High vol
    trend3 = np.cumsum(np.random.normal(-0.001, 0.015, 100))  # Medium vol, down trend
    combined_trend = np.concatenate([trend1, trend2, trend3])
    close_prices = base_price * np.exp(combined_trend)
    high_mult = 1 + np.abs(np.random.normal(0, 0.005, 300))
    low_mult = 1 - np.abs(np.random.normal(0, 0.005, 300))
    data = pd.DataFrame(
        {
            "open": close_prices * np.random.uniform(0.995, 1.005, 300),
            "high": close_prices * high_mult,
            "low": close_prices * low_mult,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, 300),
        },
        index=dates,
    )
    return data


def print_feature_info(data: pd.DataFrame, extended_features: pd.DataFrame) -> None:
    print("Testing Kalman Extended Features...")
    print(f"Data shape: {data.shape}")
    print(f"Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    print(f"\nExtended features shape: {extended_features.shape}")
    print(f"Feature columns ({len(extended_features.columns)}):")
    for i, col in enumerate(extended_features.columns):
        print(f"  {i + 1:2d}. {col}")


def print_signal_info(signals: pd.DataFrame) -> None:
    print(f"\nSignals shape: {signals.shape}")
    print("Signal counts:")
    for col in signals.columns:
        signal_count = signals[col].sum()
        print(f"  {col}: {signal_count} signals")


def print_residual_analysis(extended_features: pd.DataFrame) -> None:
    residuals = extended_features["kalman_residual"].dropna()
    print(f"\nResidual analysis:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std: {residuals.std():.6f}")
    print(f"  Skewness: {residuals.skew():.4f}")  # type: ignore[str-bytes-safe]
    print(f"  Kurtosis: {residuals.kurt():.4f}")  # type: ignore[str-bytes-safe]


def print_autocorr_analysis(extended_features: pd.DataFrame) -> None:
    if "kalman_autocorr_lag1" in extended_features.columns:
        autocorr_1 = extended_features["kalman_autocorr_lag1"].dropna()
        print(f"  Lag-1 autocorr mean: {autocorr_1.mean():.4f}")
        print(f"  Lag-1 autocorr std: {autocorr_1.std():.4f}")


def main() -> None:
    data = generate_synthetic_data()
    extended_features = calculate_kalman_extended(
        data, process_variance=0.001, measurement_variance=0.1, residual_window=20
    )
    print_feature_info(data, extended_features)
    signals = calculate_kalman_signals(extended_features)
    print_signal_info(signals)
    print_residual_analysis(extended_features)
    print_autocorr_analysis(extended_features)
    print("Kalman extended features test completed successfully!")


if __name__ == "__main__":
    main()
