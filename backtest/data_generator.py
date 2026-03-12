#!/usr/bin/env python3
"""
Synthetic Data Generator for Backtesting

This module provides functions to generate synthetic OHLCV data
for backtesting trading strategies.
"""

import numpy as np
import pandas as pd


def generate_synthetic_data(n_periods=5000, start_price=50000.0, volatility=500):
    """
    Generate synthetic OHLCV data for backtesting.

    Args:
        n_periods: Number of periods to generate
        start_price: Starting price
        volatility: Price volatility

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="h")
    np.random.seed(42)

    # Generate realistic price data with trends and volatility
    prices = []
    base_price = start_price
    trend = 0.0

    for i in range(n_periods):
        # Add trend component
        trend += np.random.normal(0, 0.001)

        # Add cyclical component
        cycle = 1000 * np.sin(i * 0.01)

        # Add noise
        noise = np.random.normal(0, volatility)

        price = base_price + trend * 1000 + cycle + noise
        prices.append(max(price, 1000))  # Ensure positive prices

    # Convert to OHLCV format
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.02)))
        low = price * (1 - abs(np.random.normal(0, 0.02)))
        open_price = prices[i - 1] if i > 0 else price
        close = price
        volume = np.random.randint(1000, 10000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def generate_trending_data(n_periods=1000, trend_strength=0.0001):
    """
    Generate data with a specific trend.

    Args:
        n_periods: Number of periods
        trend_strength: Strength of the trend

    Returns:
        DataFrame with trending OHLCV data
    """
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="h")
    np.random.seed(42)

    prices = []
    base_price = 50000.0

    for i in range(n_periods):
        trend = i * trend_strength * base_price
        noise = np.random.normal(0, 200)
        price = base_price + trend + noise
        prices.append(max(price, 1000))

    # Convert to OHLCV
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i - 1] if i > 0 else price
        close = price
        volume = np.random.randint(500, 5000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df
