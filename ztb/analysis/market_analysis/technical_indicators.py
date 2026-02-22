#!/usr/bin/env python3
"""
Enhanced Technical Indicators for Market Analysis.

This module provides enhanced technical indicator calculations with improved
accuracy and additional features for market regime analysis.
"""

from typing import Tuple

import numpy as np


class EnhancedTechnicalIndicators:
    """
    Enhanced technical indicators with improved calculations.

    Provides static methods for calculating various technical indicators
    used in market analysis and regime detection.
    """

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Array of price data
            period: Period for RSI calculation (default: 14)

        Returns:
            RSI value between 0 and 100
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI for insufficient data

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    @staticmethod
    def calculate_adx(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        """
        Calculate Average Directional Index (ADX).

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: Period for ADX calculation

        Returns:
            ADX value
        """
        if (
            len(highs) < period + 1
            or len(lows) < period + 1
            or len(closes) < period + 1
        ):
            return 25.0  # Neutral ADX

        # Calculate True Range
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(hl, np.maximum(hc, lc))

        # Calculate Directional Movement
        dm_plus = np.where(
            (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
            np.maximum(highs[1:] - highs[:-1], 0),
            0,
        )
        dm_minus = np.where(
            (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
            np.maximum(lows[:-1] - lows[1:], 0),
            0,
        )

        # Calculate Directional Indicators
        di_plus = (
            100
            * (
                np.convolve(dm_plus, np.ones(period), "valid")
                / np.convolve(tr, np.ones(period), "valid")
            )[-1]
        )
        di_minus = (
            100
            * (
                np.convolve(dm_minus, np.ones(period), "valid")
                / np.convolve(tr, np.ones(period), "valid")
            )[-1]
        )

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = float(dx)  # dx is already a scalar

        return adx

    @staticmethod
    def calculate_macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Tuple of (MACD line, Signal line, Histogram) - last values as floats
        """
        if len(prices) < slow_period:
            # Return zeros for insufficient data
            return 0.0, 0.0, 0.0

        # Calculate EMAs
        fast_ema = EnhancedTechnicalIndicators._calculate_ema(prices, fast_period)
        slow_ema = EnhancedTechnicalIndicators._calculate_ema(prices, slow_period)

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = EnhancedTechnicalIndicators._calculate_ema(
            macd_line, signal_period
        )

        # Calculate histogram
        histogram = macd_line - signal_line

        # Return last values as floats
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    @staticmethod
    def calculate_bollinger_bands(
        prices: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Price data
            period: Moving average period
            num_std: Number of standard deviations

        Returns:
            Tuple of (SMA, Upper band, Lower band) - last values as floats
        """
        if len(prices) < period:
            mean_price = np.mean(prices)
            return float(mean_price), float(mean_price), float(mean_price)

        sma = np.convolve(prices, np.ones(period), "valid") / period
        std = np.array(
            [np.std(prices[i : i + period]) for i in range(len(prices) - period + 1)]
        )

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Return last values as floats
        return float(sma[-1]), float(upper_band[-1]), float(lower_band[-1])

    @staticmethod
    def calculate_atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        """
        Calculate Average True Range (ATR).

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: Period for ATR calculation

        Returns:
            ATR value
        """
        if (
            len(highs) < period + 1
            or len(lows) < period + 1
            or len(closes) < period + 1
        ):
            return 0.0

        # Calculate True Range
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(hl, np.maximum(hc, lc))

        # Calculate ATR
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

        return float(atr)

    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: Price data
            period: EMA period

        Returns:
            EMA values
        """
        if len(prices) < period:
            return np.full(len(prices), np.mean(prices))

        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier))

        return ema
