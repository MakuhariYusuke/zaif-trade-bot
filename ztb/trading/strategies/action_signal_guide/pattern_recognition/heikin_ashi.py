"""
Heikin-Ashi pattern recognition for Action Signal Guide.

Heikin-Ashi is a Japanese candlestick technique that modifies the traditional
candlestick chart to better reflect the trend and momentum.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ztb.features.trend.heikin_ashi import HeikinAshi
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)


class HeikinAshiRecognizer(PatternRecognizer):
    """
    Recognizes patterns using Heikin-Ashi candlesticks.

    Heikin-Ashi candlesticks smooth price action and make trends more visible.
    The signals are based on the relationship between consecutive Heikin-Ashi
    candlesticks and their color changes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.period = self.config.get('period', 1)  # Number of periods to look back
        self.trend_threshold = self.config.get('trend_threshold', 0.001)  # Minimum trend strength
        self.volume_weighted = self.config.get('volume_weighted', False)  # Use volume weighting

        # Use existing HeikinAshi feature class
        self.heikin_ashi_calculator = HeikinAshi()

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize Heikin-Ashi patterns in the data.

        Args:
            data: OHLCV DataFrame
            index: Index to analyze (-1 for latest)

        Returns:
            SignalResult if pattern detected, None otherwise
        """
        if len(data) < 2:
            return None

        # Calculate Heikin-Ashi values
        ha_data = self._calculate_heikin_ashi(data)

        if index == -1:
            index = len(ha_data) - 1

        if index < 1:
            return None

        current = ha_data.iloc[index]
        previous = ha_data.iloc[index - 1]

        # Analyze trend based on Heikin-Ashi candles
        signal = self._analyze_trend(current, previous)

        if signal:
            return SignalResult(
                signal_type="heikin_ashi",
                strength=abs(signal['strength']),
                direction=signal['direction'],
                description=signal['description'],
                confidence=signal['confidence']
            )

        return None

    def _calculate_heikin_ashi(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi values using the existing HeikinAshi feature class.

        Returns:
            DataFrame with HA_Open, HA_High, HA_Low, HA_Close, HA_Body columns
        """
        # Use the existing HeikinAshi feature class
        ha_data = self.heikin_ashi_calculator.compute(data)

        # Rename columns to match expected format
        ha_data = ha_data.rename(columns={
            'ha_open': 'HA_Open',
            'ha_high': 'HA_High',
            'ha_low': 'HA_Low',
            'ha_close': 'HA_Close'
        })

        # Calculate body size for analysis
        ha_data['HA_Body'] = abs(ha_data['HA_Close'] - ha_data['HA_Open'])

        return ha_data

    def _analyze_trend(self, current: pd.Series, previous: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Analyze trend based on Heikin-Ashi candle patterns.

        Returns signal dictionary or None if no clear signal.
        """
        # Determine candle colors
        current_green = current['HA_Close'] > current['HA_Open']
        previous_green = previous['HA_Close'] > previous['HA_Open']

        # Calculate trend strength
        body_ratio = current['HA_Body'] / (current['HA_High'] - current['HA_Low'])
        trend_strength = abs(current['HA_Close'] - current['HA_Open']) / current['HA_Open']

        # Strong trend signals
        if trend_strength > self.trend_threshold:
            # Bullish trend continuation (green candle after green)
            if current_green and previous_green:
                if current['HA_Close'] > previous['HA_Close']:
                    return {
                        'direction': ACTION_BUY,
                        'strength': trend_strength,
                        'description': f"Heikin-Ashi: Strong bullish trend continuation (strength: {trend_strength:.4f})",
                        'confidence': min(0.9, trend_strength * 100)
                    }

            # Bearish trend continuation (red candle after red)
            elif not current_green and not previous_green:
                if current['HA_Close'] < previous['HA_Close']:
                    return {
                        'direction': ACTION_SELL,
                        'strength': trend_strength,
                        'description': f"Heikin-Ashi: Strong bearish trend continuation (strength: {trend_strength:.4f})",
                        'confidence': min(0.9, trend_strength * 100)
                    }

        # Reversal signals
        # Bullish reversal (green after red)
        if current_green and not previous_green:
            if current['HA_Close'] > previous['HA_Open']:
                return {
                    'direction': ACTION_BUY,
                    'strength': trend_strength,
                    'description': f"Heikin-Ashi: Bullish reversal signal (strength: {trend_strength:.4f})",
                    'confidence': min(0.7, trend_strength * 50)
                }

        # Bearish reversal (red after green)
        elif not current_green and previous_green:
            if current['HA_Close'] < previous['HA_Open']:
                return {
                    'direction': ACTION_SELL,
                    'strength': trend_strength,
                    'description': f"Heikin-Ashi: Bearish reversal signal (strength: {trend_strength:.4f})",
                    'confidence': min(0.7, trend_strength * 50)
                }

        # Doji or weak signals - neutral
        if body_ratio < 0.1:  # Very small body indicates indecision
            return {
                'direction': ACTION_HOLD,
                'strength': 0.1,
                'description': "Heikin-Ashi: Indecision/Doji pattern detected",
                'confidence': 0.5
            }

        return None