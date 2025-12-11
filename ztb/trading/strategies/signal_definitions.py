"""
Signal Definitions - Classical Technical Signals for Action Guidance

This module defines classical technical analysis signals that serve as
training wheels for reinforcement learning agents.
"""

from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np

from ztb.utils.talib_wrapper import TaLibWrapper


class SignalType(Enum):
    """Types of trading signals."""

    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


class SignalStrength(Enum):
    """Strength levels of signals."""

    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


class SignalDefinitions:
    """
    Classical technical analysis signals for BUY/SELL guidance.

    These signals serve as training wheels to help RL agents learn
    basic trading patterns before discovering novel strategies.
    """

    # Expose Enums for backward compatibility
    SignalType = SignalType
    SignalStrength = SignalStrength

    def __init__(self) -> None:
        """Initialize signal definitions."""
        self.signals = self._define_signals()
        self.talib = TaLibWrapper()

    def _define_signals(self) -> Dict[str, Dict[str, Any]]:
        """Define all classical technical signals."""
        return {
            # Moving Average Signals
            "golden_cross": {
                "type": SignalType.BUY,
                "strength": SignalStrength.STRONG,
                "description": "Short-term MA crosses above long-term MA",
                "function": self._golden_cross_signal,
            },
            "death_cross": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "Short-term MA crosses below long-term MA",
                "function": self._death_cross_signal,
            },
            # RSI Signals
            "rsi_oversold": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "RSI below 30 (oversold)",
                "function": self._rsi_oversold_signal,
            },
            "rsi_overbought": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "RSI above 70 (overbought)",
                "function": self._rsi_overbought_signal,
            },
            # MACD Signals
            "macd_bullish": {
                "type": SignalType.BUY,
                "strength": SignalStrength.STRONG,
                "description": "MACD line crosses above signal line",
                "function": self._macd_bullish_signal,
            },
            "macd_bearish": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "MACD line crosses below signal line",
                "function": self._macd_bearish_signal,
            },
            # Bollinger Band Signals
            "bollinger_lower_touch": {
                "type": SignalType.BUY,
                "strength": SignalStrength.WEAK,
                "description": "Price touches lower Bollinger Band",
                "function": self._bollinger_lower_touch_signal,
            },
            "bollinger_upper_touch": {
                "type": SignalType.SELL,
                "strength": SignalStrength.WEAK,
                "description": "Price touches upper Bollinger Band",
                "function": self._bollinger_upper_touch_signal,
            },
            # Stochastic Signals
            "stoch_oversold": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Stochastic %K below 20",
                "function": self._stoch_oversold_signal,
            },
            "stoch_overbought": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "Stochastic %K above 80",
                "function": self._stoch_overbought_signal,
            },
            # Neutral Signals
            "range_bound": {
                "type": SignalType.NEUTRAL,
                "strength": SignalStrength.WEAK,
                "description": "Price within Bollinger Bands (range bound)",
                "function": self._range_bound_signal,
            },
            "low_volatility": {
                "type": SignalType.NEUTRAL,
                "strength": SignalStrength.WEAK,
                "description": "Low ATR indicating low volatility",
                "function": self._low_volatility_signal,
            },
            # Advanced Trend Signals
            "adx_strong_trend": {
                "type": SignalType.NEUTRAL,
                "strength": SignalStrength.MODERATE,
                "description": "ADX above 25 indicating strong trend",
                "function": self._adx_strong_trend_signal,
            },
            "plus_di_bullish": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "+DI crosses above -DI",
                "function": self._plus_di_bullish_signal,
            },
            "minus_di_bearish": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "-DI crosses above +DI",
                "function": self._minus_di_bearish_signal,
            },
            # Momentum Signals
            "williams_r_oversold": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Williams %R below -80 (oversold)",
                "function": self._williams_r_oversold_signal,
            },
            "williams_r_overbought": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "Williams %R above -20 (overbought)",
                "function": self._williams_r_overbought_signal,
            },
            "cci_oversold": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "CCI below -100 (oversold)",
                "function": self._cci_oversold_signal,
            },
            "cci_overbought": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "CCI above 100 (overbought)",
                "function": self._cci_overbought_signal,
            },
            # Combined Signals (Multiple Indicators)
            "trend_momentum_bullish": {
                "type": SignalType.BUY,
                "strength": SignalStrength.STRONG,
                "description": "ADX trend + bullish momentum (RSI/MACD)",
                "function": self._trend_momentum_bullish_signal,
            },
            "trend_momentum_bearish": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "ADX trend + bearish momentum (RSI/MACD)",
                "function": self._trend_momentum_bearish_signal,
            },
            "oscillator_divergence_bullish": {
                "type": SignalType.BUY,
                "strength": SignalStrength.STRONG,
                "description": "Bullish divergence in multiple oscillators",
                "function": self._oscillator_divergence_bullish_signal,
            },
            "oscillator_divergence_bearish": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "Bearish divergence in multiple oscillators",
                "function": self._oscillator_divergence_bearish_signal,
            },
            "volume_price_confirmation": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Price and volume trend confirmation",
                "function": self._volume_price_confirmation_signal,
            },
            # Advanced Pattern Signals
            "high_volatility_breakout": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Price breakout during high volatility",
                "function": self._high_volatility_breakout_signal,
            },
            "low_volatility_breakout": {
                "type": SignalType.SELL,
                "strength": SignalStrength.WEAK,
                "description": "False breakout in low volatility",
                "function": self._low_volatility_breakout_signal,
            },
            "price_channel_breakout": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Price breaking out of established channel",
                "function": self._price_channel_breakout_signal,
            },
            "reversal_candlestick": {
                "type": SignalType.BUY,
                "strength": SignalStrength.STRONG,
                "description": "Bullish reversal candlestick pattern",
                "function": self._reversal_candlestick_signal,
            },
            "momentum_divergence": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "Negative momentum divergence",
                "function": self._momentum_divergence_signal,
            },
            # Sakata Five Methods (酒田五法) - Japanese Candlestick Patterns
            "sankuu_tataki_komi": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "Three crows pattern - three consecutive bearish candles",
                "function": self._sankuu_tataki_komi_signal,
            },
            "sante_daiinsen": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "Three methods formation - large bearish candle",
                "function": self._sante_daiinsen_signal,
            },
            "age_sanpo": {
                "type": SignalType.BUY,
                "strength": SignalStrength.STRONG,
                "description": "Rising three methods - consolidation in uptrend",
                "function": self._age_sanpo_signal,
            },
            "in_no_you_harami": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Bullish harami - small bullish candle inside bearish",
                "function": self._in_no_you_harami_signal,
            },
            # Wave Theory Patterns (波動論)
            "elliott_wave_1": {
                "type": SignalType.BUY,
                "strength": SignalStrength.MODERATE,
                "description": "Elliott Wave 1 - start of impulsive wave",
                "function": self._elliott_wave_1_signal,
            },
            "elliott_wave_5": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "Elliott Wave 5 - end of impulsive wave",
                "function": self._elliott_wave_5_signal,
            },
            "motive_wave_completion": {
                "type": SignalType.SELL,
                "strength": SignalStrength.STRONG,
                "description": "Motive wave completion pattern",
                "function": self._motive_wave_completion_signal,
            },
            "corrective_wave_a": {
                "type": SignalType.SELL,
                "strength": SignalStrength.MODERATE,
                "description": "Corrective wave A - start of correction",
                "function": self._corrective_wave_a_signal,
            },
            "time_wave_confluence": {
                "type": SignalType.NEUTRAL,
                "strength": SignalStrength.WEAK,
                "description": "Time wave confluence point",
                "function": self._time_wave_confluence_signal,
            },
        }

    def get_signal_names(self) -> List[str]:
        """Get list of all defined signal names."""
        return list(self.signals.keys())

    def get_signals_by_type(self, signal_type: SignalType) -> List[str]:
        """Get signals filtered by type."""
        return [
            name
            for name, config in self.signals.items()
            if config["type"] == signal_type
        ]

    def evaluate_signal(
        self, signal_name: str, observation: np.ndarray, feature_names: List[str]
    ) -> Tuple[SignalType, float]:
        """
        Evaluate a specific signal for given observation.

        Args:
            signal_name: Name of the signal to evaluate
            observation: Current market observation
            feature_names: Names of features in observation

        Returns:
            Tuple of (signal_type, strength) if signal is active, else (NEUTRAL, 0.0)
        """
        if signal_name not in self.signals:
            return SignalType.NEUTRAL, 0.0

        signal_config = self.signals[signal_name]
        signal_function = signal_config["function"]

        try:
            is_active = signal_function(observation, feature_names)
            if is_active:
                return signal_config["type"], signal_config["strength"].value
            else:
                return SignalType.NEUTRAL, 0.0
        except Exception:
            # If signal evaluation fails, return neutral
            return SignalType.NEUTRAL, 0.0

    # Signal evaluation functions

    def _golden_cross_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect golden cross pattern."""
        try:
            # Find short-term and long-term MA features
            short_ma_features = [
                f
                for f in feature_names
                if any(x in f.lower() for x in ["sma5", "ema5", "ma5"])
            ]
            long_ma_features = [
                f
                for f in feature_names
                if any(
                    x in f.lower()
                    for x in ["sma20", "ema20", "ma20", "sma21", "ema21", "ma21"]
                )
            ]

            if short_ma_features and long_ma_features:
                # Get the most recent values
                short_ma_idx = next(
                    (
                        i
                        for i, name in enumerate(feature_names)
                        if name in short_ma_features
                    ),
                    None,
                )
                long_ma_idx = next(
                    (
                        i
                        for i, name in enumerate(feature_names)
                        if name in long_ma_features
                    ),
                    None,
                )

                if short_ma_idx is not None and long_ma_idx is not None:
                    short_ma = float(observation[short_ma_idx])
                    long_ma = float(observation[long_ma_idx])
                    # Golden cross: short-term MA above long-term MA
                    return short_ma > long_ma
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _death_cross_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect death cross pattern."""
        try:
            # Find short-term and long-term MA features
            short_ma_features = [
                f
                for f in feature_names
                if any(x in f.lower() for x in ["sma5", "ema5", "ma5"])
            ]
            long_ma_features = [
                f
                for f in feature_names
                if any(
                    x in f.lower()
                    for x in ["sma20", "ema20", "ma20", "sma21", "ema21", "ma21"]
                )
            ]

            if short_ma_features and long_ma_features:
                # Get the most recent values
                short_ma_idx = next(
                    (
                        i
                        for i, name in enumerate(feature_names)
                        if name in short_ma_features
                    ),
                    None,
                )
                long_ma_idx = next(
                    (
                        i
                        for i, name in enumerate(feature_names)
                        if name in long_ma_features
                    ),
                    None,
                )

                if short_ma_idx is not None and long_ma_idx is not None:
                    short_ma = float(observation[short_ma_idx])
                    long_ma = float(observation[long_ma_idx])
                    # Death cross: short-term MA below long-term MA
                    return short_ma < long_ma
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _rsi_oversold_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect RSI oversold condition."""
        try:
            rsi_idx = next(
                (i for i, name in enumerate(feature_names) if "rsi" in name.lower()),
                None,
            )
            if rsi_idx is not None:
                rsi_value = float(observation[rsi_idx])
                return rsi_value < 30.0
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _rsi_overbought_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect RSI overbought condition."""
        try:
            rsi_idx = next(
                (i for i, name in enumerate(feature_names) if "rsi" in name.lower()),
                None,
            )
            if rsi_idx is not None:
                rsi_value = float(observation[rsi_idx])
                return rsi_value > 70.0
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _macd_bullish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bullish MACD signal."""
        try:
            macd_idx = next(
                (i for i, name in enumerate(feature_names) if "macd" in name.lower()),
                None,
            )
            if macd_idx is not None:
                macd_value = float(observation[macd_idx])
                return macd_value > 0.0  # Simplified
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _macd_bearish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bearish MACD signal."""
        try:
            macd_idx = next(
                (i for i, name in enumerate(feature_names) if "macd" in name.lower()),
                None,
            )
            if macd_idx is not None:
                macd_value = float(observation[macd_idx])
                return macd_value < 0.0  # Simplified
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _bollinger_lower_touch_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect price touching lower Bollinger Band."""
        try:
            # Find price and Bollinger Band features
            price_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "close" in name.lower() or "price" in name.lower()
                ),
                None,
            )
            bb_lower_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "bb_lower" in name.lower() or "bollinger_lower" in name.lower()
                ),
                None,
            )

            if price_idx is not None and bb_lower_idx is not None:
                price = float(observation[price_idx])
                bb_lower = float(observation[bb_lower_idx])
                # Price touching or below lower Bollinger Band
                return price <= bb_lower * 1.001  # Small tolerance for touching
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _bollinger_upper_touch_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect price touching upper Bollinger Band."""
        try:
            # Find price and Bollinger Band features
            price_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "close" in name.lower() or "price" in name.lower()
                ),
                None,
            )
            bb_upper_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "bb_upper" in name.lower() or "bollinger_upper" in name.lower()
                ),
                None,
            )

            if price_idx is not None and bb_upper_idx is not None:
                price = float(observation[price_idx])
                bb_upper = float(observation[bb_upper_idx])
                # Price touching or above upper Bollinger Band
                return price >= bb_upper * 0.999  # Small tolerance for touching
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _stoch_oversold_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect stochastic oversold condition."""
        try:
            stoch_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "stoch" in name.lower() or "stochastic" in name.lower()
                ),
                None,
            )
            if stoch_idx is not None:
                stoch_value = float(observation[stoch_idx])
                return stoch_value < 20.0
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _stoch_overbought_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect stochastic overbought condition."""
        try:
            stoch_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "stoch" in name.lower() or "stochastic" in name.lower()
                ),
                None,
            )
            if stoch_idx is not None:
                stoch_value = float(observation[stoch_idx])
                return stoch_value > 80.0
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _range_bound_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect range-bound market condition."""
        try:
            # Check if price is within reasonable range of moving averages
            return False  # Placeholder
        except (IndexError, ValueError, TypeError):
            return False

    def _low_volatility_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect low volatility condition."""
        try:
            atr_idx = next(
                (i for i, name in enumerate(feature_names) if "atr" in name.lower()),
                None,
            )
            if atr_idx is not None:
                atr_value = float(observation[atr_idx])
                return atr_value < 0.01  # Threshold would be tuned
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _adx_strong_trend_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect strong trend using ADX."""
        try:
            adx_idx = next(
                (i for i, name in enumerate(feature_names) if "adx" in name.lower()),
                None,
            )
            if adx_idx is not None:
                adx_value = float(observation[adx_idx])
                return adx_value > 25.0  # Strong trend threshold
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _plus_di_bullish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bullish directional movement (+DI > -DI)."""
        try:
            plus_di_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "plus_di" in name.lower() or "+di" in name.lower()
                ),
                None,
            )
            minus_di_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "minus_di" in name.lower() or "-di" in name.lower()
                ),
                None,
            )

            if plus_di_idx is not None and minus_di_idx is not None:
                plus_di = float(observation[plus_di_idx])
                minus_di = float(observation[minus_di_idx])
                return plus_di > minus_di
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _minus_di_bearish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bearish directional movement (-DI > +DI)."""
        try:
            plus_di_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "plus_di" in name.lower() or "+di" in name.lower()
                ),
                None,
            )
            minus_di_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "minus_di" in name.lower() or "-di" in name.lower()
                ),
                None,
            )

            if plus_di_idx is not None and minus_di_idx is not None:
                plus_di = float(observation[plus_di_idx])
                minus_di = float(observation[minus_di_idx])
                return minus_di > plus_di
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _williams_r_oversold_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect Williams %R oversold condition."""
        try:
            williams_r_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "williams" in name.lower() or "willr" in name.lower()
                ),
                None,
            )
            if williams_r_idx is not None:
                williams_r_value = float(observation[williams_r_idx])
                return williams_r_value < -80.0  # Oversold threshold
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _williams_r_overbought_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect Williams %R overbought condition."""
        try:
            williams_r_idx = next(
                (
                    i
                    for i, name in enumerate(feature_names)
                    if "williams" in name.lower() or "willr" in name.lower()
                ),
                None,
            )
            if williams_r_idx is not None:
                williams_r_value = float(observation[williams_r_idx])
                return williams_r_value > -20.0  # Overbought threshold
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _cci_oversold_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect CCI oversold condition."""
        try:
            cci_idx = next(
                (i for i, name in enumerate(feature_names) if "cci" in name.lower()),
                None,
            )
            if cci_idx is not None:
                cci_value = float(observation[cci_idx])
                return cci_value < -100.0  # Oversold threshold
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _cci_overbought_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect CCI overbought condition."""
        try:
            cci_idx = next(
                (i for i, name in enumerate(feature_names) if "cci" in name.lower()),
                None,
            )
            if cci_idx is not None:
                cci_value = float(observation[cci_idx])
                return cci_value > 100.0  # Overbought threshold
            return False
        except (IndexError, ValueError, TypeError):
            return False

    def _trend_momentum_bullish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bullish trend + momentum combination."""
        try:
            # Check for strong trend (ADX > 25)
            adx_idx = next(
                (i for i, name in enumerate(feature_names) if "adx" in name.lower()),
                None,
            )
            if adx_idx is None or float(observation[adx_idx]) < 25.0:
                return False

            # Check for bullish momentum (RSI oversold + MACD bullish)
            rsi_oversold = self._rsi_oversold_signal(observation, feature_names)
            macd_bullish = self._macd_bullish_signal(observation, feature_names)

            # Additional check: +DI > -DI
            plus_di_bullish = self._plus_di_bullish_signal(observation, feature_names)

            return (rsi_oversold or macd_bullish) and plus_di_bullish
        except (IndexError, ValueError, TypeError):
            return False

    def _trend_momentum_bearish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bearish trend + momentum combination."""
        try:
            # Check for strong trend (ADX > 25)
            adx_idx = next(
                (i for i, name in enumerate(feature_names) if "adx" in name.lower()),
                None,
            )
            if adx_idx is None or float(observation[adx_idx]) < 25.0:
                return False

            # Check for bearish momentum (RSI overbought + MACD bearish)
            rsi_overbought = self._rsi_overbought_signal(observation, feature_names)
            macd_bearish = self._macd_bearish_signal(observation, feature_names)

            # Additional check: -DI > +DI
            minus_di_bearish = self._minus_di_bearish_signal(observation, feature_names)

            return (rsi_overbought or macd_bearish) and minus_di_bearish
        except (IndexError, ValueError, TypeError):
            return False

    def _oscillator_divergence_bullish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bullish divergence in multiple oscillators."""
        try:
            # Check for oversold conditions in multiple oscillators
            oscillators_oversold = [
                self._rsi_oversold_signal(observation, feature_names),
                self._stoch_oversold_signal(observation, feature_names),
                self._williams_r_oversold_signal(observation, feature_names),
                self._cci_oversold_signal(observation, feature_names),
            ]

            # Require at least 2 oscillators to be oversold
            return sum(oscillators_oversold) >= 2
        except (IndexError, ValueError, TypeError):
            return False

    def _oscillator_divergence_bearish_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bearish divergence in multiple oscillators."""
        try:
            # Check for overbought conditions in multiple oscillators
            oscillators_overbought = [
                self._rsi_overbought_signal(observation, feature_names),
                self._stoch_overbought_signal(observation, feature_names),
                self._williams_r_overbought_signal(observation, feature_names),
                self._cci_overbought_signal(observation, feature_names),
            ]

            # Require at least 2 oscillators to be overbought
            return sum(oscillators_overbought) >= 2
        except (IndexError, ValueError, TypeError):
            return False

    def _volume_price_confirmation_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect volume-price trend confirmation."""
        try:
            # This is a simplified implementation - in practice would need volume features
            # For now, check if we have strong bullish signals with trend confirmation
            adx_strong = self._adx_strong_trend_signal(observation, feature_names)
            plus_di_bullish = self._plus_di_bullish_signal(observation, feature_names)

            # Look for volume confirmation (placeholder - would need actual volume data)
            volume_confirm = True  # Placeholder

            return adx_strong and plus_di_bullish and volume_confirm
        except (IndexError, ValueError, TypeError):
            return False

    def _high_volatility_breakout_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect breakout during high volatility periods."""
        try:
            # Check for high volatility (ATR above threshold)
            atr_idx = next(
                (i for i, name in enumerate(feature_names) if "atr" in name.lower()),
                None,
            )
            if atr_idx is not None:
                atr_value = float(observation[atr_idx])
                high_volatility = atr_value > 0.02  # Threshold would be tuned
            else:
                high_volatility = False

            # Check for price breakout (above recent high)
            # This is simplified - would need proper breakout detection
            breakout = self._bollinger_upper_touch_signal(observation, feature_names)

            return high_volatility and breakout
        except (IndexError, ValueError, TypeError):
            return False

    def _low_volatility_breakout_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect false breakouts in low volatility (potential reversal)."""
        try:
            # Check for low volatility
            low_volatility = self._low_volatility_signal(observation, feature_names)

            # Check for recent breakout attempt (touched upper BB but RSI overbought)
            bb_touch = self._bollinger_upper_touch_signal(observation, feature_names)
            rsi_overbought = self._rsi_overbought_signal(observation, feature_names)

            return low_volatility and bb_touch and rsi_overbought
        except (IndexError, ValueError, TypeError):
            return False

    # Sakata Five Methods (酒田五法) Signal Functions
    def _sankuu_tataki_komi_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect Three Crows pattern - three consecutive bearish candles."""
        try:
            # This is a simplified implementation - would need candlestick data
            # Check for consecutive bearish momentum signals
            rsi_overbought = self._rsi_overbought_signal(observation, feature_names)
            macd_bearish = self._macd_bearish_signal(observation, feature_names)
            stoch_overbought = self._stoch_overbought_signal(observation, feature_names)

            # Three consecutive bearish signals
            return rsi_overbought and macd_bearish and stoch_overbought
        except (IndexError, ValueError, TypeError):
            return False

    def _sante_daiinsen_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect Three Methods formation - large bearish candle."""
        try:
            # Check for strong bearish momentum with high volatility
            high_volatility = self._high_volatility_breakout_signal(
                observation, feature_names
            )
            macd_bearish = self._macd_bearish_signal(observation, feature_names)
            adx_strong = self._adx_strong_trend_signal(observation, feature_names)

            return high_volatility and macd_bearish and adx_strong
        except (IndexError, ValueError, TypeError):
            return False

    def _age_sanpo_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect Rising Three Methods - consolidation in uptrend."""
        try:
            # Check for bullish trend with consolidation
            golden_cross = self._golden_cross_signal(observation, feature_names)
            range_bound = self._range_bound_signal(observation, feature_names)
            plus_di_bullish = self._plus_di_bullish_signal(observation, feature_names)

            return golden_cross and range_bound and plus_di_bullish
        except (IndexError, ValueError, TypeError):
            return False

    def _in_no_you_harami_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect Bullish Harami - small bullish candle inside bearish."""
        try:
            # Check for oversold conditions with bullish reversal signals
            rsi_oversold = self._rsi_oversold_signal(observation, feature_names)
            oscillator_divergence = self._oscillator_divergence_bullish_signal(
                observation, feature_names
            )

            return rsi_oversold and oscillator_divergence
        except (IndexError, ValueError, TypeError):
            return False

    # Wave Theory (波動論) Signal Functions
    def _elliott_wave_1_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect start of Elliott Wave 1 - impulsive wave beginning."""
        try:
            # Check for trend change with momentum
            golden_cross = self._golden_cross_signal(observation, feature_names)
            macd_bullish = self._macd_bullish_signal(observation, feature_names)
            low_volatility = self._low_volatility_signal(observation, feature_names)

            return golden_cross and macd_bullish and low_volatility
        except (IndexError, ValueError, TypeError):
            return False

    def _elliott_wave_5_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect end of Elliott Wave 5 - impulsive wave completion."""
        try:
            # Check for overbought conditions in strong trend
            rsi_overbought = self._rsi_overbought_signal(observation, feature_names)
            adx_strong = self._adx_strong_trend_signal(observation, feature_names)
            momentum_divergence = self._momentum_divergence_signal(
                observation, feature_names
            )

            return rsi_overbought and adx_strong and momentum_divergence
        except (IndexError, ValueError, TypeError):
            return False

    def _motive_wave_completion_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect motive wave completion pattern."""
        try:
            # Check for exhaustion signals in strong trend
            oscillator_overbought = self._oscillator_divergence_bearish_signal(
                observation, feature_names
            )
            adx_strong = self._adx_strong_trend_signal(observation, feature_names)
            bb_upper_touch = self._bollinger_upper_touch_signal(
                observation, feature_names
            )

            return oscillator_overbought and adx_strong and bb_upper_touch
        except (IndexError, ValueError, TypeError):
            return False

    def _corrective_wave_a_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect start of corrective wave A."""
        try:
            # Check for trend reversal after exhaustion
            death_cross = self._death_cross_signal(observation, feature_names)
            oscillator_divergence = self._oscillator_divergence_bearish_signal(
                observation, feature_names
            )

            return death_cross and oscillator_divergence
        except (IndexError, ValueError, TypeError):
            return False

    def _time_wave_confluence_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect time wave confluence point."""
        try:
            # This is a simplified implementation - would need time-based analysis
            # Check for multiple signals aligning
            trend_momentum = self._trend_momentum_bullish_signal(
                observation, feature_names
            )
            oscillator_divergence = self._oscillator_divergence_bullish_signal(
                observation, feature_names
            )

            return trend_momentum and oscillator_divergence
        except (IndexError, ValueError, TypeError):
            return False

    def _price_channel_breakout_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect price breaking out of established channel."""
        try:
            # Simplified channel breakout detection
            # Check if price is above both SMA and upper BB
            golden_cross = self._golden_cross_signal(observation, feature_names)
            bb_upper_touch = self._bollinger_upper_touch_signal(
                observation, feature_names
            )

            # Additional confirmation from momentum
            macd_bullish = self._macd_bullish_signal(observation, feature_names)

            return golden_cross and bb_upper_touch and macd_bullish
        except (IndexError, ValueError, TypeError):
            return False

    def _reversal_candlestick_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect bullish reversal candlestick patterns."""
        try:
            # Simplified reversal detection
            # Look for oversold conditions with bullish momentum divergence
            oscillator_divergence = self._oscillator_divergence_bullish_signal(
                observation, feature_names
            )

            # Additional check for trend exhaustion (ADX high but weakening)
            adx_idx = next(
                (i for i, name in enumerate(feature_names) if "adx" in name.lower()),
                None,
            )
            adx_high = adx_idx is not None and float(observation[adx_idx]) > 30.0

            return oscillator_divergence and adx_high
        except (IndexError, ValueError, TypeError):
            return False

    def _momentum_divergence_signal(
        self, observation: np.ndarray, feature_names: List[str]
    ) -> bool:
        """Detect negative momentum divergence."""
        try:
            # Check for overbought conditions in multiple oscillators
            oscillator_overbought = self._oscillator_divergence_bearish_signal(
                observation, feature_names
            )

            # Check if trend is still bullish (potential divergence)
            plus_di_bullish = self._plus_di_bullish_signal(observation, feature_names)

            return oscillator_overbought and plus_di_bullish
        except (IndexError, ValueError, TypeError):
            return False
