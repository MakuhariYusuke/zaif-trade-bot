"""
Oscillator Pattern Recognizers - CCI, Stochastic, Williams %R, MFI

This module provides pattern recognition for oscillator-based technical indicators.
"""

from typing import Any, Dict, Optional

import pandas as pd

from .base import PatternRecognizer, SignalResult
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.features.oscillator.cci import compute_cci
from ztb.features.oscillator.stochastic import compute_stochastic
from ztb.features.momentum.williams_r import compute_williams_r
from ztb.features.volume.mfi import compute_mfi


class CCIRecognizer(PatternRecognizer):
    """
    Commodity Channel Index (CCI) pattern recognizer.
    Identifies overbought/oversold conditions and trend signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.overbought_level = self.config.get("overbought_level", 100)
        self.oversold_level = self.config.get("oversold_level", -100)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize CCI patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with CCI analysis
        """
        if index < 20:  # Need sufficient data for CCI calculation
            return SignalResult(
                signal_type="cci_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description="Insufficient data for CCI analysis",
                metadata={},
                validity_period=1,
                risk_level="low"
            )

        try:
            cci_series = compute_cci(data)
            current_cci = cci_series.iloc[index]

            # Determine signal based on CCI levels
            if current_cci >= self.overbought_level:
                # Overbought - potential sell signal
                strength = min(abs(current_cci) / 200, 1.0)  # Normalize strength
                return SignalResult(
                    signal_type="cci_overbought",
                    strength=strength,
                    direction=ACTION_SELL,  # Sell signal
                    description=f"CCI overbought at {current_cci:.2f}",
                    metadata={"cci_value": current_cci, "level": "overbought"},
                    validity_period=5,
                    risk_level="medium"
                )
            elif current_cci <= self.oversold_level:
                # Oversold - potential buy signal
                strength = min(abs(current_cci) / 200, 1.0)  # Normalize strength
                return SignalResult(
                    signal_type="cci_oversold",
                    strength=strength,
                    direction=ACTION_BUY,  # Buy signal
                    description=f"CCI oversold at {current_cci:.2f}",
                    metadata={"cci_value": current_cci, "level": "oversold"},
                    validity_period=5,
                    risk_level="medium"
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="cci_neutral",
                    strength=0.0,
                    direction=ACTION_HOLD,
                    description=f"CCI in neutral zone at {current_cci:.2f}",
                    metadata={"cci_value": current_cci, "level": "neutral"},
                    validity_period=1,
                    risk_level="low"
                )

        except Exception as e:
            return SignalResult(
                signal_type="cci_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"CCI calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low"
            )


class StochasticRecognizer(PatternRecognizer):
    """
    Stochastic Oscillator pattern recognizer.
    Identifies overbought/oversold conditions using %K and %D lines.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.overbought_level = self.config.get("overbought_level", 80)
        self.oversold_level = self.config.get("oversold_level", 20)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize Stochastic patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with Stochastic analysis
        """
        if index < 20:  # Need sufficient data for Stochastic calculation
            return SignalResult(
                signal_type="stoch_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description="Insufficient data for Stochastic analysis",
                metadata={},
                validity_period=1,
                risk_level="low"
            )

        try:
            stoch_df = compute_stochastic(data)
            current_k = stoch_df["stoch_k"].iloc[index]
            current_d = stoch_df["stoch_d"].iloc[index]

            # Determine signal based on Stochastic levels and crossovers
            if current_k >= self.overbought_level and current_d >= self.overbought_level:
                # Overbought - potential sell signal
                strength = min((current_k + current_d) / 160, 1.0)  # Normalize strength
                return SignalResult(
                    signal_type="stoch_overbought",
                    strength=strength,
                    direction=ACTION_SELL,  # Sell signal
                    description=f"Stochastic overbought: %K={current_k:.2f}, %D={current_d:.2f}",
                    metadata={"stoch_k": current_k, "stoch_d": current_d, "level": "overbought"},
                    validity_period=5,
                    risk_level="medium"
                )
            elif current_k <= self.oversold_level and current_d <= self.oversold_level:
                # Oversold - potential buy signal
                strength = min((80 - current_k + 80 - current_d) / 160, 1.0)  # Normalize strength
                return SignalResult(
                    signal_type="stoch_oversold",
                    strength=strength,
                    direction=ACTION_BUY,  # Buy signal
                    description=f"Stochastic oversold: %K={current_k:.2f}, %D={current_d:.2f}",
                    metadata={"stoch_k": current_k, "stoch_d": current_d, "level": "oversold"},
                    validity_period=5,
                    risk_level="medium"
                )
            elif current_k > current_d and index > 0:
                # %K crosses above %D - potential buy signal
                prev_k = stoch_df["stoch_k"].iloc[index-1]
                prev_d = stoch_df["stoch_d"].iloc[index-1]
                if prev_k <= prev_d:
                    return SignalResult(
                        signal_type="stoch_bullish_crossover",
                        strength=0.6,
                        direction=ACTION_BUY,  # Buy signal
                        description=f"Stochastic bullish crossover: %K crosses above %D",
                        metadata={"stoch_k": current_k, "stoch_d": current_d, "crossover": "bullish"},
                        validity_period=3,
                        risk_level="medium"
                    )
            elif current_k < current_d and index > 0:
                # %K crosses below %D - potential sell signal
                prev_k = stoch_df["stoch_k"].iloc[index-1]
                prev_d = stoch_df["stoch_d"].iloc[index-1]
                if prev_k >= prev_d:
                    return SignalResult(
                        signal_type="stoch_bearish_crossover",
                        strength=0.6,
                        direction=ACTION_SELL,  # Sell signal
                        description=f"Stochastic bearish crossover: %K crosses below %D",
                        metadata={"stoch_k": current_k, "stoch_d": current_d, "crossover": "bearish"},
                        validity_period=3,
                        risk_level="medium"
                    )

            # Neutral zone
            return SignalResult(
                signal_type="stoch_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Stochastic in neutral zone: %K={current_k:.2f}, %D={current_d:.2f}",
                metadata={"stoch_k": current_k, "stoch_d": current_d, "level": "neutral"},
                validity_period=1,
                risk_level="low"
            )

        except Exception as e:
            return SignalResult(
                signal_type="stoch_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Stochastic calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low"
            )


class WilliamsRRecognizer(PatternRecognizer):
    """
    Williams %R pattern recognizer.
    Identifies overbought/oversold conditions using Williams %R.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.overbought_level = self.config.get("overbought_level", -20)
        self.oversold_level = self.config.get("oversold_level", -80)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize Williams %R patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with Williams %R analysis
        """
        if index < 14:  # Need sufficient data for Williams %R calculation
            return SignalResult(
                signal_type="williams_r_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description="Insufficient data for Williams %R analysis",
                metadata={},
                validity_period=1,
                risk_level="low"
            )

        try:
            williams_r_series = compute_williams_r(data)
            current_r = williams_r_series.iloc[index]

            # Determine signal based on Williams %R levels
            if current_r >= self.overbought_level:
                # Overbought - potential sell signal
                strength = min(current_r / 20, 1.0)  # Normalize strength (higher %R = stronger signal)
                return SignalResult(
                    signal_type="williams_r_overbought",
                    strength=strength,
                    direction=ACTION_SELL,  # Sell signal
                    description=f"Williams %R overbought at {current_r:.2f}",
                    metadata={"williams_r": current_r, "level": "overbought"},
                    validity_period=5,
                    risk_level="medium"
                )
            elif current_r <= self.oversold_level:
                # Oversold - potential buy signal
                strength = min(abs(current_r) / 80, 1.0)  # Normalize strength (lower %R = stronger signal)
                return SignalResult(
                    signal_type="williams_r_oversold",
                    strength=strength,
                    direction=ACTION_BUY,  # Buy signal
                    description=f"Williams %R oversold at {current_r:.2f}",
                    metadata={"williams_r": current_r, "level": "oversold"},
                    validity_period=5,
                    risk_level="medium"
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="williams_r_neutral",
                    strength=0.0,
                    direction=ACTION_HOLD,
                    description=f"Williams %R in neutral zone at {current_r:.2f}",
                    metadata={"williams_r": current_r, "level": "neutral"},
                    validity_period=1,
                    risk_level="low"
                )

        except Exception as e:
            return SignalResult(
                signal_type="williams_r_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"Williams %R calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low"
            )


class MFIRecognizer(PatternRecognizer):
    """
    Money Flow Index (MFI) pattern recognizer.
    Identifies overbought/oversold conditions using MFI.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.overbought_level = self.config.get("overbought_level", 80)
        self.oversold_level = self.config.get("oversold_level", 20)

    def recognize(self, data: pd.DataFrame, index: int = -1) -> Optional[SignalResult]:
        """
        Recognize MFI patterns.

        Args:
            data: OHLCV DataFrame
            index: Current index to analyze

        Returns:
            SignalResult with MFI analysis
        """
        if index < 14:  # Need sufficient data for MFI calculation
            return SignalResult(
                signal_type="mfi_neutral",
                strength=0.0,
                direction=ACTION_HOLD,
                description="Insufficient data for MFI analysis",
                metadata={},
                validity_period=1,
                risk_level="low"
            )

        try:
            mfi_series = compute_mfi(data)
            current_mfi = mfi_series.iloc[index]

            # Determine signal based on MFI levels
            if current_mfi >= self.overbought_level:
                # Overbought - potential sell signal
                strength = min(current_mfi / 100, 1.0)  # Normalize strength
                return SignalResult(
                    signal_type="mfi_overbought",
                    strength=strength,
                    direction=ACTION_SELL,  # Sell signal
                    description=f"MFI overbought at {current_mfi:.2f}",
                    metadata={"mfi_value": current_mfi, "level": "overbought"},
                    validity_period=5,
                    risk_level="medium"
                )
            elif current_mfi <= self.oversold_level:
                # Oversold - potential buy signal
                strength = min((100 - current_mfi) / 80, 1.0)  # Normalize strength
                return SignalResult(
                    signal_type="mfi_oversold",
                    strength=strength,
                    direction=ACTION_BUY,  # Buy signal
                    description=f"MFI oversold at {current_mfi:.2f}",
                    metadata={"mfi_value": current_mfi, "level": "oversold"},
                    validity_period=5,
                    risk_level="medium"
                )
            else:
                # Neutral zone
                return SignalResult(
                    signal_type="mfi_neutral",
                    strength=0.0,
                    direction=ACTION_HOLD,
                    description=f"MFI in neutral zone at {current_mfi:.2f}",
                    metadata={"mfi_value": current_mfi, "level": "neutral"},
                    validity_period=1,
                    risk_level="low"
                )

        except Exception as e:
            return SignalResult(
                signal_type="mfi_error",
                strength=0.0,
                direction=ACTION_HOLD,
                description=f"MFI calculation error: {str(e)}",
                metadata={"error": str(e)},
                validity_period=1,
                risk_level="low"
            )