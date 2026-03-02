"""
RSI (Relative Strength Index) Indicator

Calculates RSI oscillator for momentum analysis.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ztb.trading.signal.quality.indicators.base import BaseOscillatorIndicator
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class RSIIndicator(BaseOscillatorIndicator):
    """
    RSI (Relative Strength Index) Indicator

    Measures the speed and change of price movements to evaluate
    overbought or oversold conditions.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.periods = self._get_config_int("periods", 14, minimum=1)

    def on_config_updated(self) -> None:
        """Sync derived fields when config is updated dynamically."""
        self.periods = self._get_config_int("periods", 14, minimum=1)

    def _get_default_config(self) -> dict[str, object]:
        return {"periods": 14, "smoothing": "ema"}  # 'ema' or 'sma'

    def _calculate_indicator(self, data: pd.DataFrame) -> dict[str, object]:
        """Calculate RSI values"""
        close = data["close"]

        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        if self.config.get("smoothing", "ema") == "ema":
            avg_gain = gain.ewm(span=self.periods, adjust=False).mean()
            avg_loss = loss.ewm(span=self.periods, adjust=False).mean()
        else:
            avg_gain = gain.rolling(window=self.periods).mean()
            avg_loss = loss.rolling(window=self.periods).mean()

        # Compute both SMA and EWM RS/RSI for stable short-data behavior
        sma_gain = gain.rolling(window=self.periods, min_periods=1).mean()
        sma_loss = loss.rolling(window=self.periods, min_periods=1).mean()

        with np.errstate(divide="ignore", invalid="ignore"):
            sma_rs = sma_gain.div(sma_loss.replace({0: np.nan}))
            sma_rsi = 100.0 - (100.0 / (1.0 + sma_rs))

            rs = avg_gain.div(avg_loss.replace({0: np.nan}))
            ewm_rsi = 100.0 - (100.0 / (1.0 + rs))

        # Prefer SMA for short data and fall back to EWM where SMA undefined
        rsi_series = sma_rsi.where(~sma_rsi.isna(), ewm_rsi)

        # Fill NaNs with neutral value 50 where both gain/loss are zero
        # Use tolerant equality checks to handle small floating point values
        zero_gain = np.isclose(avg_gain, 0.0)
        zero_loss = np.isclose(avg_loss, 0.0)
        neutral_mask = zero_gain & zero_loss
        rsi_series = rsi_series.where(~neutral_mask, 50.0)

        # For rows where loss ~ 0 and gain > 0 -> RSI 100
        rsi_series = rsi_series.where(~(zero_loss & (avg_gain > 0)), 100.0)
        # For rows where gain ~ 0 and loss > 0 -> RSI 0
        rsi_series = rsi_series.where(~(zero_gain & (avg_loss > 0)), 0.0)

        rsi = rsi_series

        # Debugging: log last avg_gain/avg_loss and current rsi for edge-case diagnosis
        try:
            avg_gain_last = float(avg_gain.iloc[-1]) if not avg_gain.empty else 0.0
            avg_loss_last = float(avg_loss.iloc[-1]) if not avg_loss.empty else 0.0
            logger.debug(
                f"RSI debug: avg_gain_last={avg_gain_last:.6f}, avg_loss_last={avg_loss_last:.6f}, current_rsi={float(rsi.iloc[-1]) if not rsi.empty else 'NA'}"
            )
        except Exception:
            pass

        # Get current RSI value (last value)
        current_rsi = (
            float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
        )

        # Validate and clamp
        current_rsi = self._validate_oscillator_value(current_rsi)

        # Calculate RSI signal strength
        signal = self.get_oscillator_signal(current_rsi)

        # Calculate RSI slope (momentum)
        rsi_slope = rsi.diff().iloc[-1] if len(rsi) > 1 else 0.0

        return {
            "rsi": current_rsi,
            "rsi_signal": signal,
            "rsi_slope": rsi_slope,
            "avg_gain": avg_gain.iloc[-1] if not avg_gain.empty else 0.0,
            "avg_loss": avg_loss.iloc[-1] if not avg_loss.empty else 0.0,
        }

    def _get_default_values(self) -> dict[str, object]:
        """Get default values when calculation fails"""
        return {
            "rsi": 50.0,
            "rsi_signal": "neutral",
            "rsi_slope": 0.0,
            "avg_gain": 0.0,
            "avg_loss": 0.0,
        }

    def get_required_periods(self) -> int:
        """Get minimum periods required"""
        return self.periods + 1  # Need one extra for diff
