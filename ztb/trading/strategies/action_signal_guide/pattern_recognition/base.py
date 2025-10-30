"""
Base classes for pattern recognition in Action Signal Guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.utils.exceptions.custom_exceptions import DataError, ValidationError


class SignalResult:
    """Result of a pattern recognition signal."""

    def __init__(
        self,
        signal_type: str,
        strength: float,
        direction: float,  # Continuous value from -1.0 (strong sell) to 1.0 (strong buy), 0.0 for neutral
        description: str,
        timestamp: Optional[Any] = None,
        confidence: Optional[
            float
        ] = None,  # Confidence in pattern recognition (0.0-1.0), defaults to strength
        metadata: Optional[Dict[str, Any]] = None,
        validity_period: int = 5,  # How many periods this signal is valid (required)
        risk_level: str = "medium",  # 'low', 'medium', 'high' (required)
    ) -> None:
        self.signal_type = signal_type
        self.strength = strength
        self.direction = direction
        self.description = description
        self.timestamp = timestamp
        # Confidence defaults to strength if not provided, ensuring it's always a float
        self.confidence = confidence if confidence is not None else strength
        self.metadata = metadata or {}
        self.validity_period = validity_period
        self.risk_level = risk_level

        # Validate inputs
        if not isinstance(self.confidence, (int, float)) or not (
            0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                f"Confidence must be a float between 0.0 and 1.0, got {self.confidence}"
            )
        if not isinstance(self.strength, (int, float)) or not (
            0.0 <= self.strength <= 1.0
        ):
            raise ValueError(
                f"Strength must be a float between 0.0 and 1.0, got {self.strength}"
            )
        if not isinstance(self.direction, (int, float)) or not (
            -1.0 <= self.direction <= 1.0
        ):
            raise ValueError(
                f"Direction must be a float between -1.0 and 1.0, got {self.direction}"
            )
        if self.validity_period <= 0:
            raise ValueError(
                f"Validity period must be positive, got {self.validity_period}"
            )
        if self.risk_level not in ["low", "medium", "high"]:
            raise ValueError(
                f"Risk level must be 'low', 'medium', or 'high', got {self.risk_level}"
            )

    @property
    def confidence_score(self) -> float:
        """Alias for confidence to make it clear this is the confidence score."""
        return self.confidence

    @property
    def signal_strength(self) -> float:
        """Alias for strength to clarify the difference from confidence."""
        return self.strength

    def is_expired(self, current_index: int, signal_index: int) -> bool:
        """Check if signal has expired based on validity period."""
        return (current_index - signal_index) >= self.validity_period

    def get_risk_multiplier(self) -> float:
        """Get risk multiplier based on risk level."""
        risk_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        return risk_multipliers.get(self.risk_level, 1.0)


class PatternRecognizer(ABC):
    """
    Base class for all pattern recognizers.

    Provides common functionality and interface for pattern recognition.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self._validate_config()
        self._signal_cache: Dict[int, Optional[SignalResult]] = {}

    def _validate_config(self) -> None:
        """Validate configuration parameters with enhanced type checking."""
        # Basic validation - can be overridden by subclasses
        if "enabled" in self.config and not isinstance(self.config["enabled"], bool):
            raise ValueError(f"Config 'enabled' must be boolean for {self.name}")

        # Validate confidence thresholds
        if "min_confidence" in self.config:
            min_conf = self.config["min_confidence"]
            if not isinstance(min_conf, (int, float)) or not (0.0 <= min_conf <= 1.0):
                raise ValueError(
                    f"Config 'min_confidence' must be float between 0.0 and 1.0 for {self.name}, "
                    f"got {min_conf}"
                )

        # Validate lookback periods
        if "lookback_period" in self.config:
            lookback = self.config["lookback_period"]
            if not isinstance(lookback, int) or lookback <= 0:
                raise ValueError(
                    f"Config 'lookback_period' must be positive integer for {self.name}, "
                    f"got {lookback}"
                )

        # Validate risk levels
        if "risk_level" in self.config:
            risk_level = self.config["risk_level"]
            valid_risks = ["low", "medium", "high"]
            if risk_level not in valid_risks:
                raise ValueError(
                    f"Config 'risk_level' must be one of {valid_risks} for {self.name}, "
                    f"got '{risk_level}'"
                )

        # Validate numeric thresholds
        numeric_configs = [
            "body_ratio_threshold",
            "shadow_ratio_threshold",
            "min_trend_length",
            "engulfing_ratio_threshold",
            "piercing_ratio_threshold",
        ]
        for config_key in numeric_configs:
            if config_key in self.config:
                value = self.config[config_key]
                if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                    raise ValueError(
                        f"Config '{config_key}' must be float between 0.0 and 1.0 for {self.name}, "
                        f"got {value}"
                    )

        # Validate risk levels
        if "risk_level" in self.config:
            risk_level = self.config["risk_level"]
            if risk_level not in ["low", "medium", "high"]:
                raise ValueError(
                    f"Config 'risk_level' must be 'low', 'medium', or 'high' for {self.name}"
                )

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def is_enabled(self) -> bool:
        """Check if this recognizer is enabled."""
        return self.get_config_value("enabled", True)

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self.get_config_value("min_confidence", 0.0)

    def get_lookback_period(self) -> int:
        """Get lookback period for analysis."""
        return self.get_config_value("lookback_period", 20)

    def get_risk_level(self) -> str:
        """Get risk level for this recognizer."""
        return self.get_config_value("risk_level", "medium")

    def _validate_input_data(self, data: pd.DataFrame, index: int) -> None:
        """Validate input data and index for pattern recognition."""
        if data is None or data.empty:
            raise ValueError(f"Data cannot be None or empty for {self.name}")

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns {missing_columns} for {self.name}"
            )

        if index < 0:
            index = len(data) + index

        if index < 0 or index >= len(data):
            raise ValueError(
                f"Index {index} out of bounds for data length {len(data)} in {self.name}"
            )

        # Check for minimum data length
        min_length = self.get_lookback_period() + 5  # Extra buffer
        if len(data) < min_length:
            raise ValueError(
                f"Insufficient data length {len(data)}, need at least {min_length} for {self.name}"
            )

    def validate_recognition_inputs(
        self,
        data: pd.DataFrame,
        index: int = -1,
        required_length: int = 1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Common validation for pattern recognition inputs.

        Args:
            data: OHLCV data as pandas DataFrame
            index: Index to check for pattern (default: last row)
            required_length: Minimum data length required for this pattern
            multi_timeframe_data: Optional multi-timeframe data

        Returns:
            Validated index (adjusted for negative indexing)

        Raises:
            ValidationError: If inputs are invalid
            DataError: If data is invalid
        """
        # Check if recognizer is enabled
        if not self.is_enabled():
            raise ValidationError(
                f"Pattern recognizer {self.name} is disabled",
                details={"recognizer": self.name, "enabled": False},
            )

        # Validate data
        try:
            self._validate_input_data(data, index)
        except ValueError as e:
            raise DataError(
                f"Invalid input data for pattern {self.name}: {e}",
                details={"recognizer": self.name, "error": str(e)},
            ) from e

        # Adjust negative index
        if index < 0:
            index = len(data) + index

        # Check minimum required length for this specific pattern
        if len(data) < required_length:
            raise DataError(
                f"Insufficient data length for pattern {self.name}",
                details={
                    "recognizer": self.name,
                    "required_length": required_length,
                    "actual_length": len(data),
                },
            )

        # Validate multi-timeframe data if required
        if multi_timeframe_data is not None:
            self._validate_multi_timeframe_data(multi_timeframe_data)

        return index

    def _validate_multi_timeframe_data(
        self, multi_timeframe_data: Dict[str, Any]
    ) -> None:
        """Validate multi-timeframe data structure."""
        if not isinstance(multi_timeframe_data, dict):
            raise ValueError(
                f"Multi-timeframe data must be a dictionary for {self.name}"
            )

        # Basic validation - can be extended by subclasses
        pass

    @abstractmethod
    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize pattern in the given data at the specified index.

        Args:
            data: OHLCV data as pandas DataFrame
            index: Index to check for pattern (default: last row)

        Returns:
            SignalResult if pattern found, None otherwise
        """
        pass

    def recognize_with_cache(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize pattern with caching to avoid redundant calculations.

        Args:
            data: OHLCV data as pandas DataFrame
            index: Index to check for pattern

        Returns:
            Cached SignalResult if available and valid, otherwise new recognition
        """
        cache_key = hash(
            (self.name, index, data.iloc[index]["close"] if index >= 0 else 0)
        )

        if cache_key in self._signal_cache:
            cached_signal = self._signal_cache[cache_key]
            if cached_signal and not cached_signal.is_expired(index, index):
                return cached_signal

        # Calculate new signal
        signal = self.recognize(data, index, multi_timeframe_data)
        self._signal_cache[cache_key] = signal
        return signal

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid
        """
        required_columns = ["open", "high", "low", "close"]
        return all(col in data.columns for col in required_columns)

    def get_lookback_period(self) -> int:
        """
        Get the number of periods this pattern needs to look back.

        Returns:
            Number of periods required for pattern recognition
        """
        return int(self.config.get("lookback_period", 20))

    def calculate_body_size(self, data: pd.DataFrame, index: int) -> float:
        """Calculate candle body size."""
        candle = data.iloc[index]
        close_val = cast(float, candle["close"])
        open_val = cast(float, candle["open"])
        return float(abs(close_val - open_val))

    def calculate_upper_shadow(self, data: pd.DataFrame, index: int) -> float:
        """Calculate upper shadow size."""
        candle = data.iloc[index]
        high_val = cast(float, candle["high"])
        open_val = cast(float, candle["open"])
        close_val = cast(float, candle["close"])
        return float(high_val - max(open_val, close_val))

    def calculate_lower_shadow(self, data: pd.DataFrame, index: int) -> float:
        """Calculate lower shadow size."""
        candle = data.iloc[index]
        low_val = cast(float, candle["low"])
        open_val = cast(float, candle["open"])
        close_val = cast(float, candle["close"])
        return float(abs(min(open_val, close_val) - low_val))

    def _calculate_pattern_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        pattern_factors: Dict[str, float],
        base_confidence: float = 0.5,
    ) -> float:
        """
        Calculate dynamic confidence score for patterns.

        Args:
            data: Price data
            index: Current index
            pattern_factors: Dictionary of pattern quality factors (0.0-1.0)
            base_confidence: Base confidence level

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not pattern_factors:
            return base_confidence

        # Calculate weighted average of pattern factors
        weights = {
            "trend_strength": 0.3,
            "candle_size": 0.25,
            "price_movement": 0.25,
            "pattern_completeness": 0.2,
        }

        confidence = base_confidence
        total_weight = 0.0

        for factor_name, factor_value in pattern_factors.items():
            if factor_name in weights:
                weight = weights[factor_name]
                confidence += factor_value * weight
                total_weight += weight

        # Normalize to ensure result is between 0.0 and 1.0
        if total_weight > 0:
            confidence = float(min(1.0, max(0.0, confidence)))

        return confidence

    def _calculate_trend_strength(
        self, data: pd.DataFrame, index: int, lookback: int = 10
    ) -> float:
        """
        Calculate trend strength on a scale of 0.0 to 1.0.

        Returns:
            Trend strength: 0.0 (no trend) to 1.0 (strong trend)
        """
        if index < lookback:
            return 0.0

        prices = data.iloc[index - lookback + 1 : index + 1]["close"].values
        if len(prices) < 3:
            return 0.0

        # Calculate linear trend using least squares
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        # Calculate R-squared to measure trend strength
        y_mean = np.mean(prices)
        ss_tot = np.sum((prices - y_mean) ** 2)
        ss_res = np.sum((prices - (slope * x + prices[0])) ** 2)

        if ss_tot == 0:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)

        # Convert slope to strength (absolute value, normalized)
        avg_price = np.mean(prices)
        slope_strength = min(
            1.0, abs(slope) / (avg_price * 0.01)
        )  # 1% of average price as strong slope

        # Combine R-squared and slope strength
        trend_strength = r_squared * 0.7 + slope_strength * 0.3

        return min(1.0, max(0.0, trend_strength))

    def _calculate_candle_size_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        expected_body_ratio: float = 0.5,
        lookback: int = 20,
    ) -> float:
        """
        Calculate confidence based on candle body size relative to historical average.

        Args:
            expected_body_ratio: Expected body size ratio (0.0-1.0)
            lookback: Period to calculate historical average

        Returns:
            Confidence score: 0.0 (poor match) to 1.0 (perfect match)
        """
        if index < lookback:
            return 0.5

        current_body = self.calculate_body_size(data, index)
        avg_body = self._get_average_body_size(data, index, lookback)

        if avg_body == 0:
            return 0.5

        # Calculate how close current body is to expected ratio of average
        expected_body = avg_body * expected_body_ratio
        body_ratio = min(current_body, expected_body * 2) / max(
            current_body, expected_body * 2
        )

        # Perfect match = 1.0, poor match = 0.0
        confidence = 1.0 - abs(current_body - expected_body) / max(
            current_body, expected_body
        )

        return min(1.0, max(0.0, confidence))

    def _calculate_price_movement_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        expected_movement: float,
        lookback: int = 20,
    ) -> float:
        """
        Calculate confidence based on price movement magnitude.

        Args:
            expected_movement: Expected price movement as fraction of recent volatility

        Returns:
            Confidence score: 0.0 (no movement) to 1.0 (strong movement)
        """
        if index < 1:
            return 0.0

        # Calculate recent volatility (standard deviation of returns)
        if index >= lookback:
            recent_prices = data.iloc[index - lookback + 1 : index + 1]["close"].values
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
        else:
            volatility = 0.01  # Default 1% volatility

        # Calculate actual price movement
        prev_close = data.iloc[index - 1]["close"]
        curr_close = data.iloc[index]["close"]
        movement = abs(curr_close - prev_close) / prev_close

        if volatility == 0:
            return 0.5

        # Normalize movement by volatility
        normalized_movement = movement / volatility

        # Expected movement should be significant but not extreme
        if expected_movement <= 0:
            expected_movement = 0.5

        # Calculate confidence based on how close movement is to expected
        movement_ratio = min(normalized_movement, expected_movement * 2) / max(
            normalized_movement, expected_movement * 2
        )
        confidence = 1.0 - abs(normalized_movement - expected_movement) / max(
            normalized_movement, expected_movement
        )

        return min(1.0, max(0.0, confidence))

    def _get_average_body_size(
        self, data: pd.DataFrame, index: int, lookback: int
    ) -> float:
        """Calculate average body size over lookback period."""
        if index < lookback:
            return 0
        bodies = [
            self.calculate_body_size(data, i)
            for i in range(index - lookback + 1, index + 1)
        ]
        return cast(float, np.mean(bodies)) if bodies else 0

    def is_bullish_candle(
        self, data: Union[pd.DataFrame, pd.Series], index: Optional[int] = None
    ) -> bool:
        """Check if candle is bullish."""
        if isinstance(data, pd.Series):
            # If data is a Series (single candle), check directly
            return cast(bool, data["close"] > data["open"])
        elif index is not None:
            # If data is DataFrame and index is provided
            return cast(bool, data.iloc[index]["close"] > data.iloc[index]["open"])
        else:
            raise ValueError("Either provide a Series or DataFrame with index")

    def is_bearish_candle(
        self, data: Union[pd.DataFrame, pd.Series], index: Optional[int] = None
    ) -> bool:
        """Check if candle is bearish."""
        if isinstance(data, pd.Series):
            # If data is a Series (single candle), check directly
            return cast(bool, data["close"] < data["open"])
        elif index is not None:
            # If data is DataFrame and index is provided
            return cast(bool, data.iloc[index]["close"] < data.iloc[index]["open"])
        else:
            raise ValueError("Either provide a Series or DataFrame with index")

    def get_body_ratio(self, data: pd.DataFrame, index: int) -> float:
        """Get body size as ratio of total range."""
        high = cast(float, data.iloc[index]["high"])
        low = cast(float, data.iloc[index]["low"])
        total_range = high - low
        if total_range == 0:
            return 0.0
        return self.calculate_body_size(data, index) / total_range


class CandlestickPatternRecognizer(PatternRecognizer):
    """
    Base class for candlestick pattern recognizers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.body_ratio_threshold = self.config.get("body_ratio_threshold", 0.6)
        self.shadow_ratio_threshold = self.config.get("shadow_ratio_threshold", 0.3)

    def is_hammer_like(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle resembles hammer pattern."""
        if not self.validate_data(data) or index < 0 or index >= len(data):
            return False

        body_ratio = self.get_body_ratio(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        total_range = cast(float, data.iloc[index]["high"] - data.iloc[index]["low"])

        if total_range == 0:
            return False

        # Hammer characteristics: small body, long lower shadow, small upper shadow
        return cast(
            bool,
            body_ratio < self.body_ratio_threshold
            and lower_shadow > upper_shadow * 2
            and lower_shadow > body_ratio * total_range,
        )

    def is_shooting_star_like(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle resembles shooting star pattern."""
        if not self.validate_data(data) or index < 0:
            return False

        body_ratio = self.get_body_ratio(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        total_range = cast(float, data.iloc[index]["high"] - data.iloc[index]["low"])

        if total_range == 0:
            return False

        # Shooting star characteristics: small body, long upper shadow, small lower shadow
        return cast(
            bool,
            body_ratio < self.body_ratio_threshold
            and upper_shadow > lower_shadow * 2
            and upper_shadow > body_ratio * total_range,
        )

    def _is_uptrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's an uptrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index - lookback + 1 : index + 1]["close"]
        return cast(bool, recent_prices.iloc[-1] > recent_prices.iloc[0])

    def _is_downtrend(self, data: pd.DataFrame, index: int, lookback: int) -> bool:
        """Check if there's a downtrend over the lookback period."""
        if index < lookback:
            return False
        recent_prices = data.iloc[index - lookback + 1 : index + 1]["close"]
        return cast(bool, recent_prices.iloc[-1] < recent_prices.iloc[0])

    def _is_small_candle(self, candle: pd.Series) -> bool:
        """Check if candle has small body relative to recent volatility."""
        body_size = cast(float, abs(candle["close"] - candle["open"]))
        total_range = cast(float, candle["high"] - candle["low"])
        return body_size / total_range < 0.3 if total_range > 0 else False

    def _is_large_candle(self, candle: pd.Series) -> bool:
        """Check if candle has large body relative to recent volatility."""
        body_size = cast(float, abs(candle["close"] - candle["open"]))
        total_range = cast(float, candle["high"] - candle["low"])
        return body_size / total_range > 0.6 if total_range > 0 else False

    def analyze_candle_characteristics(
        self, data: pd.DataFrame, index: int
    ) -> Dict[str, float]:
        """
        Analyze comprehensive candle characteristics for pattern recognition.

        Args:
            data: OHLCV data
            index: Candle index to analyze

        Returns:
            Dictionary with candle characteristics
        """
        if not self.validate_data(data) or index < 0 or index >= len(data):
            return {}

        candle = data.iloc[index]
        body_size = self.calculate_body_size(data, index)
        upper_shadow = self.calculate_upper_shadow(data, index)
        lower_shadow = self.calculate_lower_shadow(data, index)
        total_range = cast(float, candle["high"] - candle["low"])

        if total_range == 0:
            return {
                "body_ratio": 0.0,
                "upper_shadow_ratio": 0.0,
                "lower_shadow_ratio": 0.0,
                "body_size": body_size,
                "is_bullish": False,
                "is_bearish": False,
            }

        return {
            "body_ratio": body_size / total_range,
            "upper_shadow_ratio": upper_shadow / total_range,
            "lower_shadow_ratio": lower_shadow / total_range,
            "body_size": body_size,
            "upper_shadow_size": upper_shadow,
            "lower_shadow_size": lower_shadow,
            "total_range": total_range,
            "is_bullish": self.is_bullish_candle(data, index),
            "is_bearish": self.is_bearish_candle(data, index),
            "is_doji": body_size / total_range < 0.05,  # Very small body
            "is_marubozu": body_size / total_range > 0.95,  # Very large body
        }

    def analyze_multiple_candle_characteristics(
        self, data: pd.DataFrame, indices: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze characteristics for multiple candles.

        Args:
            data: OHLCV data
            indices: List of candle indices to analyze

        Returns:
            Dictionary with lists of candle characteristics
        """
        characteristics: Dict[str, Any] = {
            "body_sizes": [],
            "body_ratios": [],
            "upper_shadow_ratios": [],
            "lower_shadow_ratios": [],
            "is_bullish": [],
            "is_bearish": [],
            "avg_body_size": 0.0,
        }

        if not indices:
            return characteristics

        # Calculate average body size first
        body_sizes = []
        for idx in indices:
            if self.validate_data(data) and 0 <= idx < len(data):
                body_sizes.append(self.calculate_body_size(data, idx))
            else:
                body_sizes.append(0.0)

        characteristics["avg_body_size"] = (
            sum(body_sizes) / len(body_sizes) if body_sizes else 0.0
        )

        # Analyze each candle
        for idx in indices:
            if not self.validate_data(data) or idx < 0 or idx >= len(data):
                # Add default values for invalid indices
                characteristics["body_sizes"].append(0.0)
                characteristics["body_ratios"].append(0.0)
                characteristics["upper_shadow_ratios"].append(0.0)
                characteristics["lower_shadow_ratios"].append(0.0)
                characteristics["is_bullish"].append(False)
                characteristics["is_bearish"].append(False)
                continue

            candle = data.iloc[idx]
            body_size = self.calculate_body_size(data, idx)
            upper_shadow = self.calculate_upper_shadow(data, idx)
            lower_shadow = self.calculate_lower_shadow(data, idx)
            total_range = cast(float, candle["high"] - candle["low"])

            characteristics["body_sizes"].append(body_size)
            characteristics["is_bullish"].append(self.is_bullish_candle(data, idx))
            characteristics["is_bearish"].append(self.is_bearish_candle(data, idx))

            if total_range > 0:
                characteristics["body_ratios"].append(body_size / total_range)
                characteristics["upper_shadow_ratios"].append(
                    upper_shadow / total_range
                )
                characteristics["lower_shadow_ratios"].append(
                    lower_shadow / total_range
                )
            else:
                characteristics["body_ratios"].append(0.0)
                characteristics["upper_shadow_ratios"].append(0.0)
                characteristics["lower_shadow_ratios"].append(0.0)

        return characteristics

    def validate_pattern_structure(
        self, data: pd.DataFrame, indices: List[int], expected_directions: List[str]
    ) -> bool:
        """
        Validate that candles at given indices match expected directions.

        Args:
            data: OHLCV data
            indices: List of indices to check
            expected_directions: List of expected directions ("bullish", "bearish", "any")

        Returns:
            True if all candles match expected directions
        """
        if len(indices) != len(expected_directions):
            return False

        for idx, direction in zip(indices, expected_directions):
            if not self.validate_data(data) or idx < 0 or idx >= len(data):
                return False

            if direction == "bullish" and not self.is_bullish_candle(data, idx):
                return False
            elif direction == "bearish" and not self.is_bearish_candle(data, idx):
                return False
            elif direction == "any":
                continue  # Accept any direction

        return True


class MultiCandlePatternRecognizer(PatternRecognizer):
    """
    Base class for multi-candle pattern recognizers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.min_trend_length = self.config.get("min_trend_length", 5)

    def detect_trend(self, data: pd.DataFrame, start_index: int, length: int) -> int:
        """
        Detect trend direction over a period.

        Returns:
            1 for uptrend, -1 for downtrend, 0 for sideways
        """
        if start_index - length + 1 < 0 or start_index >= len(data):
            return 0

        prices = data.iloc[start_index - length + 1 : start_index + 1]["close"].values
        if len(prices) < length:
            return 0

        # Lightweight trend detection using price difference
        diff = prices[-1] - prices[0]

        if diff > 0.001:  # Uptrend threshold
            return ACTION_BUY
        elif diff < -0.001:  # Downtrend threshold
            return ACTION_SELL
        else:
            return ACTION_HOLD
