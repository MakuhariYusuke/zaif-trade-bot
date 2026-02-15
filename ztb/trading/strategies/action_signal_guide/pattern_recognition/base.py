"""
Base classes for pattern recognition in Action Signal Guide.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import pandas as pd


F = TypeVar("F", bound=Callable[..., object])
T = TypeVar("T")


def timed(func: F) -> F:
    """Simple timing decorator for performance monitoring."""

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return cast(F, wrapper)


from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.utils.exceptions.custom_exceptions import ValidationError
from ztb.utils.safety import (
    safe_config_get,
    safe_config_get_bool,
    safe_config_get_float,
    safe_config_get_str,
)
from ztb.utils.type_validation import TypeValidator

try:
    from ztb.trading.strategies.action_signal_guide.types import (
        AnalysisResult,
        MultiTimeframeAnalysis,
        MultiTimeframeData,
        PatternMetrics,
        PatternResult,
        PatternThresholds,
        RegimeAdjustment,
        SignalMetadata,
    )
except ImportError:
    # Fallback if types module not available
    if TYPE_CHECKING:
        MultiTimeframeData = Dict[str, Dict[str, object]]
        PatternThresholds = Dict[str, Union[int, float, None]]
        PatternMetrics = Dict[str, Union[int, float, str]]
        PatternResult = Dict[str, Union[int, float, str, PatternMetrics]]
        SignalMetadata = Dict[
            str, Union[int, float, str, bool, List[Union[int, float]]]
        ]
        AnalysisResult = Dict[str, Union[float, str, bool, Dict[str, float]]]
        MultiTimeframeAnalysis = Dict[str, AnalysisResult]
        RegimeAdjustment = Dict[str, Union[int, float, str]]
    else:
        # Runtime fallbacks
        MultiTimeframeData = Dict[str, Dict[str, object]]
        PatternThresholds = Dict[str, Union[int, float, None]]
        PatternMetrics = Dict[str, Union[int, float, str]]
        PatternResult = Dict[str, Union[int, float, str, PatternMetrics]]
        SignalMetadata = Dict[
            str, Union[int, float, str, bool, List[Union[int, float]]]
        ]
        AnalysisResult = Dict[str, Union[float, str, bool, Dict[str, float]]]
        MultiTimeframeAnalysis = Dict[str, AnalysisResult]
        RegimeAdjustment = Dict[str, Union[int, float, str]]


class LRUCache(Generic[T]):
    """
    Simple LRU cache implementation with size limits.

    Provides O(1) access time and automatic cleanup of old entries.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, T] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[T]:
        """Get value from cache, moving it to end (most recently used)."""
        with self._lock:
            # Check TTL if enabled
            if self.ttl_seconds and key in self.timestamps:
                if time.time() - self.timestamps[key] > self.ttl_seconds:
                    self._remove(key)
                    return None

            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: T) -> None:
        """Set value in cache with LRU eviction."""
        with self._lock:
            if key in self.cache:
                # Update existing entry
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new entry
                self.cache[key] = value
                if len(self.cache) > self.max_size:
                    # Remove least recently used
                    oldest_key, _ = self.cache.popitem(last=False)
                    self.timestamps.pop(oldest_key, None)

            # Update timestamp
            if self.ttl_seconds:
                self.timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """Remove key from cache."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def cleanup_expired(self) -> int:
        """Remove expired entries, return number removed."""
        if not self.ttl_seconds:
            return 0

        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl_seconds
            ]

            for key in expired_keys:
                self._remove(key)

            return len(expired_keys)


def preprocess_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    method: str = "robust",
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Preprocess features for pattern recognition.

    Args:
        data: Input DataFrame with features
        feature_columns: Columns to preprocess
        method: Normalization method ('standard', 'robust', 'minmax')
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier removal

    Returns:
        Preprocessed DataFrame
    """
    processed_data = data.copy()

    for col in feature_columns:
        if col not in processed_data.columns:
            continue

        series = processed_data[col].astype(float)

        # Remove outliers using IQR method
        if remove_outliers and len(series) > 10:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            series = series.clip(lower_bound, upper_bound)

        # Apply normalization
        if method == "standard":
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                series = (series - mean_val) / std_val
        elif method == "robust":
            median_val = series.median()
            mad_val = (series - median_val).abs().median()
            if mad_val > 0:
                series = (series - median_val) / mad_val
        elif method == "minmax":
            min_val = series.min()
            max_val = series.max()
            if max_val > min_val:
                series = (series - min_val) / (max_val - min_val)

        processed_data[col] = series

    return processed_data


def calculate_technical_features(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate additional technical features for pattern recognition.

    Args:
        data: OHLCV DataFrame
        window: Rolling window size

    Returns:
        DataFrame with additional technical features
    """
    df = data.copy()

    # Price-based features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility features
    df["volatility"] = df["returns"].rolling(window=window).std()
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]

    # Volume features
    if "volume" in df.columns:
        df["volume_ma"] = df["volume"].rolling(window=window).mean()
        df["volume_std"] = df["volume"].rolling(window=window).std()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

    # Momentum features
    df["momentum"] = df["close"] / df["close"].shift(window)
    df["roc"] = df["close"].pct_change(periods=window)

    # Trend features
    df["sma_short"] = df["close"].rolling(window=window // 2).mean()
    df["sma_long"] = df["close"].rolling(window=window).mean()
    df["trend_strength"] = (df["sma_short"] - df["sma_long"]) / df["sma_long"]

    # Fill NaN values
    df = df.bfill().ffill().fillna(value=0)

    return df


class SignalResult:
    """Result of a pattern recognition signal."""

    def __init__(
        self,
        signal_type: str,
        strength: float,
        direction: float,  # Continuous value from -1.0 (strong sell) to 1.0 (strong buy), 0.0 for neutral
        description: str,
        timestamp: object | None = None,
        confidence: Optional[
            float
        ] = None,  # Confidence in pattern recognition (0.0-1.0), defaults to strength
        metadata: Optional[SignalMetadata] = None,
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

    def __init__(self, config: Optional[Dict[str, object]] = None):
        self.config: Dict[str, object] = dict(config) if config else {}
        self.name = self.__class__.__name__
        self._validate_config()
        # Use LRU cache for signal results to prevent memory leaks and improve performance
        self._signal_cache: LRUCache[tuple[SignalResult, int]] = LRUCache(
            max_size=500, ttl_seconds=300
        )  # 500 entries, 5 minutes TTL

    def _validate_config(self) -> None:
        """Validate configuration parameters with runtime type checking."""
        validator = TypeValidator()

        # Define validation schema for common config parameters
        config_schema = {
            "enabled": bool,
            "min_confidence": Union[
                int, float
            ],  # Will be validated for range separately
            "lookback_period": int,
            "risk_level": str,  # Will be validated for specific values separately
        }

        # Validate types using TypeValidator
        for param_name, expected_type in config_schema.items():
            if param_name in self.config:
                try:
                    validator.validate_type(
                        self.config[param_name], expected_type, f"config.{param_name}"
                    )
                except TypeError as e:
                    raise ValidationError(
                        f"Configuration validation failed for {self.name}: {e}"
                    )

        # Range and value validations
        if "min_confidence" in self.config:
            min_conf = self.config["min_confidence"]
            if not (0.0 <= min_conf <= 1.0):
                raise ValidationError(
                    f"Config 'min_confidence' must be between 0.0 and 1.0 for {self.name}, "
                    f"got {min_conf}"
                )

        if "lookback_period" in self.config:
            lookback = self.config["lookback_period"]
            if lookback <= 0:
                raise ValidationError(
                    f"Config 'lookback_period' must be positive for {self.name}, "
                    f"got {lookback}"
                )

        if "risk_level" in self.config:
            risk_level = self.config["risk_level"]
            valid_risks = ["low", "medium", "high"]
            if risk_level not in valid_risks:
                raise ValidationError(
                    f"Config 'risk_level' must be one of {valid_risks} for {self.name}, "
                    f"got '{risk_level}'"
                )

        # Validate numeric thresholds with range checking
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
                try:
                    validator.validate_type(
                        value, Union[int, float], f"config.{config_key}"
                    )
                    if not (0.0 <= value <= 1.0):
                        raise ValidationError(
                            f"Config '{config_key}' must be between 0.0 and 1.0 for {self.name}, "
                            f"got {value}"
                        )
                except TypeError as e:
                    raise ValidationError(
                        f"Configuration validation failed for {self.name}: {e}"
                    )

    def get_config_value(self, key: str, default: object = None) -> object:
        """Get configuration value with optional default."""
        return safe_config_get(self.config, key, default)

    def is_enabled(self) -> bool:
        """Check if this recognizer is enabled."""
        return safe_config_get_bool(self.config, "enabled", True)

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return safe_config_get_float(self.config, "min_confidence", 0.0)

    def get_risk_level(self) -> str:
        """Get risk level for this recognizer."""
        return safe_config_get_str(self.config, "risk_level", "medium")

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
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
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
            raise ValidationError(
                f"Invalid input data for pattern {self.name}: {e}",
                details={"recognizer": self.name, "error": str(e)},
            ) from e

        # Adjust negative index
        if index < 0:
            index = len(data) + index

        # Check minimum required length for this specific pattern
        if len(data) < required_length:
            raise ValidationError(
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

    def _validate_multi_timeframe_data(self, multi_timeframe_data: MultiTimeframeData) -> None:
        """Validate multi-timeframe data structure."""
        if not isinstance(multi_timeframe_data, dict):
            raise ValueError(
                f"Multi-timeframe data must be a dictionary for {self.name}"
            )

        # Basic schema validation - can be extended by subclasses.
        for tf_key, tf_payload in multi_timeframe_data.items():
            if not isinstance(tf_key, str):
                raise ValueError(
                    f"Multi-timeframe key must be str for {self.name}, got {type(tf_key).__name__}"
                )
            if not isinstance(tf_payload, dict):
                raise ValueError(
                    f"Multi-timeframe payload must be dict for {self.name}, got {type(tf_payload).__name__}"
                )

    def preprocess_data(
        self,
        data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        add_technical_features: bool = True,
        normalize_features: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess data for pattern recognition.

        Args:
            data: Raw OHLCV data
            feature_columns: Specific columns to preprocess (default: OHLCV)
            add_technical_features: Whether to add technical indicators
            normalize_features: Whether to normalize features

        Returns:
            Preprocessed data
        """
        processed_data = data.copy()

        # Add technical features if requested
        if add_technical_features:
            processed_data = calculate_technical_features(
                processed_data, window=getattr(self, "lookback_period", 20)
            )

        # Normalize features if requested
        if normalize_features:
            if feature_columns is None:
                # Default OHLCV columns plus common technical features
                feature_columns = ["open", "high", "low", "close", "volume"]
                if add_technical_features:
                    feature_columns.extend(
                        ["returns", "volatility", "momentum", "trend_strength"]
                    )

            # Only preprocess columns that exist
            existing_features = [
                col for col in feature_columns if col in processed_data.columns
            ]
            if existing_features:
                processed_data = preprocess_features(
                    processed_data,
                    existing_features,
                    method="robust",  # Use robust scaling for outlier resistance
                    remove_outliers=True,
                )

        return processed_data

    def _analyze_multi_timeframe_alignment(
        self,
        data: pd.DataFrame,
        index: int,
        multi_timeframe_data: MultiTimeframeData,
        pattern_type: str = "general",
    ) -> float:
        """
        Analyze pattern alignment across multiple timeframes.

        Base implementation provides general alignment analysis.
        Subclasses should override for pattern-specific logic.

        Args:
            data: Current timeframe data
            index: Current index
            multi_timeframe_data: Multi-timeframe data dictionary
            pattern_type: Type of pattern for specific analysis

        Returns:
            Confidence multiplier based on timeframe alignment (0.5-1.5)
        """
        try:
            alignment_score = 1.0
            aligned_timeframes = 0

            # Check alignment with higher timeframes
            for tf, tf_data in multi_timeframe_data.items():
                if isinstance(tf_data, dict) and "data" in tf_data:
                    tf_df = tf_data["data"]
                    if len(tf_df) > 10:  # Minimum data requirement
                        try:
                            # Basic alignment check using price direction
                            current_close = data.iloc[index]["close"]
                            prev_close = data.iloc[index - 1]["close"]
                            current_trend = 1 if current_close > prev_close else -1
                            tf_current_close = tf_df.iloc[-1]["close"]
                            tf_prev_close = tf_df.iloc[-2]["close"]
                            tf_trend = 1 if tf_current_close > tf_prev_close else -1

                            if current_trend == tf_trend:
                                alignment_score += 0.1
                                aligned_timeframes += 1
                            else:
                                alignment_score -= 0.05

                        except Exception:
                            continue

            # Normalize alignment score
            if aligned_timeframes > 0:
                alignment_score = max(0.5, min(1.5, alignment_score))

            return alignment_score

        except Exception:
            return 1.0  # Default neutral multiplier

    def _adjust_thresholds_for_regime(
        self,
        multi_timeframe_data: Optional[MultiTimeframeData],
        pattern_type: str = "general",
    ) -> RegimeAdjustment:
        """
        Adjust pattern thresholds based on market regime.

        Base implementation provides general regime analysis.
        Subclasses should override for pattern-specific adjustments.

        Args:
            multi_timeframe_data: Multi-timeframe data dictionary
            pattern_type: Type of pattern for specific analysis

        Returns:
            Dictionary with adjusted parameters
        """
        try:
            # Default parameters (subclasses should define their own)
            adjusted_params = {}

            # Analyze market regime from multi-timeframe data
            if multi_timeframe_data:
                volatility_indicators = []
                trend_indicators = []

                for tf, tf_data in multi_timeframe_data.items():
                    if isinstance(tf_data, dict) and "data" in tf_data:
                        tf_df = tf_data["data"]
                        if len(tf_df) > 20:
                            try:
                                # Calculate volatility (ATR proxy)
                                high_low = tf_df["high"] - tf_df["low"]
                                high_close = (
                                    tf_df["high"] - tf_df["close"].shift(1)
                                ).abs()
                                low_close = (
                                    tf_df["low"] - tf_df["close"].shift(1)
                                ).abs()
                                tr = pd.concat(
                                    [high_low, high_close, low_close], axis=1
                                ).max(axis=1)
                                atr = tr.rolling(14).mean()

                                if len(atr) > 0:
                                    current_atr = atr.iloc[-1]
                                    avg_price = tf_df["close"].iloc[-1]
                                    volatility = (
                                        current_atr / avg_price if avg_price > 0 else 0
                                    )
                                    volatility_indicators.append(volatility)

                                # Calculate trend strength
                                recent_prices = tf_df["close"].tail(20).values
                                if len(recent_prices) >= 10:
                                    x = np.arange(len(recent_prices))
                                    slope, _ = np.polyfit(x, recent_prices, 1)
                                    trend_strength = abs(slope) / np.mean(recent_prices)
                                    trend_indicators.append(trend_strength)

                            except Exception:
                                continue

                # Calculate average regime indicators
                avg_volatility = (
                    sum(volatility_indicators) / len(volatility_indicators)
                    if volatility_indicators
                    else 0.01
                )
                avg_trend = (
                    sum(trend_indicators) / len(trend_indicators)
                    if trend_indicators
                    else 0.005
                )

                # Store regime information for subclasses to use
                adjusted_params.update(
                    {
                        "avg_volatility": avg_volatility,
                        "avg_trend_strength": avg_trend,
                        "regime": "high_volatility"
                        if avg_volatility > 0.02
                        else "low_volatility",
                        "trend_regime": "trending" if avg_trend > 0.01 else "sideways",
                    }
                )

            return adjusted_params

        except Exception:
            return {}

    @abstractmethod
    def recognize(
        self,
        data: pd.DataFrame,
        index: int = -1,
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
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
        multi_timeframe_data: Optional[MultiTimeframeData] = None,
    ) -> Optional[SignalResult]:
        """
        Recognize pattern with caching to avoid redundant calculations.

        Args:
            data: OHLCV data as pandas DataFrame
            index: Index to check for pattern

        Returns:
            Cached SignalResult if available and valid, otherwise new recognition
        """
        resolved_index = index if index >= 0 else len(data) + index
        if resolved_index < 0 or resolved_index >= len(data):
            return None

        cache_key = (
            f"{self.name}_{resolved_index}_{float(data.iloc[resolved_index]['close']):.8f}"
        )

        # Check cache first
        cached_entry = self._signal_cache.get(cache_key)
        if cached_entry is not None:
            cached_signal, signal_index = cached_entry
            if not cached_signal.is_expired(resolved_index, signal_index):
                return cached_signal

        # Preprocess data for better pattern recognition
        processed_data = self.preprocess_data(data)

        # Calculate new signal
        signal: Optional[SignalResult] = self.recognize(
            processed_data, resolved_index, multi_timeframe_data
        )
        if signal is not None:
            self._signal_cache.set(cache_key, (signal, resolved_index))
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

    @timed
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

        # Convert to numpy array for type safety and memory efficiency
        prices_array = np.array(prices, dtype=np.float64)
        # Calculate linear trend using least squares
        x = np.arange(len(prices_array))
        slope, _ = np.polyfit(x, prices_array, 1)

        # Calculate R-squared to measure trend strength
        y_mean = np.mean(prices_array)
        ss_tot = np.sum((prices_array - y_mean) ** 2)
        ss_res = np.sum((prices_array - (slope * x + prices_array[0])) ** 2)

        if ss_tot == 0:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)

        # Convert slope to strength (absolute value, normalized)
        avg_price = np.mean(prices_array)
        slope_strength = min(
            1.0, abs(slope) / (avg_price * 0.01)
        )  # 1% of average price as strong slope

        # Combine R-squared and slope strength
        trend_strength = r_squared * 0.7 + slope_strength * 0.3

        return cast(float, min(1.0, max(0.0, trend_strength)))

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
            recent_prices = cast(
                np.ndarray, data.iloc[index - lookback + 1 : index + 1]["close"].values
            )
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
            normalized_movement, expected_movement
        )
        confidence = 1.0 - abs(normalized_movement - expected_movement) / max(
            normalized_movement, expected_movement
        )

        return cast(float, min(1.0, max(0.0, confidence)))

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

    # Standardized constants for candlestick pattern confidence calculation
    PATTERN_CONFIDENCE_WEIGHTS = {
        "trend_strength": 0.25,
        "candle_size": 0.2,
        "price_movement": 0.2,
        "pattern_completeness": 0.2,
        "volume": 0.15,
    }

    BASE_CONFIDENCE_LEVELS = {
        "sakata_five_methods": 0.7,
        "morning_star": 0.7,
        "evening_star": 0.7,
        "three_white_soldiers": 0.8,
        "three_black_crows": 0.8,
        "hammer": 0.6,
        "hanging_man": 0.6,
        "shooting_star": 0.6,
        "engulfing": 0.7,
        "bullish_engulfing": 0.7,
        "bearish_engulfing": 0.7,
        "harami": 0.6,
        "piercing": 0.7,
        "dark_cloud_cover": 0.7,
        "rising_three_methods": 0.75,
        "doji": 0.5,
    }

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
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

    class MultiCandleCharacteristics(TypedDict):
        body_sizes: list[float]
        body_ratios: list[float]
        upper_shadow_ratios: list[float]
        lower_shadow_ratios: list[float]
        is_bullish: list[bool]
        is_bearish: list[bool]
        avg_body_size: float

    def analyze_multiple_candle_characteristics(
        self, data: pd.DataFrame, indices: List[int]
    ) -> MultiCandleCharacteristics:
        """
        Analyze characteristics for multiple candles.

        Args:
            data: OHLCV data
            indices: List of candle indices to analyze

        Returns:
            Dictionary with lists of candle characteristics
        """
        characteristics: CandlestickPatternRecognizer.MultiCandleCharacteristics = {
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

    def _calculate_pattern_confidence(
        self,
        data: pd.DataFrame,
        index: int,
        pattern_factors: Dict[str, float],
        base_confidence: float = 0.5,
    ) -> float:
        """
        Calculate dynamic confidence score for candlestick patterns using standardized weights.

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

        # Use standardized weights for candlestick patterns
        weights = self.PATTERN_CONFIDENCE_WEIGHTS

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

    def get_base_confidence_for_pattern(self, pattern_type: str) -> float:
        """
        Get standardized base confidence level for a specific pattern type.

        Args:
            pattern_type: The pattern type identifier

        Returns:
            Base confidence level for the pattern
        """
        return self.BASE_CONFIDENCE_LEVELS.get(pattern_type, 0.5)

    def _calculate_volume_confidence(
        self, data: pd.DataFrame, index: int, expected_volume_change: float = 1.0
    ) -> float:
        """
        Calculate volume confidence for pattern validation.

        Args:
            data: OHLCV data
            index: Current index
            expected_volume_change: Expected volume change ratio (1.0 = no change)

        Returns:
            Volume confidence score (0.0-1.0)
        """
        if not self.validate_data(data) or index < 5 or index >= len(data):
            return 0.5

        # Check if volume column exists
        if "volume" not in data.columns:
            return 0.5

        try:
            current_volume = float(data.iloc[index]["volume"])
            if current_volume <= 0:
                return 0.5

            # Calculate average volume over recent period
            recent_volumes = data.iloc[index - 5 : index]["volume"].values
            volumes_array = np.asarray(recent_volumes, dtype=np.float64)
            avg_volume = float(np.mean(volumes_array))

            if avg_volume <= 0:
                return 0.5

            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume

            # Calculate confidence based on expected volume change
            if expected_volume_change >= 1.0:
                # Expecting volume increase
                if volume_ratio >= expected_volume_change:
                    confidence = min(1.0, volume_ratio / expected_volume_change)
                else:
                    confidence = max(0.3, volume_ratio / expected_volume_change)
            else:
                # Expecting volume decrease
                if volume_ratio <= expected_volume_change:
                    confidence = min(1.0, expected_volume_change / volume_ratio)
                else:
                    confidence = max(0.3, expected_volume_change / volume_ratio)

            return min(1.0, max(0.0, confidence))

        except Exception:
            return 0.5


class MultiCandlePatternRecognizer(PatternRecognizer):
    """
    Base class for multi-candle pattern recognizers.
    """

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
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
