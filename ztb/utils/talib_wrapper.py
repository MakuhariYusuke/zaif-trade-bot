"""
Ta-Lib integration utilities for technical analysis.

This module provides a unified interface for technical analysis indicators,
supporting both Ta-Lib and custom implementations with fallback logic.
"""

import logging
import warnings
from collections import OrderedDict
from typing import Any, Dict, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.metrics.statistics import calculate_volatility
from ztb.utils.types import IndicatorInfo

# Try to import Ta-Lib, fallback to custom implementations if not available
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Try to import ta library as additional fallback
try:
    import ta

    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

if not TALIB_AVAILABLE:
    if TA_AVAILABLE:
        warnings.warn(
            "Ta-Lib not available. Using ta library for technical analysis. "
            "Install Ta-Lib for better performance: pip install TA-Lib"
        )
    else:
        warnings.warn(
            "Ta-Lib not available. Using custom implementations. "
            "Install Ta-Lib for better performance: pip install TA-Lib"
        )

logger = logging.getLogger(__name__)

from ztb.utils.performance_utils import timed


class TaLibError(Exception):
    """Custom exception for Ta-Lib related errors."""

    pass


class TaLibWrapper:
    """
    Wrapper class for Ta-Lib functions with fallback to custom implementations.

    This class provides a unified interface for technical analysis indicators,
    automatically falling back to custom implementations when Ta-Lib is not available.
    Enhanced with input validation, error handling, caching, and performance optimizations.
    """

    # Default periods for technical indicators
    DEFAULT_PERIODS: Dict[str, Union[int, float]] = {
        "SMA": 30,
        "EMA": 30,
        "RSI": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "BBANDS": 20,
        "ATR": 14,
        "CCI": 20,
        "STOCH_FAST_K": 14,
        "STOCH_SLOW_K": 3,
        "STOCH_SLOW_D": 3,
        "WILLIAMS_R": 14,
        "MFI": 14,
        "ROC": 10,
        "OBV": 0,  # OBV doesn't use period, but for consistency
        "SAR_ACCELERATION": 0.02,
        "SAR_MAXIMUM": 0.2,
    }

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 128,
        strict_validation: bool = True,
    ):
        """
        Initialize TaLibWrapper.

        Args:
            enable_cache: Whether to enable result caching
            cache_size: Maximum cache size for LRU cache
            strict_validation: Whether to perform strict input validation
        """
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.strict_validation = strict_validation
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def check_talib_availability() -> bool:
        """Check if Ta-Lib is available."""
        return TALIB_AVAILABLE

    @staticmethod
    def _validate_input_data(
        data: Union[NDArray[np.float64], pd.Series],
        name: str,
        strict_validation: bool = True,
    ) -> NDArray[np.float64]:
        """Validate and convert input data to numpy array."""
        if strict_validation:
            if data is None:
                raise TaLibError(f"{name} cannot be None")

            try:
                arr = np.asarray(data, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TaLibError(f"Cannot convert {name} to numeric array: {e}")

            if arr.ndim != 1:
                raise TaLibError(f"{name} must be 1-dimensional")

            if len(arr) == 0:
                raise TaLibError(f"{name} cannot be empty")

            if not np.isfinite(arr).all():
                if strict_validation:
                    raise TaLibError(f"{name} contains NaN or infinite values")
                else:
                    logger.warning(
                        f"{name} contains NaN or infinite values, proceeding anyway"
                    )
        else:
            # Minimal validation for performance
            arr = np.asarray(data, dtype=np.float64)

        return arr

    @staticmethod
    def _validate_period(period: int, name: str) -> int:
        """Validate period parameter."""
        if not isinstance(period, int) or period <= 0:
            raise TaLibError(f"{name} must be a positive integer, got {period}")
        return period

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key for function calls."""
        # Create a hash of the arguments for caching
        import hashlib

        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cache_get(self, key: str) -> Any:
        """Get item from cache with LRU update."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        """Put item in cache with LRU eviction."""
        if key in self._cache:
            # Update existing item
            self._cache.move_to_end(key)
        else:
            # Add new item
            if len(self._cache) >= self.cache_size:
                # Remove least recently used
                self._cache.popitem(last=False)
        self._cache[key] = value

    @timed
    def sma(
        self, data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """
        Simple Moving Average.

        Args:
            data: Input price data
            period: Period for SMA calculation

        Returns:
            SMA values

        Raises:
            TaLibError: If input validation fails
        """
        data = self._validate_input_data(data, "data")
        period = self._validate_period(period, "period")

        if len(data) < period:
            raise TaLibError(f"Data length {len(data)} is less than period {period}")

        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key("sma", data.tobytes(), period)
            cached_result = self._cache_get(cache_key)
            if cached_result is not None:
                return cached_result.copy()

        try:
            if TALIB_AVAILABLE:
                result = talib.SMA(data, timeperiod=period)
                # Handle NaN values at the beginning
                result = cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            elif TA_AVAILABLE:
                # Use ta library
                sma_indicator = ta.trend.SMAIndicator(pd.Series(data), window=period)
                result = sma_indicator.sma_indicator().values
                result = cast(NDArray[np.float64], result)
            else:
                result = self._sma_custom(data, period)

            # Cache result
            if self.enable_cache:
                self._cache_put(cache_key, result.copy())

            return result

        except Exception as e:
            raise TaLibError(f"SMA calculation failed: {e}")

    @timed
    @staticmethod
    def ema(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """
        Exponential Moving Average.

        Args:
            data: Input price data
            period: Period for EMA calculation

        Returns:
            EMA values

        Raises:
            TaLibError: If input validation fails
        """
        # Use static validation helpers to keep interface callable as class method
        data = TaLibWrapper._validate_input_data(data, "data")
        period = TaLibWrapper._validate_period(period, "period")

        if len(data) < period:
            raise TaLibError(f"Data length {len(data)} is less than period {period}")

        try:
            if TALIB_AVAILABLE:
                result = talib.EMA(data, timeperiod=period)
                result = cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            elif TA_AVAILABLE:
                # Use ta library
                ema_indicator = ta.trend.EMAIndicator(pd.Series(data), window=period)
                result = ema_indicator.ema_indicator().values
                result = cast(NDArray[np.float64], result)
            else:
                result = self._ema_custom(data, period)

            # No caching here when called as staticmethod (keep implementation simple)

            return result

        except Exception as e:
            raise TaLibError(f"EMA calculation failed: {e}")

    @timed
    def rsi(
        self, data: Union[NDArray[np.float64], pd.Series], period: int = 14
    ) -> NDArray[np.float64]:
        """
        Relative Strength Index.

        Args:
            data: Input price data
            period: Period for RSI calculation

        Returns:
            RSI values

        Raises:
            TaLibError: If input validation fails
        """
        data = self._validate_input_data(data, "data")
        period = self._validate_period(period, "period")

        if len(data) < period + 1:  # RSI needs at least period + 1 data points
            raise TaLibError(
                f"Data length {len(data)} is insufficient for RSI with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.RSI(data, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._rsi_custom(data, period)
        except Exception as e:
            raise TaLibError(f"RSI calculation failed: {e}")

    @timed
    def bbands(
        self,
        data: Union[NDArray[np.float64], pd.Series],
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Bollinger Bands.

        Bollinger Bands are volatility bands placed above and below a moving average.
        Volatility is based on the standard deviation, which changes as volatility increases or decreases.
        The bands automatically widen when volatility increases and narrow when volatility decreases.

        Args:
            data: Input price data (typically closing prices)
            period: Period for moving average calculation (default: 20)
            nbdevup: Standard deviation multiplier for upper band (default: 2.0)
            nbdevdn: Standard deviation multiplier for lower band (default: 2.0)

        Returns:
            Tuple of (Upper Band, Middle Band (SMA), Lower Band)

        Raises:
            TaLibError: If input validation fails
        """
        data = self._validate_input_data(data, "data")
        period = self._validate_period(period, "period")

        if len(data) < period:
            raise TaLibError(f"Data length {len(data)} is less than period {period}")

        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key(
                "bbands", data.tobytes(), period, nbdevup, nbdevdn
            )
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                return (
                    cached_result[0].copy(),
                    cached_result[1].copy(),
                    cached_result[2].copy(),
                )

        try:
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(
                    data, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0
                )
                result = (
                    np.nan_to_num(upper, nan=np.nan),
                    np.nan_to_num(middle, nan=np.nan),
                    np.nan_to_num(lower, nan=np.nan),
                )
            elif TA_AVAILABLE:
                # Use ta library
                bb_indicator = ta.volatility.BollingerBands(
                    pd.Series(data), window=period, window_dev=nbdevup
                )
                upper = bb_indicator.bollinger_hband().values
                lower = bb_indicator.bollinger_lband().values
                middle = bb_indicator.bollinger_mavg().values
                result = (
                    np.nan_to_num(upper, nan=np.nan),
                    np.nan_to_num(middle, nan=np.nan),
                    np.nan_to_num(lower, nan=np.nan),
                )
            else:
                result = self._bbands_custom(data, period, nbdevup, nbdevdn)

            # Cache result
            if self.enable_cache:
                self._cache[cache_key] = (
                    result[0].copy(),
                    result[1].copy(),
                    result[2].copy(),
                )
                if len(self._cache) > self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            return result

        except Exception as e:
            raise TaLibError(f"BBANDS calculation failed: {e}")

    def macd(
        self,
        data: Union[NDArray[np.float64], pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        MACD (Moving Average Convergence Divergence).

        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price. It calculates the
        difference between a fast EMA and a slow EMA, then plots a signal line
        (EMA of the MACD line) and a histogram (MACD - Signal).

        Args:
            data: Input price data (typically closing prices)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)

        Raises:
            TaLibError: If input validation fails
        """
        data = self._validate_input_data(data, "data")
        fast_period = self._validate_period(fast_period, "fast_period")
        slow_period = self._validate_period(slow_period, "slow_period")
        signal_period = self._validate_period(signal_period, "signal_period")

        if fast_period >= slow_period:
            raise TaLibError(
                f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
            )

        min_length = max(fast_period, slow_period) + signal_period
        if len(data) < min_length:
            raise TaLibError(
                f"Data length {len(data)} is insufficient for MACD calculation"
            )

        cache_key = None
        if self.enable_cache:
            cache_key = self._get_cache_key(
                "macd", data.tobytes(), fast_period, slow_period, signal_period
            )
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                return (
                    cached_result[0].copy(),
                    cached_result[1].copy(),
                    cached_result[2].copy(),
                )

        try:
            if TALIB_AVAILABLE:
                macd, signal, hist = talib.MACD(
                    data,
                    fastperiod=fast_period,
                    slowperiod=slow_period,
                    signalperiod=signal_period,
                )
                result = (
                    np.nan_to_num(macd, nan=np.nan),
                    np.nan_to_num(signal, nan=np.nan),
                    np.nan_to_num(hist, nan=np.nan),
                )
            else:
                result = self._macd_custom(
                    data, fast_period, slow_period, signal_period
                )

            # Cache result
            if self.enable_cache and cache_key:
                self._cache[cache_key] = (
                    result[0].copy(),
                    result[1].copy(),
                    result[2].copy(),
                )
                if len(self._cache) > self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            return result

        except Exception as e:
            raise TaLibError(f"MACD calculation failed: {e}")

    def stoch(
        self,
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            fastk_period: Fast %K period
            slowk_period: Slow %K period
            slowd_period: Slow %D period

        Returns:
            Tuple of (Slow %K, Slow %D)

        Raises:
            TaLibError: If input validation fails
        """
        high = self._validate_input_data(high, "high")
        low = self._validate_input_data(low, "low")
        close = self._validate_input_data(close, "close")
        fastk_period = self._validate_period(fastk_period, "fastk_period")
        slowk_period = self._validate_period(slowk_period, "slowk_period")
        slowd_period = self._validate_period(slowd_period, "slowd_period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        min_length = fastk_period + max(slowk_period, slowd_period)
        if len(high) < min_length:
            raise TaLibError(
                f"Data length {len(high)} is insufficient for Stochastic calculation"
            )

        cache_key = None
        if self.enable_cache:
            cache_key = self._get_cache_key(
                "stoch",
                high.tobytes(),
                low.tobytes(),
                close.tobytes(),
                fastk_period,
                slowk_period,
                slowd_period,
            )
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                return (cached_result[0].copy(), cached_result[1].copy())

        try:
            if TALIB_AVAILABLE:
                slowk, slowd = talib.STOCH(
                    high,
                    low,
                    close,
                    fastk_period=fastk_period,
                    slowk_period=slowk_period,
                    slowd_period=slowd_period,
                )
                result = (
                    np.nan_to_num(slowk, nan=np.nan),
                    np.nan_to_num(slowd, nan=np.nan),
                )
            else:
                result = self._stoch_custom(
                    high, low, close, fastk_period, slowk_period, slowd_period
                )

            # Cache result
            if self.enable_cache and cache_key:
                self._cache[cache_key] = (result[0].copy(), result[1].copy())
                if len(self._cache) > self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            return result

        except Exception as e:
            raise TaLibError(f"Stochastic calculation failed: {e}")

    def stochf(
        self,
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        fastk_period: int = 14,
        fastd_period: int = 3,
        fastd_matype: int = 0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Stochastic Fast.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            fastk_period: Fast %K period
            fastd_period: Fast %D period
            fastd_matype: Fast %D Moving Average Type (0=SMA)

        Returns:
            Tuple of (Fast %K, Fast %D)

        Raises:
            TaLibError: If input validation fails
        """
        high = self._validate_input_data(high, "high")
        low = self._validate_input_data(low, "low")
        close = self._validate_input_data(close, "close")
        fastk_period = self._validate_period(fastk_period, "fastk_period")
        fastd_period = self._validate_period(fastd_period, "fastd_period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        min_length = fastk_period + fastd_period
        if len(high) < min_length:
            raise TaLibError(
                f"Data length {len(high)} is insufficient for Stochastic Fast calculation"
            )

        cache_key = None
        if self.enable_cache:
            cache_key = self._get_cache_key(
                "stochf",
                high.tobytes(),
                low.tobytes(),
                close.tobytes(),
                fastk_period,
                fastd_period,
                fastd_matype,
            )
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                return (cached_result[0].copy(), cached_result[1].copy())

        try:
            if TALIB_AVAILABLE:
                fastk, fastd = talib.STOCHF(
                    high,
                    low,
                    close,
                    fastk_period=fastk_period,
                    fastd_period=fastd_period,
                    fastd_matype=fastd_matype,
                )
                result = (
                    np.nan_to_num(fastk, nan=np.nan),
                    np.nan_to_num(fastd, nan=np.nan),
                )
            else:
                result = self._stochf_custom(
                    high, low, close, fastk_period, fastd_period
                )

            # Cache result
            if self.enable_cache and cache_key:
                self._cache[cache_key] = (result[0].copy(), result[1].copy())
                if len(self._cache) > self.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            return result

        except Exception as e:
            raise TaLibError(f"Stochastic Fast calculation failed: {e}")

    @staticmethod
    def adx(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Average Directional Index (ADX).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for ADX calculation

        Returns:
            ADX values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        if len(high) < period * 2:  # ADX needs sufficient data
            raise TaLibError(
                f"Data length {len(high)} is insufficient for ADX with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.ADX(high, low, close, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._adx_custom(high, low, close, period)
        except Exception as e:
            raise TaLibError(f"ADX calculation failed: {e}")

    @staticmethod
    def plus_di(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Plus Directional Indicator (+DI).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for +DI calculation

        Returns:
            +DI values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        if len(high) < period * 2:
            raise TaLibError(
                f"Data length {len(high)} is insufficient for +DI with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.PLUS_DI(high, low, close, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._plus_di_custom(high, low, close, period)
        except Exception as e:
            raise TaLibError(f"+DI calculation failed: {e}")

    @staticmethod
    def minus_di(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Minus Directional Indicator (-DI).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for -DI calculation

        Returns:
            -DI values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        if len(high) < period * 2:
            raise TaLibError(
                f"Data length {len(high)} is insufficient for -DI with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.MINUS_DI(high, low, close, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._minus_di_custom(high, low, close, period)
        except Exception as e:
            raise TaLibError(f"-DI calculation failed: {e}")

    @staticmethod
    def _plus_di_custom(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Custom +DI implementation as fallback.
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])

        # Calculate Directional Movement
        dm_plus = np.where(
            (high - np.roll(high, 1)) > (np.roll(low, 1) - low),
            np.maximum(high - np.roll(high, 1), 0),
            0,
        )

        # Smooth with EMA
        tr_smooth = TaLibWrapper._ema_custom(tr, period)
        dm_plus_smooth = TaLibWrapper._ema_custom(dm_plus, period)

        # Calculate +DI
        di_plus = 100 * dm_plus_smooth / tr_smooth

        return np.nan_to_num(di_plus, nan=np.nan)

    @staticmethod
    def _minus_di_custom(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Custom -DI implementation as fallback.
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])

        # Calculate Directional Movement
        dm_minus = np.where(
            (np.roll(low, 1) - low) > (high - np.roll(high, 1)),
            np.maximum(np.roll(low, 1) - low, 0),
            0,
        )

        # Smooth with EMA
        tr_smooth = TaLibWrapper._ema_custom(tr, period)
        dm_minus_smooth = TaLibWrapper._ema_custom(dm_minus, period)

        # Calculate -DI
        di_minus = 100 * dm_minus_smooth / tr_smooth

        return np.nan_to_num(di_minus, nan=np.nan)

    @staticmethod
    def williams_r(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Williams %R Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for Williams %R calculation

        Returns:
            Williams %R values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        if len(high) < period:
            raise TaLibError(f"Data length {len(high)} is less than period {period}")

        try:
            if TALIB_AVAILABLE:
                result = talib.WILLR(high, low, close, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._williams_r_custom(high, low, close, period)
        except Exception as e:
            raise TaLibError(f"Williams %R calculation failed: {e}")

    @staticmethod
    def cci(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 20,
    ) -> NDArray[np.float64]:
        """
        Commodity Channel Index (CCI).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for CCI calculation

        Returns:
            CCI values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        if len(high) < period:
            raise TaLibError(f"Data length {len(high)} is less than period {period}")

        try:
            if TALIB_AVAILABLE:
                result = talib.CCI(high, low, close, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._cci_custom(high, low, close, period)
        except Exception as e:
            raise TaLibError(f"CCI calculation failed: {e}")

    @staticmethod
    def tema(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """
        Triple Exponential Moving Average (TEMA).

        Args:
            data: Input price data
            period: Period for TEMA calculation

        Returns:
            TEMA values

        Raises:
            TaLibError: If input validation fails
        """
        data = TaLibWrapper._validate_input_data(data, "data")
        period = TaLibWrapper._validate_period(period, "period")

        if len(data) < period * 3:  # TEMA needs 3 EMAs
            raise TaLibError(
                f"Data length {len(data)} is insufficient for TEMA with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.TEMA(data, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._tema_custom(data, period)
        except Exception as e:
            raise TaLibError(f"TEMA calculation failed: {e}")

    @staticmethod
    def kama(
        data: Union[NDArray[np.float64], pd.Series], period: int = 30
    ) -> NDArray[np.float64]:
        """
        Kaufman Adaptive Moving Average (KAMA).

        Args:
            data: Input price data
            period: Period for KAMA calculation

        Returns:
            KAMA values

        Raises:
            TaLibError: If input validation fails
        """
        data = TaLibWrapper._validate_input_data(data, "data")
        period = TaLibWrapper._validate_period(period, "period")

        if len(data) < period:
            raise TaLibError(f"Data length {len(data)} is less than period {period}")

        try:
            if TALIB_AVAILABLE:
                result = talib.KAMA(data, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._kama_custom(data, period)
        except Exception as e:
            raise TaLibError(f"KAMA calculation failed: {e}")

    @staticmethod
    def roc(
        data: Union[NDArray[np.float64], pd.Series], period: int = 10
    ) -> NDArray[np.float64]:
        """
        Rate of Change (ROC).

        Args:
            data: Input price data
            period: Period for ROC calculation

        Returns:
            ROC values

        Raises:
            TaLibError: If input validation fails
        """
        data = TaLibWrapper._validate_input_data(data, "data")
        period = TaLibWrapper._validate_period(period, "period")

        if len(data) <= period:
            raise TaLibError(
                f"Data length {len(data)} must be greater than period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.ROC(data, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._roc_custom(data, period)
        except Exception as e:
            raise TaLibError(f"ROC calculation failed: {e}")

    @staticmethod
    def wma(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """
        Weighted Moving Average (WMA).

        Args:
            data: Input price data
            period: Period for WMA calculation

        Returns:
            WMA values

        Raises:
            TaLibError: If input validation fails
        """
        data = TaLibWrapper._validate_input_data(data, "data")
        period = TaLibWrapper._validate_period(period, "period")

        if len(data) < period:
            raise TaLibError(f"Data length {len(data)} is less than period {period}")

        try:
            if TALIB_AVAILABLE:
                result = talib.WMA(data, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._wma_custom(data, period)
        except Exception as e:
            raise TaLibError(f"WMA calculation failed: {e}")

    @staticmethod
    def mfi(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        volume: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Money Flow Index (MFI).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: Period for MFI calculation

        Returns:
            MFI values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        volume = TaLibWrapper._validate_input_data(volume, "volume")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
            raise TaLibError(
                "High, low, close, and volume arrays must have the same length"
            )

        if len(high) < period * 2:
            raise TaLibError(
                f"Data length {len(high)} is insufficient for MFI with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.MFI(high, low, close, volume, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                raise TaLibError("MFI calculation requires Ta-Lib library")
        except Exception as e:
            raise TaLibError(f"MFI calculation failed: {e}")

    @staticmethod
    def obv(
        close: Union[NDArray[np.float64], pd.Series],
        volume: Union[NDArray[np.float64], pd.Series],
    ) -> NDArray[np.float64]:
        """
        On Balance Volume (OBV).

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            OBV values

        Raises:
            TaLibError: If input validation fails
        """
        close = TaLibWrapper._validate_input_data(close, "close")
        volume = TaLibWrapper._validate_input_data(volume, "volume")

        if len(close) != len(volume):
            raise TaLibError("Close and volume arrays must have the same length")

        try:
            if TALIB_AVAILABLE:
                result = talib.OBV(close, volume)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._obv_custom(close, volume)
        except Exception as e:
            raise TaLibError(f"OBV calculation failed: {e}")

    @staticmethod
    def sar(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> NDArray[np.float64]:
        """
        Parabolic SAR.

        Args:
            high: High prices
            low: Low prices
            acceleration: Acceleration factor
            maximum: Maximum acceleration

        Returns:
            SAR values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")

        if len(high) != len(low):
            raise TaLibError("High and low arrays must have the same length")

        try:
            if TALIB_AVAILABLE:
                result = talib.SAR(
                    high, low, acceleration=acceleration, maximum=maximum
                )
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._sar_custom(high, low, acceleration, maximum)
        except Exception as e:
            raise TaLibError(f"SAR calculation failed: {e}")

    @staticmethod
    def atr(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Average True Range (ATR).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for ATR calculation

        Returns:
            ATR values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        period = TaLibWrapper._validate_period(period, "period")

        if len(high) != len(low) or len(high) != len(close):
            raise TaLibError("High, low, and close arrays must have the same length")

        if len(high) < period + 1:
            raise TaLibError(
                f"Data length {len(high)} is insufficient for ATR with period {period}"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.ATR(high, low, close, timeperiod=period)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                return TaLibWrapper._atr_custom(high, low, close, period)
        except Exception as e:
            raise TaLibError(f"ATR calculation failed: {e}")

    @staticmethod
    def bbands(
        data: Union[NDArray[np.float64], pd.Series],
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Bollinger Bands.

        Args:
            data: Input price data
            period: Period for moving average
            nbdevup: Standard deviation multiplier for upper band
            nbdevdn: Standard deviation multiplier for lower band

        Returns:
            Tuple of (upper_band, middle_band, lower_band)

        Raises:
            TaLibError: If input validation fails
        """
        data = TaLibWrapper._validate_input_data(data, "data")
        period = TaLibWrapper._validate_period(period, "period")

        if len(data) < period:
            raise TaLibError(f"Data length {len(data)} is less than period {period}")

        try:
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(
                    data, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn
                )
                return (
                    np.nan_to_num(upper, nan=np.nan),
                    np.nan_to_num(middle, nan=np.nan),
                    np.nan_to_num(lower, nan=np.nan),
                )
            else:
                return TaLibWrapper._bbands_custom(data, period, nbdevup, nbdevdn)
        except Exception as e:
            raise TaLibError(f"BBANDS calculation failed: {e}")

    @staticmethod
    def _mfi_custom(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
        volume: NDArray[np.float64],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """
        Custom MFI implementation as fallback.
        """
        # Typical Price
        typical_price = ((high + low + close) / 3).astype(np.float64)

        # Raw Money Flow
        money_flow = (typical_price * volume).astype(np.float64)

        # Positive and Negative Money Flow
        price_diff = np.diff(close, prepend=close[0])
        positive_flow = np.where(price_diff > 0, money_flow, 0.0).astype(np.float64)
        negative_flow = np.where(price_diff < 0, money_flow, 0.0).astype(np.float64)

        # Money Flow Ratio
        pos_mf_sum: NDArray[np.float64] = TaLibWrapper._rolling_sum_custom(
            positive_flow, period
        )
        neg_mf_sum: NDArray[np.float64] = TaLibWrapper._rolling_sum_custom(
            negative_flow, period
        )

        money_flow_ratio: NDArray[np.float64] = pos_mf_sum / np.where(
            neg_mf_sum == 0, 1e-8, neg_mf_sum
        )

        # MFI calculation
        mfi = (100 - (100 / (1 + money_flow_ratio))).astype(np.float64)

        return np.nan_to_num(mfi, nan=50.0).astype(np.float64)

    @staticmethod
    def _obv_custom(
        close: NDArray[np.float64], volume: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Custom OBV implementation as fallback.
        """
        obv = np.zeros_like(close)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    @staticmethod
    def ad(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        volume: Union[NDArray[np.float64], pd.Series],
    ) -> NDArray[np.float64]:
        """
        Chaikin Accumulation/Distribution (AD) line.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data

        Returns:
            AD values

        Raises:
            TaLibError: If input validation fails
        """
        high = TaLibWrapper._validate_input_data(high, "high")
        low = TaLibWrapper._validate_input_data(low, "low")
        close = TaLibWrapper._validate_input_data(close, "close")
        volume = TaLibWrapper._validate_input_data(volume, "volume")

        if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
            raise TaLibError(
                "High, low, close, and volume arrays must have the same length"
            )

        try:
            if TALIB_AVAILABLE:
                result = talib.AD(high, low, close, volume)
                return cast(NDArray[np.float64], np.nan_to_num(result, nan=np.nan))
            else:
                # Custom AD implementation
                high = np.asarray(high, dtype=float)
                low = np.asarray(low, dtype=float)
                close = np.asarray(close, dtype=float)
                volume = np.asarray(volume, dtype=float)

                money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
                # Handle division by zero and infinities
                money_flow_multiplier = np.nan_to_num(
                    money_flow_multiplier, nan=0.0, posinf=0.0, neginf=0.0
                )
                mfv = money_flow_multiplier * volume
                ad = np.cumsum(mfv)
                return cast(NDArray[np.float64], ad)
        except Exception as e:
            raise TaLibError(f"AD calculation failed: {e}")

    @staticmethod
    def _sar_custom(
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        acceleration: float = 0.02,
        maximum: float = 0.2,
    ) -> NDArray[np.float64]:
        """
        Custom Parabolic SAR implementation as fallback.
        This is a simplified implementation.
        """
        n = len(high)
        sar = np.zeros(n)

        # Initialize
        trend = 1  # 1 for uptrend, -1 for downtrend
        sar[0] = low[0] if trend == 1 else high[0]
        ep = high[0] if trend == 1 else low[0]  # Extreme point
        af = acceleration  # Acceleration factor

        for i in range(1, n):
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            # Check if trend changes
            if trend == 1:
                if low[i] <= sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
            else:  # downtrend
                if high[i] >= sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)

        return sar

    @staticmethod
    def _sma_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """Custom SMA implementation."""
        data = np.asarray(data, dtype=np.float64)
        weights = np.ones(period) / period
        return np.convolve(data, weights, mode="valid").astype(np.float64)

    @staticmethod
    def _ema_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """Custom EMA implementation."""
        data = np.asarray(data, dtype=np.float64)
        return cast(
            NDArray[np.float64],
            pd.Series(data)
            .ewm(span=period, adjust=False)
            .mean()
            .values.astype(np.float64),
        )

    @staticmethod
    def _rolling_sum_custom(
        data: NDArray[np.float64], period: int
    ) -> NDArray[np.float64]:
        """Custom rolling sum implementation."""
        return cast(
            NDArray[np.float64],
            pd.Series(data).rolling(window=period).sum().values.astype(np.float64),
        )

    @staticmethod
    def _rsi_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int = 14
    ) -> NDArray[np.float64]:
        """Custom RSI implementation."""
        data = np.asarray(data, dtype=float)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = (
            pd.Series(gain)
            .ewm(span=period, adjust=False)
            .mean()
            .values.astype(np.float64)
        )
        avg_loss = (
            pd.Series(loss)
            .ewm(span=period, adjust=False)
            .mean()
            .values.astype(np.float64)
        )

        rs = avg_gain / np.where(avg_loss == 0, 1e-8, avg_loss).astype(np.float64)
        rsi = 100 - (100 / (1 + rs))

        # Pad with NaN to match input length
        result = np.full(len(data), np.nan)
        result[period:] = rsi[period - 1 :]
        return result

    @staticmethod
    def _macd_custom(
        data: Union[NDArray[np.float64], pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Custom MACD implementation."""
        data = np.asarray(data, dtype=float)
        ema_fast = TaLibWrapper._ema_custom(data, fast_period)
        ema_slow = TaLibWrapper._ema_custom(data, slow_period)
        macd = ema_fast - ema_slow
        signal = TaLibWrapper._ema_custom(macd, signal_period)
        hist = macd - signal
        return macd, signal, hist

    @staticmethod
    def _bbands_custom(
        data: Union[NDArray[np.float64], pd.Series],
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Custom Bollinger Bands implementation."""
        data = np.asarray(data, dtype=float)

        # Calculate SMA using the same method as _sma_custom
        weights = np.ones(period) / period
        sma = np.convolve(data, weights, mode="valid").astype(np.float64)

        # Calculate rolling standard deviation for the same windows
        # Use centralized calculation
        std_list = calculate_volatility(data, window=period)
        std = np.array(std_list, dtype=np.float64)

        upper = sma + (std * nbdevup)
        lower = sma - (std * nbdevdn)

        # Pad results to match input length
        result_len = len(data)
        upper_padded = np.full(result_len, np.nan)
        lower_padded = np.full(result_len, np.nan)
        sma_padded = np.full(result_len, np.nan)

        offset = period - 1
        upper_padded[offset:] = upper
        lower_padded[offset:] = lower
        sma_padded[offset:] = sma

        return upper_padded, sma_padded, lower_padded

    @staticmethod
    def _atr_custom(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """Custom ATR implementation."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value

        # ATR as EMA of True Range
        return TaLibWrapper._ema_custom(tr, period)

    @staticmethod
    def _stoch_custom(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Custom Stochastic Oscillator implementation."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # Fast %K
        lowest_low = (
            pd.Series(low).rolling(window=fastk_period).min().values.astype(np.float64)
        )
        highest_high = (
            pd.Series(high).rolling(window=fastk_period).max().values.astype(np.float64)
        )
        fastk = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # Slow %K (SMA of Fast %K)
        slowk = TaLibWrapper._sma_custom(fastk, slowk_period)

        # Slow %D (SMA of Slow %K)
        slowd = TaLibWrapper._sma_custom(slowk, slowd_period)

        return slowk, slowd

    @staticmethod
    def _stochf_custom(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        fastk_period: int = 14,
        fastd_period: int = 3,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Custom Stochastic Fast implementation."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # Fast %K
        lowest_low = (
            pd.Series(low).rolling(window=fastk_period).min().values.astype(np.float64)
        )
        highest_high = (
            pd.Series(high).rolling(window=fastk_period).max().values.astype(np.float64)
        )

        # Avoid division by zero
        denom = highest_high - lowest_low
        denom[denom == 0] = np.nan

        fastk = 100 * (close - lowest_low) / denom

        # Fast %D (SMA of Fast %K)
        fastd = TaLibWrapper._sma_custom(fastk, fastd_period)

        return fastk, fastd

    @staticmethod
    def _adx_custom(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """Custom ADX implementation."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]

        # Directional Movement
        dm_plus = high - np.roll(high, 1)
        dm_minus = np.roll(low, 1) - low

        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)

        # Smooth with EMA
        tr_smooth = TaLibWrapper._ema_custom(tr, period)
        dm_plus_smooth = TaLibWrapper._ema_custom(dm_plus, period)
        dm_minus_smooth = TaLibWrapper._ema_custom(dm_minus, period)

        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / np.where(tr_smooth == 0, 1e-8, tr_smooth)
        minus_di = 100 * dm_minus_smooth / np.where(tr_smooth == 0, 1e-8, tr_smooth)

        # DX and ADX
        dx = (
            100
            * np.abs(plus_di - minus_di)
            / np.where(plus_di + minus_di == 0, 1e-8, plus_di + minus_di)
        )
        adx = TaLibWrapper._ema_custom(dx, period)

        return adx

    @staticmethod
    def _williams_r_custom(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 14,
    ) -> NDArray[np.float64]:
        """Custom Williams %R implementation."""
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        highest_high: NDArray[np.float64] = (
            pd.Series(high).rolling(window=period).max().values.astype(np.float64)
        )
        lowest_low: NDArray[np.float64] = (
            pd.Series(low).rolling(window=period).min().values.astype(np.float64)
        )

        denominator = np.where(
            highest_high - lowest_low == 0, 1e-8, highest_high - lowest_low
        ).astype(np.float64)
        williams_r = (-100 * (highest_high - close) / denominator).astype(np.float64)

        return williams_r.astype(np.float64)

    @staticmethod
    def _cci_custom(
        high: Union[NDArray[np.float64], pd.Series],
        low: Union[NDArray[np.float64], pd.Series],
        close: Union[NDArray[np.float64], pd.Series],
        period: int = 20,
    ) -> NDArray[np.float64]:
        """Custom CCI implementation."""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # Typical Price
        tp = (high + low + close) / 3

        # SMA of Typical Price
        tp_sma = TaLibWrapper._sma_custom(tp, period)

        # Mean Deviation
        mean_dev = np.zeros_like(tp)
        for i in range(period - 1, len(tp)):
            mean_dev[i] = np.mean(
                np.abs(tp[i - period + 1 : i + 1] - tp_sma[i - period + 1])
            )

        # CCI
        cci = (tp - tp_sma) / (0.015 * np.where(mean_dev == 0, 1e-8, mean_dev))

        # Pad to match input length
        result = np.full(len(tp), np.nan)
        result[period - 1 :] = cci[period - 1 :]
        return result

    @staticmethod
    def _tema_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """Custom TEMA implementation."""
        data = np.asarray(data, dtype=float)

        # Three EMAs
        ema1 = TaLibWrapper._ema_custom(data, period)
        ema2 = TaLibWrapper._ema_custom(ema1, period)
        ema3 = TaLibWrapper._ema_custom(ema2, period)

        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3

        return tema

    @staticmethod
    def _kama_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int = 30
    ) -> NDArray[np.float64]:
        """Custom KAMA implementation (simplified)."""
        data = np.asarray(data, dtype=float)

        # Simplified KAMA - just use EMA for now
        # Full KAMA implementation would require efficiency ratio calculation
        return TaLibWrapper._ema_custom(data, period)

    @staticmethod
    def _roc_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int = 10
    ) -> NDArray[np.float64]:
        """Custom ROC implementation."""
        data = np.asarray(data, dtype=np.float64)

        rolled_data = np.roll(data, period).astype(np.float64)
        denominator = np.where(rolled_data == 0, 1e-8, rolled_data).astype(np.float64)
        roc = (100 * (data - rolled_data) / denominator).astype(np.float64)

        # Set first 'period' values to NaN
        roc[:period] = np.nan

        return roc

    @staticmethod
    def _wma_custom(
        data: Union[NDArray[np.float64], pd.Series], period: int
    ) -> NDArray[np.float64]:
        """Custom WMA implementation."""
        data = np.asarray(data, dtype=float)
        n = len(data)
        result = np.full(n, np.nan)

        for i in range(period - 1, n):
            weights = np.arange(1, period + 1)
            result[i] = np.sum(data[i - period + 1 : i + 1] * weights) / np.sum(weights)

        return result

    @staticmethod
    def get_indicator_info() -> Dict[str, IndicatorInfo]:
        return {
            "SMA": {
                "description": "Simple Moving Average",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "EMA": {
                "description": "Exponential Moving Average",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "RSI": {
                "description": "Relative Strength Index",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "MACD": {
                "description": "Moving Average Convergence Divergence",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["fast_period", "slow_period", "signal_period"],
            },
            "BBANDS": {
                "description": "Bollinger Bands",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period", "nbdevup", "nbdevdn"],
            },
            "ATR": {
                "description": "Average True Range",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "STOCH": {
                "description": "Stochastic Oscillator",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["fastk_period", "slowk_period", "slowd_period"],
            },
            "ADX": {
                "description": "Average Directional Index",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "PLUS_DI": {
                "description": "Plus Directional Indicator",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "MINUS_DI": {
                "description": "Minus Directional Indicator",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "WILLR": {
                "description": "Williams %R",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "CCI": {
                "description": "Commodity Channel Index",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "TEMA": {
                "description": "Triple Exponential Moving Average",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "WMA": {
                "description": "Weighted Moving Average",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "KAMA": {
                "description": "Kaufman Adaptive Moving Average",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "ROC": {
                "description": "Rate of Change",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
            },
            "MFI": {
                "description": "Money Flow Index",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
                "inputs": ["high", "low", "close", "volume"],
                "output_range": (0, 100),
                "interpretation": "Values above 80 indicate overbought, below 20 indicate oversold",
            },
            "OBV": {
                "description": "On Balance Volume",
                "talib_available": TALIB_AVAILABLE,
                "parameters": [],
                "inputs": ["close", "volume"],
                "output_range": None,
                "interpretation": "Rising OBV indicates accumulation, falling indicates distribution",
            },
            "SAR": {
                "description": "Parabolic SAR",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["acceleration", "maximum"],
                "inputs": ["high", "low"],
                "output_range": None,
                "interpretation": "SAR below price indicates uptrend, above indicates downtrend",
            },
            "ATR": {
                "description": "Average True Range",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
                "inputs": ["high", "low", "close"],
                "output_range": (0, None),
                "interpretation": "Higher ATR indicates higher volatility",
            },
            "BBANDS": {
                "description": "Bollinger Bands",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period", "nbdevup", "nbdevdn"],
                "inputs": ["close"],
                "output_range": None,
                "interpretation": "Price touching upper band indicates overbought, lower band indicates oversold",
            },
            "CCI": {
                "description": "Commodity Channel Index",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["period"],
                "inputs": ["high", "low", "close"],
                "output_range": None,
                "interpretation": "Values above 100 indicate overbought, below -100 indicate oversold",
            },
            "STOCH": {
                "description": "Stochastic Oscillator",
                "talib_available": TALIB_AVAILABLE,
                "parameters": ["fastk_period", "slowk_period", "slowd_period"],
                "inputs": ["high", "low", "close"],
                "output_range": (0, 100),
                "interpretation": "Values above 80 indicate overbought, below 20 indicate oversold",
            },
        }
