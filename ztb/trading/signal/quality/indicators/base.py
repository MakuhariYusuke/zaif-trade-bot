"""
Base classes for technical indicators.

This module provides base classes and interfaces for technical indicator
calculations, promoting code reuse and consistency.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from ztb.trading.signal.common.base_classes import BaseIndicatorCalculator
from ztb.trading.signal.common.utilities import normalize_weights, validate_market_data
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

IndicatorConfig = dict[str, object]
IndicatorResult = dict[str, object]
IndicatorWeights = dict[str, float]

def _normalize_mapping(value: Mapping[str, object] | None) -> IndicatorConfig:
    """Normalize mapping keys to strings."""
    if value is None:
        return {}
    return {str(key): mapped for key, mapped in value.items()}

def _to_int(value: object, default: int, minimum: int | None = None) -> int:
    """Convert object to int with optional lower bound."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None and result < minimum:
        return minimum
    return result

class BaseTechnicalIndicator(BaseIndicatorCalculator):
    """
    Base class for technical indicators.

    Provides common functionality for indicator calculation and caching.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        default_config = self._get_default_config()
        merged_config = _normalize_mapping(default_config)
        merged_config.update(_normalize_mapping(config))

        super().__init__(merged_config)
        self.config: IndicatorConfig = _normalize_mapping(self.config)
        self.name = self.__class__.__name__.replace("Indicator", "").lower()
        self.required_columns: list[str] = ["close"]  # Override in subclasses.
        self.on_config_updated()

    def _get_default_config(self) -> IndicatorConfig:
        """Get default configuration for this indicator."""
        return {}

    def on_config_updated(self) -> None:
        """
        Hook for subclasses when config is updated.

        Used by adaptive indicators that temporarily override indicator config.
        """

    @contextmanager
    def temporary_config(self, updates: Mapping[str, object] | None) -> Iterator[None]:
        """Temporarily apply config updates and always restore original config."""
        original_config = dict(self.config)
        normalized_updates = _normalize_mapping(updates)
        if normalized_updates:
            self.config.update(normalized_updates)
            self.on_config_updated()
        try:
            yield
        finally:
            self.config = original_config
            self.on_config_updated()

    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        """
        Calculate indicator values.

        Args:
            data: Market data DataFrame.

        Returns:
            Dictionary of indicator values.
        """
        cached_result = self.get_cached_result(data)
        if cached_result is not None:
            return dict(cached_result)

        if not validate_market_data(data, self.required_columns):
            logger.warning(f"Invalid data for {self.name} indicator")
            return dict(self._get_default_values())

        try:
            result = self._calculate_indicator(data)
            normalized_result = {
                str(key): value for key, value in result.items() if isinstance(key, str)
            }
            self.cache_result(data, normalized_result)
            return dict(normalized_result)
        except Exception as exc:
            logger.error(f"Error calculating {self.name} indicator: {exc}")
            return dict(self._get_default_values())

    @abstractmethod
    def _calculate_indicator(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate the specific indicator values."""

    @abstractmethod
    def _get_default_values(self) -> IndicatorResult:
        """Get default values when calculation fails."""

    def _get_config_int(self, key: str, default: int, minimum: int | None = None) -> int:
        """Read integer config value with fallback."""
        return _to_int(self.config.get(key, default), default, minimum=minimum)

    def get_required_periods(self) -> int:
        """Get minimum periods required for calculation."""
        return self._get_config_int("periods", 14, minimum=1)

class BaseOscillatorIndicator(BaseTechnicalIndicator):
    """
    Base class for oscillator-type indicators (0-100 range).

    Examples: RSI, Stochastic, CCI.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.oscillator_range = (0.0, 100.0)

    def _validate_oscillator_value(self, value: float) -> float:
        """Validate and clamp oscillator value to valid range."""
        min_val, max_val = self.oscillator_range
        return max(min_val, min(max_val, value))

    def get_oscillator_signal(self, value: float) -> str:
        """
        Get oscillator signal interpretation.

        Returns:
            "oversold", "overbought", or "neutral".
        """
        if value <= 30:
            return "oversold"
        if value >= 70:
            return "overbought"
        return "neutral"

class BaseTrendIndicator(BaseTechnicalIndicator):
    """
    Base class for trend-following indicators.

    Examples: Moving Averages, MACD, ADX.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.required_columns = ["open", "high", "low", "close"]

    def get_trend_direction(self, value: float) -> str:
        """
        Get trend direction interpretation.

        Returns:
            "bullish", "bearish", or "sideways".
        """
        if value > 0.5:
            return "bullish"
        if value < -0.5:
            return "bearish"
        return "sideways"

class BaseVolatilityIndicator(BaseTechnicalIndicator):
    """
    Base class for volatility indicators.

    Examples: Bollinger Bands, ATR, Standard Deviation.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.required_columns = ["high", "low", "close"]

    def get_volatility_regime(self, value: float) -> str:
        """
        Get volatility regime interpretation.

        Returns:
            "low", "normal", "high", or "extreme".
        """
        if value <= 0.02:
            return "low"
        if value <= 0.05:
            return "normal"
        if value <= 0.10:
            return "high"
        return "extreme"

class BaseVolumeIndicator(BaseTechnicalIndicator):
    """
    Base class for volume-based indicators.

    Examples: Volume Moving Average, OBV, Volume Rate of Change.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        super().__init__(config)
        self.required_columns = ["close", "volume"]

    def get_volume_signal(self, value: float) -> str:
        """
        Get volume signal interpretation.

        Returns:
            "increasing", "decreasing", or "neutral".
        """
        if value > 1.2:
            return "increasing"
        if value < 0.8:
            return "decreasing"
        return "neutral"

class CompositeIndicator(BaseTechnicalIndicator):
    """
    Composite indicator that combines multiple base indicators.

    Allows building complex indicators from simpler components.
    """

    def __init__(
        self,
        indicators: Sequence[BaseTechnicalIndicator],
        weights: Mapping[str, float] | None = None,
        config: Mapping[str, object] | None = None,
    ):
        super().__init__(config)
        self.indicators = list(indicators)
        self.weights = self._resolve_weights(weights)
        self.name = "composite"

    def _resolve_weights(self, weights: Mapping[str, float] | None) -> IndicatorWeights:
        """Resolve and normalize indicator weights."""
        if not self.indicators:
            return {}

        if weights is None:
            return self._get_equal_weights()

        indicator_names = [indicator.name for indicator in self.indicators]
        normalized: IndicatorWeights = {}
        for name in indicator_names:
            try:
                raw_weight = float(weights.get(name, 0.0))
            except (TypeError, ValueError):
                raw_weight = 0.0
            normalized[name] = max(0.0, raw_weight)

        if sum(normalized.values()) <= 0.0:
            return self._get_equal_weights()
        return normalize_weights(normalized)

    def _get_equal_weights(self) -> IndicatorWeights:
        """Get equal weights for all indicators."""
        if not self.indicators:
            return {}
        equal_weight = 1.0 / len(self.indicators)
        return {indicator.name: equal_weight for indicator in self.indicators}

    def _calculate_indicator(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate composite indicator."""
        component_results: IndicatorResult = {}

        for indicator in self.indicators:
            try:
                result = indicator.calculate(data)
                component_results.update(result)
            except Exception as exc:
                logger.warning(f"Failed to calculate {indicator.name}: {exc}")
                default_values = indicator._get_default_values()
                component_results.update(default_values)

        return self._combine_components(component_results)

    def _combine_components(self, component_results: Mapping[str, object]) -> IndicatorResult:
        """Combine component results into final composite score."""
        composite_score = 0.0
        total_weight = 0.0

        for key, value in component_results.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                weight = self.weights.get(key, 0.1)  # Default small weight.
                composite_score += float(value) * weight
                total_weight += weight

        if total_weight > 0:
            composite_score /= total_weight

        return {
            "composite_score": composite_score,
            "component_count": len(self.indicators),
            "total_weight": total_weight,
        }

    def _get_default_values(self) -> IndicatorResult:
        """Get default values when calculation fails."""
        return {
            "composite_score": 50.0,
            "component_count": 0,
            "total_weight": 0.0,
        }

class AdaptiveIndicator(BaseTechnicalIndicator):
    """
    Adaptive indicator that adjusts parameters based on market conditions.

    Uses market regime information to optimize indicator parameters.
    """

    def __init__(
        self,
        base_indicator: BaseTechnicalIndicator,
        config: Mapping[str, object] | None = None,
    ):
        super().__init__(config)
        self.base_indicator = base_indicator
        self.market_regime: str | None = None
        self.adaptive_params = self._parse_adaptive_params(
            self.config.get("adaptive_params", {})
        )

    @staticmethod
    def _parse_adaptive_params(raw: object) -> dict[str, IndicatorConfig]:
        """Parse adaptive params into regime -> config mapping."""
        if not isinstance(raw, Mapping):
            return {}

        parsed: dict[str, IndicatorConfig] = {}
        for regime, params in raw.items():
            if isinstance(regime, str) and isinstance(params, Mapping):
                parsed[regime] = _normalize_mapping(params)
        return parsed

    def set_market_regime(self, regime: str | None) -> None:
        """set current market regime for adaptation."""
        if isinstance(regime, str) and regime:
            self.market_regime = regime
        else:
            self.market_regime = None

    def get_adaptive_config(self) -> IndicatorConfig:
        """Get regime-adapted configuration."""
        base_config = _normalize_mapping(self.base_indicator.config)
        if self.market_regime and self.market_regime in self.adaptive_params:
            base_config.update(self.adaptive_params[self.market_regime])
        return base_config

    @contextmanager
    def _temporary_base_config(self, updates: Mapping[str, object]) -> Iterator[None]:
        """
        Temporarily apply config updates to base indicator.

        Falls back to manual update for mocked base indicators used by tests.
        """
        temporary_config = getattr(self.base_indicator, "temporary_config", None)
        if callable(temporary_config):
            context = temporary_config(updates)
            if hasattr(context, "__enter__") and hasattr(context, "__exit__"):
                with context:
                    yield
                return

        original_config = _normalize_mapping(getattr(self.base_indicator, "config", {}))
        normalized_updates = _normalize_mapping(updates)
        if normalized_updates:
            self.base_indicator.config.update(normalized_updates)
            on_config_updated = getattr(self.base_indicator, "on_config_updated", None)
            if callable(on_config_updated):
                on_config_updated()
        try:
            yield
        finally:
            self.base_indicator.config = original_config
            on_config_updated = getattr(self.base_indicator, "on_config_updated", None)
            if callable(on_config_updated):
                on_config_updated()

    def _calculate_with_regime(
        self, data: pd.DataFrame, market_regime: str | None
    ) -> IndicatorResult:
        """Run base indicator with regime-adapted parameters."""
        adapted_params = self.adapt_parameters(market_regime)
        with self._temporary_base_config(adapted_params):
            base_result = self.base_indicator.calculate(data)

        # Copy to avoid mutating cached dict returned by base indicator.
        result = dict(base_result)
        result["adaptive_regime"] = market_regime or "none"
        result["adapted_config"] = bool(adapted_params)
        return result

    def _calculate_indicator(self, data: pd.DataFrame) -> IndicatorResult:
        """Calculate indicator with currently configured adaptive regime."""
        return self._calculate_with_regime(data, self.market_regime)

    def _get_default_values(self) -> IndicatorResult:
        """Get default values from base indicator."""
        defaults = dict(self.base_indicator._get_default_values())
        defaults.update({"adaptive_regime": "none", "adapted_config": False})
        return defaults

    def adapt_parameters(self, market_regime: str | None) -> IndicatorConfig:
        """
        Adapt indicator parameters based on market regime.

        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile').

        Returns:
            Adapted configuration parameters.
        """
        if isinstance(market_regime, str):
            return dict(self.adaptive_params.get(market_regime, {}))
        return {}

    def calculate_adaptive(
        self, data: pd.DataFrame, market_regime: str | None
    ) -> IndicatorResult:
        """
        Calculate indicator with adaptive parameters for a given regime.

        Args:
            data: Market data.
            market_regime: Current market regime.

        Returns:
            Indicator values with adaptation metadata.
        """
        return self._calculate_with_regime(data, market_regime)
