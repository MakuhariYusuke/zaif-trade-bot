"""
Base Classes for Technical Indicators

This module provides base classes and interfaces for technical indicator
calculations, promoting code reuse and consistency.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd

from ztb.trading.signal.common.base_classes import BaseIndicatorCalculator
from ztb.trading.signal.common.utilities import validate_market_data
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTechnicalIndicator(BaseIndicatorCalculator):
    """
    Base class for technical indicators

    Provides common functionality for indicator calculation and caching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = self.__class__.__name__.replace('Indicator', '').lower()
        self.required_columns = ['close']  # Override in subclasses

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicator values

        Args:
            data: Market data DataFrame

        Returns:
            Dictionary of indicator values
        """
        # Check cache first
        cached_result = self.get_cached_result(data)
        if cached_result is not None:
            return cached_result

        # Validate input data
        if not validate_market_data(data, self.required_columns):
            logger.warning(f"Invalid data for {self.name} indicator")
            return self._get_default_values()

        # Calculate indicator
        try:
            result = self._calculate_indicator(data)

            # Cache result
            self.cache_result(data, result)

            return result

        except Exception as e:
            logger.error(f"Error calculating {self.name} indicator: {e}")
            return self._get_default_values()

    @abstractmethod
    def _calculate_indicator(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate the specific indicator values"""
        pass

    @abstractmethod
    def _get_default_values(self) -> Dict[str, Any]:
        """Get default values when calculation fails"""
        pass

    def get_required_periods(self) -> int:
        """Get minimum periods required for calculation"""
        return self.config.get('periods', 14)


class BaseOscillatorIndicator(BaseTechnicalIndicator):
    """
    Base class for oscillator-type indicators (0-100 range)

    Examples: RSI, Stochastic, CCI
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.oscillator_range = (0.0, 100.0)

    def _validate_oscillator_value(self, value: float) -> float:
        """Validate and clamp oscillator value to valid range"""
        min_val, max_val = self.oscillator_range
        return max(min_val, min(max_val, value))

    def get_oscillator_signal(self, value: float) -> str:
        """
        Get oscillator signal interpretation

        Returns:
            'oversold', 'overbought', or 'neutral'
        """
        if value <= 30:
            return 'oversold'
        elif value >= 70:
            return 'overbought'
        else:
            return 'neutral'


class BaseTrendIndicator(BaseTechnicalIndicator):
    """
    Base class for trend-following indicators

    Examples: Moving Averages, MACD, ADX
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.required_columns = ['open', 'high', 'low', 'close']

    def get_trend_direction(self, value: float) -> str:
        """
        Get trend direction interpretation

        Returns:
            'bullish', 'bearish', or 'sideways'
        """
        if value > 0.5:
            return 'bullish'
        elif value < -0.5:
            return 'bearish'
        else:
            return 'sideways'


class BaseVolatilityIndicator(BaseTechnicalIndicator):
    """
    Base class for volatility indicators

    Examples: Bollinger Bands, ATR, Standard Deviation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.required_columns = ['high', 'low', 'close']

    def get_volatility_regime(self, value: float) -> str:
        """
        Get volatility regime interpretation

        Returns:
            'low', 'normal', 'high', or 'extreme'
        """
        if value <= 0.02:
            return 'low'
        elif value <= 0.05:
            return 'normal'
        elif value <= 0.10:
            return 'high'
        else:
            return 'extreme'


class BaseVolumeIndicator(BaseTechnicalIndicator):
    """
    Base class for volume-based indicators

    Examples: Volume Moving Average, OBV, Volume Rate of Change
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.required_columns = ['close', 'volume']

    def get_volume_signal(self, value: float) -> str:
        """
        Get volume signal interpretation

        Returns:
            'increasing', 'decreasing', or 'neutral'
        """
        if value > 1.2:
            return 'increasing'
        elif value < 0.8:
            return 'decreasing'
        else:
            return 'neutral'


class CompositeIndicator(BaseTechnicalIndicator):
    """
    Composite indicator that combines multiple base indicators

    Allows building complex indicators from simpler components.
    """

    def __init__(self, indicators: List[BaseTechnicalIndicator], weights: Optional[Dict[str, float]] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.indicators = indicators
        self.weights = weights or self._get_equal_weights()
        self.name = 'composite'

    def _get_equal_weights(self) -> Dict[str, float]:
        """Get equal weights for all indicators"""
        if not self.indicators:
            return {}
        equal_weight = 1.0 / len(self.indicators)
        return {indicator.name: equal_weight for indicator in self.indicators}

    def _calculate_indicator(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate composite indicator"""
        component_results = {}

        # Calculate each component indicator
        for indicator in self.indicators:
            try:
                result = indicator.calculate(data)
                component_results.update(result)
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator.name}: {e}")
                # Use default values
                default_values = indicator._get_default_values()
                component_results.update(default_values)

        # Combine results using weights
        return self._combine_components(component_results)

    def _combine_components(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine component results into final composite score"""
        # Simple weighted average of numeric values
        composite_score = 0.0
        total_weight = 0.0

        for key, value in component_results.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                weight = self.weights.get(key, 0.1)  # Default small weight
                composite_score += value * weight
                total_weight += weight

        if total_weight > 0:
            composite_score /= total_weight

        return {
            'composite_score': composite_score,
            'component_count': len(self.indicators),
            'total_weight': total_weight
        }

    def _get_default_values(self) -> Dict[str, Any]:
        """Get default values when calculation fails"""
        return {
            'composite_score': 50.0,
            'component_count': 0,
            'total_weight': 0.0
        }


class AdaptiveIndicator(BaseTechnicalIndicator):
    """
    Adaptive indicator that adjusts parameters based on market conditions

    Uses market regime information to optimize indicator parameters.
    """

    def __init__(self, base_indicator: BaseTechnicalIndicator, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_indicator = base_indicator
        self.market_regime = None
        self.adaptive_params = self.config.get('adaptive_params', {})

    def set_market_regime(self, regime: str):
        """Set current market regime for adaptation"""
        self.market_regime = regime

    def get_adaptive_config(self) -> Dict[str, Any]:
        """Get regime-adapted configuration"""
        if self.market_regime and self.market_regime in self.adaptive_params:
            base_config = self.base_indicator.config.copy()
            base_config.update(self.adaptive_params[self.market_regime])
            return base_config
        return self.base_indicator.config

    def _calculate_indicator(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicator with adaptive parameters"""
        # Temporarily override base indicator config
        original_config = self.base_indicator.config
        adaptive_config = self.get_adaptive_config()

        self.base_indicator.config = adaptive_config

        try:
            result = self.base_indicator.calculate(data)
            # Add adaptation metadata
            result['adaptive_regime'] = self.market_regime or 'none'
            result['adapted_config'] = bool(self.market_regime in self.adaptive_params)
            return result
        finally:
            # Restore original config
            self.base_indicator.config = original_config

    def _get_default_values(self) -> Dict[str, Any]:
        """Get default values from base indicator"""
        defaults = self.base_indicator._get_default_values()
        defaults.update({
            'adaptive_regime': 'none',
            'adapted_config': False
        })
        return defaults

    def adapt_parameters(self, market_regime: str) -> Dict[str, Any]:
        """
        Adapt indicator parameters based on market regime

        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile')

        Returns:
            Adapted configuration parameters
        """
        if market_regime in self.adaptive_params:
            return self.adaptive_params[market_regime]
        return {}

    def calculate_adaptive(self, data: pd.DataFrame, market_regime: str) -> Dict[str, Any]:
        """
        Calculate indicator with adaptive parameters for given regime

        Args:
            data: Market data
            market_regime: Current market regime

        Returns:
            Indicator values with adaptation
        """
        # Adapt parameters for the regime
        adapted_params = self.adapt_parameters(market_regime)

        # Update base indicator config temporarily
        original_config = self.base_indicator.config.copy()
        self.base_indicator.config.update(adapted_params)

        try:
            result = self.base_indicator.calculate(data)
            # Add adaptation metadata
            result['adaptive_regime'] = market_regime
            result['adapted_config'] = bool(adapted_params)
            return result
        finally:
            # Restore original config
            self.base_indicator.config = original_config
