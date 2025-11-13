"""
Common Base Classes for Signal Processing

This module provides shared base classes and interfaces to reduce code duplication
across different signal processing components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SignalContext:
    """Common signal processing context"""

    market_data: pd.DataFrame
    position_context: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    timestamp: pd.Timestamp


@dataclass
class SignalResult:
    """Common signal processing result"""

    discrete_action: int  # -1, 0, 1
    quality_score: float  # 0-100
    confidence: float     # 0-1
    metadata: Dict[str, Any]


class BaseSignalProcessor(ABC):
    """
    Base class for all signal processing components

    Provides common functionality and interface standardization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        pass

    @abstractmethod
    def process_signal(self, context: SignalContext) -> SignalResult:
        """Process signal with given context"""
        pass

    def validate_input(self, context: SignalContext) -> bool:
        """Validate input context"""
        required_fields = ['market_data', 'position_context', 'portfolio_state']
        for field in required_fields:
            if not hasattr(context, field):
                self.logger.error(f"Missing required field: {field}")
                return False
        return True

    def log_processing_result(self, result: SignalResult, context: SignalContext):
        """Log signal processing result"""
        self.logger.debug(
            f"Signal processed - Action: {result.discrete_action}, "
            f"Score: {result.quality_score:.2f}, Confidence: {result.confidence:.2f}"
        )


class BaseIndicatorCalculator(ABC):
    """
    Base class for technical indicator calculations

    Standardizes indicator calculation interface and provides caching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._cache = {}

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicator values"""
        pass

    def get_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key for data"""
        # Simple cache key based on data shape and last index value
        if len(data) == 0:
            return "empty"

        # Handle different index types
        last_index = data.index[-1]
        if hasattr(last_index, 'isoformat'):
            index_str = last_index.isoformat()
        else:
            index_str = str(last_index)

        return f"{len(data)}_{index_str}"

    def get_cached_result(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Get cached calculation result"""
        cache_key = self.get_cache_key(data)
        return self._cache.get(cache_key)

    def cache_result(self, data: pd.DataFrame, result: Dict[str, float]):
        """Cache calculation result"""
        cache_key = self.get_cache_key(data)
        self._cache[cache_key] = result

        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]


class BaseSignalScorer(ABC):
    """
    Base class for signal quality scoring

    Provides common scoring functionality and threshold management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.weights = self.config.get('weights', {})
        self.thresholds = self.config.get('thresholds', {})

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        pass

    @abstractmethod
    def calculate_score(self, indicators: Dict[str, float],
                       context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate signal quality score"""
        pass

    def apply_thresholds(self, score: float) -> Tuple[int, float]:
        """
        Apply thresholds to convert score to discrete action

        Returns:
            Tuple of (discrete_action, confidence)
        """
        buy_threshold = self.thresholds.get('buy', 75)
        sell_threshold = self.thresholds.get('sell', 25)
        hold_threshold = self.thresholds.get('hold', 45)

        if score >= buy_threshold:
            action = 1  # BUY
            confidence = min(1.0, (score - buy_threshold) / (100 - buy_threshold))
        elif score <= sell_threshold:
            action = -1  # SELL
            confidence = min(1.0, (sell_threshold - score) / sell_threshold)
        else:
            action = 0  # HOLD
            # Confidence based on distance from decision boundaries
            distance_to_buy = abs(score - buy_threshold)
            distance_to_sell = abs(score - sell_threshold)
            min_distance = min(distance_to_buy, distance_to_sell)
            confidence = min(1.0, min_distance / 10.0)  # Scale by 10 points

        return action, confidence

    def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights"""
        self.weights.update(new_weights)
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}


class BaseRegimeAdapter(ABC):
    """
    Base class for market regime adaptation

    Provides common adaptation functionality and parameter management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.adaptation_params = self.config.get('adaptation_params', {})

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        pass

    @abstractmethod
    def adapt_parameters(self, regime_type: str,
                        base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt parameters for specific regime"""
        pass

    def get_regime_config(self, regime_type: str) -> Dict[str, Any]:
        """Get configuration for specific regime"""
        return self.adaptation_params.get(regime_type, {})

    def validate_regime_type(self, regime_type: str) -> bool:
        """Validate regime type"""
        return regime_type in self.adaptation_params