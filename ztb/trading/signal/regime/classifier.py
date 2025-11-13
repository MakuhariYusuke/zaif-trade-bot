"""
Market Regime Classifier

Enhanced market regime classification system with 16 distinct regimes
for adaptive signal processing and trading strategy optimization.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from ztb.trading.signal.common.base_classes import BaseSignalProcessor, SignalContext, SignalResult
from ztb.trading.signal.common.metrics import (
    calculate_trend_metrics, calculate_volatility_metrics,
    calculate_momentum_metrics, calculate_volume_metrics,
    calculate_support_resistance_metrics
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RegimeType:
    """Market regime type constants"""

    # SELL特化レジーム（最高優先度）
    SELL_BREAKDOWN = "sell_breakdown"
    SELL_DIVERGENCE = "sell_divergence"
    SELL_MOMENTUM_WEAK = "sell_momentum_weak"
    SELL_VOLUME_SURGE = "sell_volume_surge"

    # Bullトレンドレジーム
    STRONG_BULL_TREND = "strong_bull_trend"
    MODERATE_BULL_TREND = "moderate_bull_trend"
    WEAK_BULL_TREND = "weak_bull_trend"

    # Bearトレンドレジーム
    STRONG_BEAR_TREND = "strong_bear_trend"
    MODERATE_BEAR_TREND = "moderate_bear_trend"
    WEAK_BEAR_TREND = "weak_bear_trend"

    # レンジ相場レジーム
    HIGH_VOLATILITY_RANGE = "high_volatility_range"
    MODERATE_VOLATILITY_RANGE = "moderate_volatility_range"
    LOW_VOLATILITY_RANGE = "low_volatility_range"

    # 特殊条件レジーム
    EXTREME_VOLATILITY = "extreme_volatility"
    CONSOLIDATION = "consolidation"
    BREAKOUT_SETUP = "breakout_setup"
    BREAKDOWN_SETUP = "breakdown_setup"


class MarketRegimeClassifier(BaseSignalProcessor):
    """
    Enhanced market regime classifier with 16 distinct regimes

    Provides sophisticated market regime detection for adaptive trading
    strategies and signal processing optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.regime_definitions = self._initialize_regime_definitions()
        self.regime_history = []

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'lookback_periods': {'short': 20, 'medium': 50, 'long': 100},
            'regime_scheme': 'comprehensive',
            'confidence_threshold': 0.6,
            'max_history': 1000
        }

    def _initialize_regime_definitions(self) -> List[Dict[str, Any]]:
        """Initialize the 16 regime definitions with priority-based classification"""
        return [
            # SELL特化レジーム（最高優先度）
            {
                'name': 'Sell Breakdown',
                'type': RegimeType.SELL_BREAKDOWN,
                'priority': 16,
                'conditions': {
                    'trend_strength': {'max': -2.5},
                    'bear_strength': {'min': 1.8},
                    'volatility': {'min': 0.08},
                    'price_range_ratio': {'min': 0.02}
                },
                'description': 'Strong breakdown pattern - SELL signal priority'
            },
            {
                'name': 'Sell Divergence',
                'type': RegimeType.SELL_DIVERGENCE,
                'priority': 15,
                'conditions': {
                    'trend_strength': {'min': -1.5, 'max': 1.5},
                    'bear_strength': {'min': 1.2},
                    'macd_signal': {'max': -0.5},
                    'rsi': {'max': 65}
                },
                'description': 'Bearish divergence detected - SELL opportunity'
            },
            {
                'name': 'Sell Momentum Weak',
                'type': RegimeType.SELL_MOMENTUM_WEAK,
                'priority': 14,
                'conditions': {
                    'trend_strength': {'max': -1.0},
                    'momentum': {'max': -0.3},
                    'volatility': {'min': 0.05},
                    'adx': {'min': 20}
                },
                'description': 'Weakening momentum in downtrend - SELL reinforcement'
            },
            {
                'name': 'Sell Volume Surge',
                'type': RegimeType.SELL_VOLUME_SURGE,
                'priority': 13,
                'conditions': {
                    'trend_strength': {'max': -1.2},
                    'volume_trend': {'min': 0.15},
                    'price_range_ratio': {'min': 0.015},
                    'bollinger_position': {'max': 0.3}
                },
                'description': 'Volume surge in downtrend - SELL confirmation'
            },

            # Bullトレンドレジーム
            {
                'name': 'Strong Bull Trend',
                'type': RegimeType.STRONG_BULL_TREND,
                'priority': 12,
                'conditions': {
                    'trend_strength': {'min': 3.0},
                    'bull_strength': {'min': 2.5},
                    'volatility': {'max': 0.15}
                },
                'description': 'Strong upward momentum with high conviction'
            },
            {
                'name': 'Moderate Bull Trend',
                'type': RegimeType.MODERATE_BULL_TREND,
                'priority': 11,
                'conditions': {
                    'trend_strength': {'min': 2.0, 'max': 3.0},
                    'bull_strength': {'min': 1.5, 'max': 2.5},
                    'volatility': {'max': 0.20}
                },
                'description': 'Moderate upward trend with steady gains'
            },
            {
                'name': 'Weak Bull Trend',
                'type': RegimeType.WEAK_BULL_TREND,
                'priority': 10,
                'conditions': {
                    'trend_strength': {'min': 1.0, 'max': 2.0},
                    'bull_strength': {'min': 0.5, 'max': 1.5},
                    'volatility': {'max': 0.25}
                },
                'description': 'Weak upward movement with low momentum'
            },

            # Bearトレンドレジーム
            {
                'name': 'Strong Bear Trend',
                'type': RegimeType.STRONG_BEAR_TREND,
                'priority': 9,
                'conditions': {
                    'trend_strength': {'max': -2.8},
                    'bear_strength': {'min': 2.2},
                    'volatility': {'max': 0.18}
                },
                'description': 'Strong downward momentum with high conviction'
            },
            {
                'name': 'Moderate Bear Trend',
                'type': RegimeType.MODERATE_BEAR_TREND,
                'priority': 8,
                'conditions': {
                    'trend_strength': {'max': -1.8, 'min': -2.8},
                    'bear_strength': {'min': 1.3, 'max': 2.2},
                    'volatility': {'max': 0.22}
                },
                'description': 'Moderate downward trend with steady losses'
            },
            {
                'name': 'Weak Bear Trend',
                'type': RegimeType.WEAK_BEAR_TREND,
                'priority': 7,
                'conditions': {
                    'trend_strength': {'max': -0.8, 'min': -1.8},
                    'bear_strength': {'min': 0.3, 'max': 1.3},
                    'volatility': {'max': 0.28}
                },
                'description': 'Weak downward movement with low momentum'
            },

            # レンジ相場レジーム
            {
                'name': 'High Volatility Range',
                'type': RegimeType.HIGH_VOLATILITY_RANGE,
                'priority': 6,
                'conditions': {
                    'volatility': {'min': 0.15},
                    'trend_strength': {'min': -2.0, 'max': 2.0}
                },
                'description': 'High volatility sideways movement'
            },
            {
                'name': 'Moderate Volatility Range',
                'type': RegimeType.MODERATE_VOLATILITY_RANGE,
                'priority': 5,
                'conditions': {
                    'volatility': {'min': 0.10, 'max': 0.15},
                    'trend_strength': {'min': -1.5, 'max': 1.5}
                },
                'description': 'Moderate volatility consolidation'
            },
            {
                'name': 'Low Volatility Range',
                'type': RegimeType.LOW_VOLATILITY_RANGE,
                'priority': 4,
                'conditions': {
                    'volatility': {'max': 0.10},
                    'trend_strength': {'min': -1.0, 'max': 1.0}
                },
                'description': 'Low volatility tight range'
            },

            # 特殊条件レジーム
            {
                'name': 'Extreme Volatility',
                'type': RegimeType.EXTREME_VOLATILITY,
                'priority': 3,
                'conditions': {'volatility': {'min': 0.20}},
                'description': 'Extreme market volatility conditions'
            },
            {
                'name': 'Consolidation',
                'type': RegimeType.CONSOLIDATION,
                'priority': 2,
                'conditions': {
                    'volatility': {'max': 0.08},
                    'trend_strength': {'min': -0.8, 'max': 0.8}
                },
                'description': 'Tight consolidation with minimal movement'
            },
            {
                'name': 'Breakout Setup',
                'type': RegimeType.BREAKOUT_SETUP,
                'priority': 1,
                'conditions': {
                    'volatility': {'max': 0.12},
                    'trend_strength': {'min': -1.2, 'max': 1.2},
                    'support_resistance_strength': {'min': 0.7}
                },
                'description': 'Potential breakout from consolidation'
            },
            {
                'name': 'Breakdown Setup',
                'type': RegimeType.BREAKDOWN_SETUP,
                'priority': 1,
                'conditions': {
                    'volatility': {'max': 0.12},
                    'trend_strength': {'min': -1.2, 'max': 1.2},
                    'support_resistance_strength': {'min': 0.7}
                },
                'description': 'Potential breakdown from consolidation'
            }
        ]

    def detect_regime(self, data: pd.DataFrame, current_index: int = -1) -> Dict[str, Any]:
        """
        Detect current market regime from price data

        Args:
            data: OHLCV DataFrame
            current_index: Index to analyze (default: latest)

        Returns:
            Dictionary containing regime detection results
        """
        if current_index == -1:
            current_index = len(data) - 1

        # Calculate comprehensive regime metrics
        metrics = self._calculate_regime_metrics(data, current_index)

        # Classify regime using priority-based system
        regime_type, confidence, classification_path = self._classify_regime(metrics)

        # Create result
        result = {
            'primary_regime': regime_type,
            'confidence': confidence,
            'secondary_regimes': self._calculate_secondary_regimes(metrics, regime_type),
            'metrics': metrics,
            'detection_timestamp': data.index[current_index] if hasattr(data.index, '__getitem__') and current_index < len(data.index) else pd.Timestamp.now(),
            'lookback_period': self.config['lookback_periods']['medium'],
            'classification_path': classification_path
        }

        # Update history
        self.regime_history.append(result)
        if len(self.regime_history) > self.config['max_history']:
            self.regime_history = self.regime_history[-self.config['max_history']:]

        return result

    def _calculate_regime_metrics(self, data: pd.DataFrame, index: int) -> Dict[str, float]:
        """Calculate comprehensive regime detection metrics"""
        # Ensure we have enough data
        min_periods = max(self.config['lookback_periods'].values())
        if index < min_periods:
            return self._get_default_metrics()

        # Extract data window
        start_idx = max(0, index - self.config['lookback_periods']['long'])
        data_window = data.iloc[start_idx:index + 1]

        # Calculate various metrics
        trend_metrics = calculate_trend_metrics(data_window)
        volatility_metrics = calculate_volatility_metrics(data_window)
        momentum_metrics = calculate_momentum_metrics(data_window)
        volume_metrics = calculate_volume_metrics(data_window)
        sr_metrics = calculate_support_resistance_metrics(data_window)

        # Calculate additional technical indicators
        additional_metrics = self._calculate_additional_indicators(data_window)

        # Combine all metrics
        metrics = {}
        metrics.update(trend_metrics)
        metrics.update(volatility_metrics)
        metrics.update(momentum_metrics)
        metrics.update(volume_metrics)
        metrics.update(sr_metrics)
        metrics.update(additional_metrics)

        return metrics

    def _calculate_additional_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional technical indicators for regime detection"""
        if len(data) < 14:
            return {
                'adx': 0.0,
                'rsi': 50.0,
                'macd_signal': 0.0,
                'bollinger_position': 0.5,
                'price_range_ratio': 0.0
            }

        close = data['close']
        high = data['high']
        low = data['low']

        # ADX calculation (simplified)
        adx = self._calculate_adx(high, low, close)

        # RSI calculation
        rsi = self._calculate_rsi(close)

        # MACD signal
        macd_signal = self._calculate_macd_signal(close)

        # Bollinger Band position
        bollinger_position = self._calculate_bollinger_position(close)

        # Price range ratio
        price_range_ratio = (high - low) / close.shift(1)
        price_range_ratio = price_range_ratio.rolling(10).mean().iloc[-1] if not price_range_ratio.empty else 0.0

        return {
            'adx': adx,
            'rsi': rsi,
            'macd_signal': macd_signal,
            'bollinger_position': bollinger_position,
            'price_range_ratio': price_range_ratio
        }

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate ADX (simplified)"""
        try:
            high_diff = high - high.shift(1)
            low_diff = low.shift(1) - low

            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

            tr = np.maximum(
                high - low,
                np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
            )

            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / pd.Series(tr).rolling(14).mean()
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / pd.Series(tr).rolling(14).mean()

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()

            return float(adx.iloc[-1]) if not adx.empty else 0.0
        except:
            return 0.0

    def _calculate_rsi(self, close: pd.Series) -> float:
        """Calculate RSI"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except:
            return 50.0

    def _calculate_macd_signal(self, close: pd.Series) -> float:
        """Calculate MACD signal"""
        try:
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_signal = macd - signal

            return float(macd_signal.iloc[-1]) if not macd_signal.empty else 0.0
        except:
            return 0.0

    def _calculate_bollinger_position(self, close: pd.Series) -> float:
        """Calculate Bollinger Band position"""
        try:
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            band_width = upper_band - lower_band
            position = (close - lower_band) / band_width

            return float(position.iloc[-1]) if not position.empty else 0.5
        except:
            return 0.5

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when insufficient data"""
        return {
            'trend_strength': 0.0,
            'bull_strength': 0.0,
            'bear_strength': 0.0,
            'volatility': 0.0,
            'momentum': 0.0,
            'volume_trend': 0.0,
            'price_range_ratio': 0.0,
            'adx': 0.0,
            'rsi': 50.0,
            'macd_signal': 0.0,
            'bollinger_position': 0.5,
            'support_resistance_strength': 0.0
        }

    def _classify_regime(self, metrics: Dict[str, float]) -> Tuple[str, float, List[str]]:
        """Classify regime using priority-based system"""
        classification_path = []

        # Evaluate each regime definition by priority
        for regime_def in sorted(self.regime_definitions, key=lambda x: x['priority'], reverse=True):
            score, confidence = self._evaluate_regime_conditions(metrics, regime_def)

            if confidence >= self.config['confidence_threshold']:
                classification_path.append(regime_def['name'])
                return regime_def['type'], confidence, classification_path

            classification_path.append(f"{regime_def['name']} (failed)")

        # Default to consolidation if no regime matches well
        classification_path.append("Default: Consolidation")
        return RegimeType.CONSOLIDATION, 0.5, classification_path

    def _evaluate_regime_conditions(self, metrics: Dict[str, float],
                                  regime_def: Dict[str, Any]) -> Tuple[float, float]:
        """Evaluate how well metrics match regime conditions"""
        score = 0
        total_conditions = len(regime_def['conditions'])
        matched_conditions = 0

        for metric_name, conditions in regime_def['conditions'].items():
            if metric_name not in metrics:
                continue

            metric_value = metrics[metric_name]
            condition_met = True

            # Check min/max conditions
            if 'min' in conditions and metric_value < conditions['min']:
                condition_met = False
            if 'max' in conditions and metric_value > conditions['max']:
                condition_met = False

            if condition_met:
                matched_conditions += 1
                score += 1

        confidence = matched_conditions / total_conditions if total_conditions > 0 else 0.0
        return score, confidence

    def _calculate_secondary_regimes(self, metrics: Dict[str, float],
                                   primary_regime: str) -> List[Tuple[str, float]]:
        """Calculate secondary regime candidates"""
        secondary_regimes = []

        for regime_def in self.regime_definitions:
            if regime_def['type'] == primary_regime:
                continue

            score, confidence = self._evaluate_regime_conditions(metrics, regime_def)

            if confidence > 0.3:  # Lower threshold for secondary regimes
                secondary_regimes.append((regime_def['type'], confidence))

        # Sort by confidence and return top 3
        secondary_regimes.sort(key=lambda x: x[1], reverse=True)
        return secondary_regimes[:3]

    def process_signal(self, context: SignalContext) -> SignalResult:
        """
        Process signal using regime detection

        Args:
            context: Signal processing context

        Returns:
            Signal processing result with regime information
        """
        if not self.validate_input(context):
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={'error': 'Invalid input context'}
            )

        try:
            # Detect regime
            regime_result = self.detect_regime(context.market_data)

            return SignalResult(
                discrete_action=0,  # Regime classifier doesn't produce actions
                quality_score=50.0,  # Neutral score
                confidence=regime_result['confidence'],
                metadata={
                    'regime': regime_result,
                    'regime_type': regime_result['primary_regime'],
                    'secondary_regimes': regime_result['secondary_regimes']
                }
            )

        except Exception as e:
            logger.error(f"Error processing regime detection: {e}")
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={'error': str(e)}
            )

    def get_regime_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get regime detection history"""
        return self.regime_history[-limit:] if limit else self.regime_history

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics"""
        if not self.regime_history:
            return {}

        regime_counts = {}
        confidence_sum = 0.0

        for result in self.regime_history:
            regime = result['primary_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            confidence_sum += result['confidence']

        return {
            'total_detections': len(self.regime_history),
            'regime_counts': regime_counts,
            'average_confidence': confidence_sum / len(self.regime_history),
            'most_common_regime': max(regime_counts, key=regime_counts.get) if regime_counts else None
        }

    def process_signal(self, context: SignalContext) -> SignalResult:
        """
        Process signal using market regime classification

        Args:
            context: Signal processing context

        Returns:
            SignalResult with regime classification
        """
        try:
            # Detect current regime (use latest data)
            regime_info = self.detect_regime(context.market_data)

            return SignalResult(
                discrete_action=0,  # Regime classifier doesn't generate actions
                quality_score=regime_info['confidence'] * 100,  # Convert to 0-100 scale
                confidence=regime_info['confidence'],
                metadata={
                    'regime_type': regime_info['primary_regime'],
                    'secondary_regime': regime_info.get('secondary_regime'),
                    'classification_path': regime_info.get('classification_path', []),
                    'regime_metrics': regime_info.get('metrics', {})
                }
            )
        except Exception as e:
            logger.error(f"Error processing regime detection: {e}")
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={'error': str(e)}
            )