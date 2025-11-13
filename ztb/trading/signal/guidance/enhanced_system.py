"""
Enhanced Signal Guidance System

Advanced signal processing system that integrates market regime adaptation
with quality scoring and strategic guidance for optimal trading decisions.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ztb.trading.signal.common.base_classes import BaseSignalProcessor, SignalContext, SignalResult
from ztb.trading.signal.regime.classifier import MarketRegimeClassifier, RegimeType
from ztb.trading.signal.quality.scorer import SignalQualityScorer
from ztb.trading.signal.common.utilities import calculate_confidence_score, normalize_weights
from ztb.trading.signal.common.metrics import calculate_composite_score
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EnhancedSignalGuidanceSystem(BaseSignalProcessor):
    """
    Enhanced signal guidance system with regime-adaptive processing

    Integrates market regime classification with signal quality scoring
    to provide context-aware trading guidance and decision optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Initialize core components
        regime_config = self.config.get('regime_config')
        self.regime_classifier = MarketRegimeClassifier(regime_config)

        quality_config = self.config.get('quality_config')
        self.quality_scorer = SignalQualityScorer(quality_config)

        # Initialize regime adaptation parameters
        self.regime_adaptation_params = self._initialize_regime_adaptation()

        # Performance tracking
        self.performance_history = []
        self.regime_performance = {}

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'regime_config': {
                'lookback_periods': {'short': 20, 'medium': 50, 'long': 100},
                'confidence_threshold': 0.6,
                'max_history': 1000
            },
            'quality_config': {
                'weights': {
                    'trend': 0.25,
                    'momentum': 0.20,
                    'volatility': 0.15,
                    'volume': 0.15,
                    'regime': 0.25
                },
                'thresholds': {
                    'strong_buy': 80,
                    'buy': 65,
                    'hold': 50,
                    'sell': 35,
                    'strong_sell': 20
                }
            },
            'adaptation_config': {
                'learning_rate': 0.1,
                'performance_window': 100,
                'regime_memory': 50
            }
        }

    def _initialize_regime_adaptation(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regime-specific adaptation parameters"""
        return {
            # SELL特化レジーム（最高優先度）
            RegimeType.SELL_BREAKDOWN: {
                'signal_bias': -2.0,  # Strong SELL bias
                'confidence_multiplier': 1.5,
                'threshold_adjustment': -10,  # Lower threshold for SELL signals
                'description': 'Strong breakdown - prioritize SELL signals'
            },
            RegimeType.SELL_DIVERGENCE: {
                'signal_bias': -1.5,
                'confidence_multiplier': 1.3,
                'threshold_adjustment': -8,
                'description': 'Bearish divergence - favor SELL signals'
            },
            RegimeType.SELL_MOMENTUM_WEAK: {
                'signal_bias': -1.2,
                'confidence_multiplier': 1.2,
                'threshold_adjustment': -5,
                'description': 'Weakening momentum - reinforce SELL'
            },
            RegimeType.SELL_VOLUME_SURGE: {
                'signal_bias': -1.8,
                'confidence_multiplier': 1.4,
                'threshold_adjustment': -12,
                'description': 'Volume surge in downtrend - SELL confirmation'
            },

            # Bullトレンドレジーム
            RegimeType.STRONG_BULL_TREND: {
                'signal_bias': 1.5,
                'confidence_multiplier': 1.3,
                'threshold_adjustment': 8,
                'description': 'Strong bull trend - favor BUY signals'
            },
            RegimeType.MODERATE_BULL_TREND: {
                'signal_bias': 1.0,
                'confidence_multiplier': 1.1,
                'threshold_adjustment': 5,
                'description': 'Moderate bull trend - support BUY signals'
            },
            RegimeType.WEAK_BULL_TREND: {
                'signal_bias': 0.5,
                'confidence_multiplier': 1.0,
                'threshold_adjustment': 2,
                'description': 'Weak bull trend - neutral to slightly bullish'
            },

            # Bearトレンドレジーム
            RegimeType.STRONG_BEAR_TREND: {
                'signal_bias': -1.5,
                'confidence_multiplier': 1.3,
                'threshold_adjustment': -8,
                'description': 'Strong bear trend - favor SELL signals'
            },
            RegimeType.MODERATE_BEAR_TREND: {
                'signal_bias': -1.0,
                'confidence_multiplier': 1.1,
                'threshold_adjustment': -5,
                'description': 'Moderate bear trend - support SELL signals'
            },
            RegimeType.WEAK_BEAR_TREND: {
                'signal_bias': -0.5,
                'confidence_multiplier': 1.0,
                'threshold_adjustment': -2,
                'description': 'Weak bear trend - neutral to slightly bearish'
            },

            # レンジ相場レジーム
            RegimeType.HIGH_VOLATILITY_RANGE: {
                'signal_bias': 0.0,
                'confidence_multiplier': 0.8,
                'threshold_adjustment': 0,
                'description': 'High volatility range - reduce signal strength'
            },
            RegimeType.MODERATE_VOLATILITY_RANGE: {
                'signal_bias': 0.0,
                'confidence_multiplier': 0.9,
                'threshold_adjustment': 0,
                'description': 'Moderate range - neutral processing'
            },
            RegimeType.LOW_VOLATILITY_RANGE: {
                'signal_bias': 0.0,
                'confidence_multiplier': 1.0,
                'threshold_adjustment': 0,
                'description': 'Low volatility range - standard processing'
            },

            # 特殊条件レジーム
            RegimeType.EXTREME_VOLATILITY: {
                'signal_bias': 0.0,
                'confidence_multiplier': 0.7,
                'threshold_adjustment': 0,
                'description': 'Extreme volatility - conservative approach'
            },
            RegimeType.CONSOLIDATION: {
                'signal_bias': 0.0,
                'confidence_multiplier': 0.9,
                'threshold_adjustment': 0,
                'description': 'Consolidation - wait for clearer signals'
            },
            RegimeType.BREAKOUT_SETUP: {
                'signal_bias': 0.3,
                'confidence_multiplier': 1.2,
                'threshold_adjustment': 3,
                'description': 'Breakout setup - slightly bullish bias'
            },
            RegimeType.BREAKDOWN_SETUP: {
                'signal_bias': -0.3,
                'confidence_multiplier': 1.2,
                'threshold_adjustment': -3,
                'description': 'Breakdown setup - slightly bearish bias'
            }
        }

    def process_signal(self, context: SignalContext) -> SignalResult:
        """
        Process signal with regime-adaptive guidance

        Args:
            context: Signal processing context

        Returns:
            Enhanced signal result with regime adaptation
        """
        if not self.validate_input(context):
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={'error': 'Invalid input context'}
            )

        try:
            # Step 1: Detect current market regime
            regime_result = self.regime_classifier.process_signal(context)
            current_regime = regime_result.metadata.get('regime_type', RegimeType.CONSOLIDATION)

            # Step 2: Get base quality score
            quality_result = self.quality_scorer.process_signal(context)

            # Step 3: Apply regime adaptation
            adapted_result = self._apply_regime_adaptation(
                quality_result, current_regime, regime_result
            )

            # Step 4: Generate strategic guidance
            guidance = self._generate_strategic_guidance(
                adapted_result, current_regime, context
            )

            # Step 5: Update performance tracking
            self._update_performance_tracking(adapted_result, current_regime)

            # Create enhanced result
            enhanced_result = SignalResult(
                discrete_action=adapted_result.discrete_action,
                quality_score=adapted_result.quality_score,
                confidence=adapted_result.confidence,
                metadata={
                    'regime': current_regime,
                    'regime_confidence': regime_result.confidence,
                    'base_quality_score': quality_result.quality_score,
                    'regime_adaptation': adapted_result.metadata,
                    'strategic_guidance': guidance,
                    'performance_metrics': self._get_performance_metrics(current_regime)
                }
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Error in enhanced signal processing: {e}")
            return SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.5,
                metadata={'error': str(e)}
            )

    def _apply_regime_adaptation(self, quality_result: SignalResult,
                               current_regime: str, regime_result: SignalResult) -> SignalResult:
        """Apply regime-specific signal adaptation"""
        adaptation_params = self.regime_adaptation_params.get(
            current_regime, self.regime_adaptation_params[RegimeType.CONSOLIDATION]
        )

        # Get base scores
        base_score = quality_result.quality_score
        base_confidence = quality_result.confidence

        # Apply regime bias
        regime_bias = adaptation_params['signal_bias']
        adapted_score = base_score + (regime_bias * 10)  # Convert bias to score adjustment

        # Clamp score to valid range
        adapted_score = max(0, min(100, adapted_score))

        # Apply confidence multiplier
        confidence_multiplier = adaptation_params['confidence_multiplier']
        adapted_confidence = min(1.0, base_confidence * confidence_multiplier)

        # Apply threshold adjustment for action determination
        threshold_adjustment = adaptation_params['threshold_adjustment']
        thresholds = self.config['quality_config']['thresholds'].copy()

        for key in thresholds:
            thresholds[key] += threshold_adjustment
            thresholds[key] = max(0, min(100, thresholds[key]))

        # Determine adapted action
        adapted_action = self._determine_adapted_action(adapted_score, thresholds)

        return SignalResult(
            discrete_action=adapted_action,
            quality_score=adapted_score,
            confidence=adapted_confidence,
            metadata={
                'adaptation_params': adaptation_params,
                'original_score': base_score,
                'original_confidence': base_confidence,
                'thresholds_used': thresholds,
                'regime_bias_applied': regime_bias
            }
        )

    def _determine_adapted_action(self, score: float, thresholds: Dict[str, float]) -> int:
        """Determine trading action based on adapted score and thresholds"""
        # Remove 'hold' from thresholds if present (not used in action determination)
        action_thresholds = {k: v for k, v in thresholds.items() if k != 'hold'}

        if score >= action_thresholds.get('strong_buy', 80):
            return 2  # Strong BUY
        elif score >= action_thresholds.get('buy', 65):
            return 1  # BUY
        elif score < action_thresholds.get('strong_sell', 20):
            return -2  # Strong SELL
        elif score < action_thresholds.get('sell', 35):
            return -1  # SELL
        else:
            return 0  # HOLD

    def _generate_strategic_guidance(self, adapted_result: SignalResult,
                                   current_regime: str, context: SignalContext) -> Dict[str, Any]:
        """Generate strategic guidance based on regime and signal analysis"""
        guidance = {
            'primary_action': self._action_to_string(adapted_result.discrete_action),
            'regime_context': self._get_regime_context(current_regime),
            'confidence_level': self._confidence_to_level(adapted_result.confidence),
            'risk_assessment': self._assess_risk(current_regime, adapted_result),
            'position_sizing': self._recommend_position_sizing(current_regime, adapted_result),
            'time_horizon': self._recommend_time_horizon(current_regime),
            'stop_loss_guidance': self._generate_stop_loss_guidance(current_regime, context),
            'take_profit_guidance': self._generate_take_profit_guidance(current_regime, context)
        }

        return guidance

    def _action_to_string(self, action: int) -> str:
        """Convert discrete action to string"""
        action_map = {
            2: 'STRONG_BUY',
            1: 'BUY',
            0: 'HOLD',
            -1: 'SELL',
            -2: 'STRONG_SELL'
        }
        return action_map.get(action, 'HOLD')

    def _get_regime_context(self, regime: str) -> str:
        """Get human-readable regime context"""
        regime_contexts = {
            RegimeType.SELL_BREAKDOWN: 'Strong breakdown pattern - SELL priority',
            RegimeType.SELL_DIVERGENCE: 'Bearish divergence detected',
            RegimeType.SELL_MOMENTUM_WEAK: 'Weakening momentum in downtrend',
            RegimeType.SELL_VOLUME_SURGE: 'Volume surge confirming downtrend',
            RegimeType.STRONG_BULL_TREND: 'Strong upward momentum',
            RegimeType.MODERATE_BULL_TREND: 'Moderate upward trend',
            RegimeType.WEAK_BULL_TREND: 'Weak upward movement',
            RegimeType.STRONG_BEAR_TREND: 'Strong downward momentum',
            RegimeType.MODERATE_BEAR_TREND: 'Moderate downward trend',
            RegimeType.WEAK_BEAR_TREND: 'Weak downward movement',
            RegimeType.HIGH_VOLATILITY_RANGE: 'High volatility sideways movement',
            RegimeType.MODERATE_VOLATILITY_RANGE: 'Moderate volatility consolidation',
            RegimeType.LOW_VOLATILITY_RANGE: 'Low volatility tight range',
            RegimeType.EXTREME_VOLATILITY: 'Extreme market volatility',
            RegimeType.CONSOLIDATION: 'Tight consolidation phase',
            RegimeType.BREAKOUT_SETUP: 'Potential breakout from consolidation',
            RegimeType.BREAKDOWN_SETUP: 'Potential breakdown from consolidation'
        }
        return regime_contexts.get(regime, 'Unknown regime')

    def _confidence_to_level(self, confidence: float) -> str:
        """Convert confidence to descriptive level"""
        if confidence >= 0.8:
            return 'VERY_HIGH'
        elif confidence >= 0.6:
            return 'HIGH'
        elif confidence >= 0.4:
            return 'MODERATE'
        elif confidence >= 0.2:
            return 'LOW'
        else:
            return 'VERY_LOW'

    def _assess_risk(self, regime: str, result: SignalResult) -> str:
        """Assess risk level based on regime and signal"""
        high_risk_regimes = [
            RegimeType.EXTREME_VOLATILITY,
            RegimeType.SELL_BREAKDOWN,
            RegimeType.STRONG_BEAR_TREND
        ]

        if regime in high_risk_regimes:
            return 'HIGH'
        elif result.confidence < 0.4:
            return 'MODERATE'
        else:
            return 'LOW'

    def _recommend_position_sizing(self, regime: str, result: SignalResult) -> str:
        """Recommend position sizing based on regime and confidence"""
        if result.confidence >= 0.8:
            base_size = 'LARGE'
        elif result.confidence >= 0.6:
            base_size = 'MEDIUM'
        else:
            base_size = 'SMALL'

        # Adjust for regime
        regime_multipliers = {
            RegimeType.EXTREME_VOLATILITY: 'REDUCED',
            RegimeType.CONSOLIDATION: 'REDUCED',
            RegimeType.SELL_BREAKDOWN: 'INCREASED',
            RegimeType.STRONG_BULL_TREND: 'INCREASED'
        }

        if regime in regime_multipliers:
            return f"{base_size}_{regime_multipliers[regime]}"
        else:
            return base_size

    def _recommend_time_horizon(self, regime: str) -> str:
        """Recommend trading time horizon based on regime"""
        short_term_regimes = [
            RegimeType.EXTREME_VOLATILITY,
            RegimeType.HIGH_VOLATILITY_RANGE,
            RegimeType.SELL_BREAKDOWN,
            RegimeType.SELL_VOLUME_SURGE
        ]

        if regime in short_term_regimes:
            return 'SHORT_TERM'
        else:
            return 'MEDIUM_TERM'

    def _generate_stop_loss_guidance(self, regime: str, context: SignalContext) -> Dict[str, Any]:
        """Generate stop loss guidance"""
        if len(context.market_data) < 2:
            return {'type': 'PERCENTAGE', 'value': 0.02}

        current_price = context.market_data['close'].iloc[-1]
        recent_high = context.market_data['high'].rolling(20).max().iloc[-1]
        recent_low = context.market_data['low'].rolling(20).min().iloc[-1]

        # Regime-specific stop loss logic
        if regime in [RegimeType.SELL_BREAKDOWN, RegimeType.STRONG_BEAR_TREND]:
            # Tighter stops in strong trends
            stop_distance = abs(current_price - recent_low) * 0.5
            return {
                'type': 'PRICE_LEVEL',
                'value': current_price + stop_distance,
                'reason': 'Tight stop in strong downtrend'
            }
        elif regime == RegimeType.EXTREME_VOLATILITY:
            # Wider stops in high volatility
            return {
                'type': 'PERCENTAGE',
                'value': 0.05,
                'reason': 'Wide stop for volatility protection'
            }
        else:
            return {
                'type': 'PERCENTAGE',
                'value': 0.03,
                'reason': 'Standard stop loss'
            }

    def _generate_take_profit_guidance(self, regime: str, context: SignalContext) -> Dict[str, Any]:
        """Generate take profit guidance"""
        if len(context.market_data) < 2:
            return {'type': 'PERCENTAGE', 'value': 0.05}

        # Regime-specific take profit logic
        if regime in [RegimeType.STRONG_BULL_TREND, RegimeType.SELL_BREAKDOWN]:
            # Larger targets in strong trends
            return {
                'type': 'PERCENTAGE',
                'value': 0.08,
                'reason': 'Large target in strong trend'
            }
        elif regime in [RegimeType.CONSOLIDATION, RegimeType.LOW_VOLATILITY_RANGE]:
            # Smaller targets in range/consolidation
            return {
                'type': 'PERCENTAGE',
                'value': 0.03,
                'reason': 'Conservative target in consolidation'
            }
        else:
            return {
                'type': 'PERCENTAGE',
                'value': 0.05,
                'reason': 'Standard take profit'
            }

    def _update_performance_tracking(self, result: SignalResult, regime: str):
        """Update performance tracking for regime adaptation learning"""
        performance_entry = {
            'timestamp': pd.Timestamp.now(),
            'regime': regime,
            'action': result.discrete_action,
            'confidence': result.confidence,
            'quality_score': result.quality_score
        }

        self.performance_history.append(performance_entry)

        # Keep only recent history
        max_history = self.config['adaptation_config']['performance_window']
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]

        # Update regime-specific performance
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []

        self.regime_performance[regime].append(performance_entry)

        # Keep regime memory
        regime_memory = self.config['adaptation_config']['regime_memory']
        if len(self.regime_performance[regime]) > regime_memory:
            self.regime_performance[regime] = self.regime_performance[regime][-regime_memory:]

    def _get_performance_metrics(self, regime: str) -> Dict[str, Any]:
        """Get performance metrics for current regime"""
        if regime not in self.regime_performance or not self.regime_performance[regime]:
            return {}

        regime_history = self.regime_performance[regime]
        recent_entries = regime_history[-20:]  # Last 20 signals

        avg_confidence = np.mean([entry['confidence'] for entry in recent_entries])
        avg_quality = np.mean([entry['quality_score'] for entry in recent_entries])

        return {
            'regime_signal_count': len(regime_history),
            'recent_avg_confidence': float(avg_confidence),
            'recent_avg_quality': float(avg_quality),
            'regime_adaptation_active': True
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'regime_classifier_status': self.regime_classifier.get_regime_statistics(),
            'performance_history_length': len(self.performance_history),
            'active_regimes': list(self.regime_performance.keys()),
            'total_regime_adaptations': sum(len(history) for history in self.regime_performance.values()),
            'system_config': self.config
        }