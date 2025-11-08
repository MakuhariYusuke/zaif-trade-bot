"""
AdvancedSignalAggregator Component.

Implements sophisticated signal aggregation strategies for Action Signal Guide.
Provides pattern correlation modeling and SAC action pattern coordination.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from ztb.utils.logging_utils import get_logger

from ..types import ActionSignal, SignalList


class AdvancedSignalAggregator:
    """
    Advanced signal aggregation with pattern correlation and SAC coordination.

    This class provides:
    - Pattern correlation modeling
    - SAC action pattern coordination
    - Time series consistency validation
    - Adaptive aggregation strategies
    """

    def __init__(self):
        self.logger = get_logger("ztb.trading.strategies.advanced_signal_aggregator")
        
        # Pattern correlation matrix
        self.pattern_correlations: Dict[Tuple[str, str], float] = {}
        
        # SAC action pattern learning
        self.sac_action_patterns: Dict[str, List[float]] = {}
        
        # Aggregation strategies
        self.aggregation_strategies = {
            'weighted_average': self._weighted_average_aggregation,
            'correlation_aware': self._correlation_aware_aggregation,
            'sac_coordinated': self._sac_coordinated_aggregation,
            'time_series_consistent': self._time_series_consistent_aggregation
        }

    def aggregate_signals(
        self,
        signals: SignalList,
        market_data: pd.DataFrame,
        sac_context: Optional[Dict[str, Any]] = None,
        strategy: str = 'correlation_aware'
    ) -> ActionSignal:
        """
        Aggregate signals using advanced strategies.

        Args:
            signals: List of ActionSignal objects
            market_data: Current market data
            sac_context: SAC system context (actions, rewards, etc.)
            strategy: Aggregation strategy to use

        Returns:
            Aggregated ActionSignal
        """
        if not signals:
            return self._create_null_signal()

        if strategy not in self.aggregation_strategies:
            self.logger.warning(f"Unknown strategy {strategy}, using weighted_average")
            strategy = 'weighted_average'

        aggregator = self.aggregation_strategies[strategy]
        return aggregator(signals, market_data, sac_context)

    def _weighted_average_aggregation(
        self,
        signals: SignalList,
        market_data: pd.DataFrame,
        sac_context: Optional[Dict[str, Any]] = None
    ) -> ActionSignal:
        """Standard weighted average aggregation."""
        total_weight = sum(s.strength * s.confidence for s in signals)
        
        if total_weight == 0:
            return self._create_null_signal()

        weighted_direction = sum(
            s.direction * s.strength * s.confidence for s in signals
        ) / total_weight
        
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])

        return ActionSignal(
            direction=1 if weighted_direction > 0 else -1,
            strength=min(abs(weighted_direction), 1.0),
            confidence=min(avg_confidence, 1.0),
            pattern_type='aggregated',
            timestamp=signals[0].timestamp if signals else None
        )

    def _correlation_aware_aggregation(
        self,
        signals: SignalList,
        market_data: pd.DataFrame,
        sac_context: Optional[Dict[str, Any]] = None
    ) -> ActionSignal:
        """Correlation-aware aggregation considering pattern relationships."""
        if len(signals) < 2:
            return self._weighted_average_aggregation(signals, market_data, sac_context)

        # Build correlation matrix for current signals
        correlation_matrix = self._build_signal_correlation_matrix(signals)
        
        # Adjust weights based on correlations
        adjusted_signals = []
        for i, signal in enumerate(signals):
            correlation_factor = self._calculate_correlation_factor(
                signal, signals, correlation_matrix, i
            )
            
            adjusted_signal = ActionSignal(
                direction=signal.direction,
                strength=signal.strength * correlation_factor,
                confidence=signal.confidence * correlation_factor,
                pattern_type=signal.pattern_type,
                timestamp=signal.timestamp
            )
            adjusted_signals.append(adjusted_signal)

        return self._weighted_average_aggregation(adjusted_signals, market_data, sac_context)

    def _sac_coordinated_aggregation(
        self,
        signals: SignalList,
        market_data: pd.DataFrame,
        sac_context: Optional[Dict[str, Any]] = None
    ) -> ActionSignal:
        """SAC-coordinated aggregation based on learned action patterns."""
        if not sac_context:
            return self._correlation_aware_aggregation(signals, market_data, sac_context)

        # Extract SAC action patterns
        recent_actions = sac_context.get('recent_actions', [])
        action_rewards = sac_context.get('action_rewards', [])

        if not recent_actions or not action_rewards:
            return self._correlation_aware_aggregation(signals, market_data, sac_context)

        # Learn action pattern preferences
        action_pattern_scores = self._learn_action_patterns(recent_actions, action_rewards)
        
        # Adjust signal weights based on SAC preferences
        adjusted_signals = []
        for signal in signals:
            sac_adjustment = self._calculate_sac_adjustment(signal, action_pattern_scores)
            
            adjusted_signal = ActionSignal(
                direction=signal.direction,
                strength=signal.strength * sac_adjustment,
                confidence=signal.confidence * sac_adjustment,
                pattern_type=signal.pattern_type,
                timestamp=signal.timestamp
            )
            adjusted_signals.append(adjusted_signal)

        return self._weighted_average_aggregation(adjusted_signals, market_data, sac_context)

    def _time_series_consistent_aggregation(
        self,
        signals: SignalList,
        market_data: pd.DataFrame,
        sac_context: Optional[Dict[str, Any]] = None
    ) -> ActionSignal:
        """Time series consistent aggregation with trend validation."""
        if len(market_data) < 10:
            return self._correlation_aware_aggregation(signals, market_data, sac_context)

        # Analyze time series consistency
        trend_direction = self._analyze_market_trend(market_data)
        consistency_scores = []

        for signal in signals:
            consistency = self._calculate_signal_consistency(signal, market_data, trend_direction)
            consistency_scores.append(consistency)

        # Adjust signals by consistency
        adjusted_signals = []
        for signal, consistency in zip(signals, consistency_scores):
            adjusted_signal = ActionSignal(
                direction=signal.direction,
                strength=signal.strength * consistency,
                confidence=signal.confidence * consistency,
                pattern_type=signal.pattern_type,
                timestamp=signal.timestamp
            )
            adjusted_signals.append(adjusted_signal)

        return self._weighted_average_aggregation(adjusted_signals, market_data, sac_context)

    def _build_signal_correlation_matrix(self, signals: SignalList) -> np.ndarray:
        """Build correlation matrix for signals."""
        n_signals = len(signals)
        correlation_matrix = np.eye(n_signals)  # Identity matrix as base
        
        for i in range(n_signals):
            for j in range(i+1, n_signals):
                corr = self._calculate_signal_correlation(signals[i], signals[j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
                
        return correlation_matrix

    def _calculate_signal_correlation(self, signal1: ActionSignal, signal2: ActionSignal) -> float:
        """Calculate correlation between two signals."""
        # Simple correlation based on direction and pattern type
        direction_corr = 1.0 if signal1.direction == signal2.direction else -1.0
        
        # Pattern type similarity
        pattern_corr = 1.0 if signal1.pattern_type == signal2.pattern_type else 0.5
        
        return (direction_corr + pattern_corr) / 2.0

    def _calculate_correlation_factor(
        self,
        signal: ActionSignal,
        all_signals: SignalList,
        correlation_matrix: np.ndarray,
        signal_index: int
    ) -> float:
        """Calculate correlation-based adjustment factor."""
        correlations = correlation_matrix[signal_index]
        avg_correlation = np.mean(correlations)
        
        # Boost highly correlated signals, penalize conflicting ones
        factor = 1.0 + (avg_correlation * 0.2)  # ±20% adjustment
        return max(0.5, min(1.5, factor))

    def _learn_action_patterns(self, actions: List[int], rewards: List[float]) -> Dict[str, float]:
        """Learn SAC action pattern preferences."""
        pattern_scores = {}
        
        # Simple pattern learning: action -> average reward
        unique_actions = set(actions)
        for action in unique_actions:
            action_rewards = [r for a, r in zip(actions, rewards) if a == action]
            if action_rewards:
                pattern_scores[str(action)] = np.mean(action_rewards)
                
        return pattern_scores

    def _calculate_sac_adjustment(self, signal: ActionSignal, action_scores: Dict[str, float]) -> float:
        """Calculate SAC-based adjustment for signal."""
        signal_action = str(signal.direction)  # Assume direction maps to action
        base_score = action_scores.get(signal_action, 0.0)
        
        # Normalize to adjustment factor
        adjustment = 1.0 + (base_score * 0.1)  # ±10% based on SAC performance
        return max(0.8, min(1.2, adjustment))

    def _analyze_market_trend(self, market_data: pd.DataFrame) -> int:
        """Analyze overall market trend direction."""
        if 'close' not in market_data.columns:
            return 0
            
        closes = market_data['close'].values
        if len(closes) < 5:
            return 0
            
        # Simple trend: compare recent vs older prices
        recent_avg = np.mean(closes[-5:])
        older_avg = np.mean(closes[:-5]) if len(closes) > 5 else recent_avg
        
        if recent_avg > older_avg * 1.001:  # 0.1% threshold
            return 1
        elif recent_avg < older_avg * 0.999:
            return -1
        else:
            return 0

    def _calculate_signal_consistency(
        self,
        signal: ActionSignal,
        market_data: pd.DataFrame,
        trend_direction: int
    ) -> float:
        """Calculate how consistent signal is with market trend."""
        if trend_direction == 0:
            return 1.0  # Neutral trend, full consistency
            
        # Signal aligns with trend
        if signal.direction == trend_direction:
            return 1.0
        else:
            return 0.7  # Partial consistency for conflicting signals

    def _create_null_signal(self) -> ActionSignal:
        """Create a null/empty signal."""
        return ActionSignal(
            direction=0,
            strength=0.0,
            confidence=0.0,
            pattern_type='null',
            timestamp=None
        )