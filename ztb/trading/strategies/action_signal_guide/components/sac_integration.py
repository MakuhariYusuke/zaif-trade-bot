"""
SAC Integration Components for Action Signal Guide.

This module provides integration between Action Signal Guide and SAC (Soft Actor-Critic)
reinforcement learning system for enhanced signal validation and decision making.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import pandas as pd
import logging

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal

logger = logging.getLogger(__name__)


class SACSignalValidator:
    """
    Validates Action Signal Guide signals using SAC decisions.
    """

    def __init__(self):
        """Initialize SAC signal validator."""
        self.correlation_history = []
        self.validation_threshold = 0.6
        self.confidence_boost_factor = 1.3
        self.confidence_penalty_factor = 0.7

    def validate_with_sac(
        self,
        signals: List["ActionSignal"],
        sac_decisions: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> List["ActionSignal"]:
        """
        Validate signals using SAC decisions.

        Args:
            signals: Action Signal Guide signals
            sac_decisions: SAC system decisions
            market_data: Current market data

        Returns:
            Validated signals with SAC correlation scores
        """
        validated_signals = []

        for signal in signals:
            validation_result = self._validate_single_signal(signal, sac_decisions, market_data)

            if validation_result["is_valid"]:
                # Apply confidence adjustment based on SAC correlation
                correlation_score = validation_result["correlation_score"]
                original_confidence = getattr(signal, 'confidence', 0.5)

                if correlation_score >= self.validation_threshold:
                    signal.confidence = min(1.0, original_confidence * self.confidence_boost_factor)
                else:
                    signal.confidence = max(0.1, original_confidence * self.confidence_penalty_factor)

                # Add SAC validation metadata
                if not hasattr(signal, 'sac_validation'):
                    signal.sac_validation = {}

                signal.sac_validation.update(validation_result)
                validated_signals.append(signal)

        # Store validation results for learning
        self._update_correlation_history(validated_signals)

        return validated_signals

    def _validate_single_signal(
        self,
        signal: "ActionSignal",
        sac_decisions: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate a single signal against SAC decisions.

        Args:
            signal: Signal to validate
            sac_decisions: SAC decisions
            market_data: Market data

        Returns:
            Validation result dictionary
        """
        signal_action = getattr(signal, 'action', '').upper()
        sac_action = sac_decisions.get('action', '').upper()

        # Calculate action alignment
        action_alignment = 1.0 if signal_action == sac_action else 0.0

        # Calculate confidence correlation
        signal_confidence = getattr(signal, 'confidence', 0.5)
        sac_confidence = sac_decisions.get('confidence', 0.5)
        confidence_correlation = 1.0 - abs(signal_confidence - sac_confidence)

        # Calculate timing alignment
        signal_timing = getattr(signal, 'timestamp', None)
        sac_timing = sac_decisions.get('timestamp', None)

        timing_alignment = 0.5  # Default neutral
        if signal_timing and sac_timing:
            time_diff = abs((signal_timing - sac_timing).total_seconds())
            # Perfect alignment within 5 minutes
            timing_alignment = max(0.0, 1.0 - (time_diff / 300))

        # Calculate market condition alignment
        market_alignment = self._calculate_market_alignment(signal, sac_decisions, market_data)

        # Combine all correlations
        correlation_weights = {
            "action": 0.4,
            "confidence": 0.3,
            "timing": 0.2,
            "market_condition": 0.1,
        }

        overall_correlation = (
            action_alignment * correlation_weights["action"] +
            confidence_correlation * correlation_weights["confidence"] +
            timing_alignment * correlation_weights["timing"] +
            market_alignment * correlation_weights["market_condition"]
        )

        return {
            "is_valid": overall_correlation >= self.validation_threshold,
            "correlation_score": overall_correlation,
            "action_alignment": action_alignment,
            "confidence_correlation": confidence_correlation,
            "timing_alignment": timing_alignment,
            "market_alignment": market_alignment,
            "sac_action": sac_action,
            "signal_action": signal_action,
        }

    def _calculate_market_alignment(
        self,
        signal: "ActionSignal",
        sac_decisions: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> float:
        """
        Calculate alignment based on market conditions.

        Args:
            signal: Signal being validated
            sac_decisions: SAC decisions
            market_data: Market data

        Returns:
            Market alignment score (0-1)
        """
        if len(market_data) < 10:
            return 0.5

        # Get recent market metrics
        recent_data = market_data.tail(10)
        volatility = recent_data['close'].pct_change().std()
        volume_trend = recent_data['volume'].pct_change().mean()

        # SAC market state assessment
        sac_market_state = sac_decisions.get('market_state', 'neutral')

        # Signal market sensitivity
        pattern_type = getattr(signal, 'pattern_type', 'unknown')

        # Adjust alignment based on market conditions and pattern type
        alignment_score = 0.5

        if sac_market_state == 'high_volatility' and volatility > 0.03:
            if pattern_type in ['volume', 'gann', 'granville']:
                alignment_score = 0.8
            else:
                alignment_score = 0.6
        elif sac_market_state == 'low_volatility' and volatility < 0.01:
            if pattern_type in ['fibonacci', 'harmonic', 'candlestick']:
                alignment_score = 0.8
            else:
                alignment_score = 0.6
        elif sac_market_state in ['trending_bullish', 'trending_bearish']:
            if pattern_type in ['fibonacci', 'harmonic', 'gann', 'dow_theory']:
                alignment_score = 0.8
            else:
                alignment_score = 0.6

        return alignment_score

    def _update_correlation_history(self, validated_signals: List["ActionSignal"]) -> None:
        """
        Update correlation history for learning.

        Args:
            validated_signals: Recently validated signals
        """
        for signal in validated_signals:
            if hasattr(signal, 'sac_validation'):
                self.correlation_history.append({
                    "timestamp": pd.Timestamp.now(),
                    "correlation_score": signal.sac_validation["correlation_score"],
                    "pattern_type": getattr(signal, 'pattern_type', 'unknown'),
                    "signal_confidence": getattr(signal, 'confidence', 0.5),
                    "is_valid": signal.sac_validation["is_valid"],
                })

        # Keep only recent history
        if len(self.correlation_history) > 1000:
            self.correlation_history = self.correlation_history[-500:]


class SACDecisionIntegrator:
    """
    Integrates SAC decisions with Action Signal Guide for enhanced decision making.
    """

    def __init__(self):
        """Initialize SAC decision integrator."""
        self.decision_weights = {
            "signal_guide": 0.6,
            "sac_decision": 0.4,
        }
        self.confidence_threshold = 0.7
        self.integration_history = []

    def integrate_decisions(
        self,
        signal_guide_signals: List["ActionSignal"],
        sac_decisions: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Integrate Action Signal Guide and SAC decisions.

        Args:
            signal_guide_signals: Signals from Action Signal Guide
            sac_decisions: Decisions from SAC system
            market_data: Current market data

        Returns:
            Integrated decision with confidence scores
        """
        # Get best signal from Action Signal Guide
        best_signal = self._select_best_signal(signal_guide_signals)

        if not best_signal:
            # No valid signals, rely on SAC
            return {
                "action": sac_decisions.get('action', 'HOLD'),
                "confidence": sac_decisions.get('confidence', 0.5) * 0.8,  # Penalty for no signals
                "source": "sac_only",
                "reason": "no_valid_signals",
            }

        # Calculate integrated decision
        integrated_decision = self._calculate_integrated_decision(
            best_signal, sac_decisions, market_data
        )

        # Store integration result
        self.integration_history.append({
            "timestamp": pd.Timestamp.now(),
            "signal_action": getattr(best_signal, 'action', 'UNKNOWN'),
            "sac_action": sac_decisions.get('action', 'UNKNOWN'),
            "integrated_action": integrated_decision["action"],
            "signal_confidence": getattr(best_signal, 'confidence', 0.5),
            "sac_confidence": sac_decisions.get('confidence', 0.5),
            "integrated_confidence": integrated_decision["confidence"],
        })

        # Keep history manageable
        if len(self.integration_history) > 500:
            self.integration_history = self.integration_history[-250:]

        return integrated_decision

    def _select_best_signal(self, signals: List["ActionSignal"]) -> Optional["ActionSignal"]:
        """
        Select the best signal from Action Signal Guide.

        Args:
            signals: Available signals

        Returns:
            Best signal or None
        """
        if not signals:
            return None

        # Sort by confidence and recency
        sorted_signals = sorted(
            signals,
            key=lambda s: (
                getattr(s, 'confidence', 0.0),
                getattr(s, 'timestamp', pd.Timestamp.min)
            ),
            reverse=True
        )

        return sorted_signals[0] if sorted_signals else None

    def _calculate_integrated_decision(
        self,
        best_signal: "ActionSignal",
        sac_decisions: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate integrated decision from signal and SAC.

        Args:
            best_signal: Best Action Signal Guide signal
            sac_decisions: SAC decisions
            market_data: Market data

        Returns:
            Integrated decision
        """
        signal_action = getattr(best_signal, 'action', 'HOLD').upper()
        signal_confidence = getattr(best_signal, 'confidence', 0.5)

        sac_action = sac_decisions.get('action', 'HOLD').upper()
        sac_confidence = sac_decisions.get('confidence', 0.5)

        # Check for agreement
        if signal_action == sac_action:
            # Agreement - boost confidence
            integrated_confidence = min(1.0, (signal_confidence + sac_confidence) / 2 * 1.2)
            integrated_action = signal_action
            source = "agreement"
        else:
            # Disagreement - use higher confidence decision
            if signal_confidence >= sac_confidence:
                integrated_action = signal_action
                integrated_confidence = signal_confidence * 0.9  # Slight penalty for disagreement
                source = "signal_guide_dominant"
            else:
                integrated_action = sac_action
                integrated_confidence = sac_confidence * 0.9  # Slight penalty for disagreement
                source = "sac_dominant"

        # Apply market condition adjustments
        market_adjustment = self._calculate_market_adjustment(
            integrated_action, integrated_confidence, market_data
        )

        integrated_confidence = min(1.0, integrated_confidence * market_adjustment)

        return {
            "action": integrated_action,
            "confidence": integrated_confidence,
            "source": source,
            "signal_contribution": signal_confidence,
            "sac_contribution": sac_confidence,
            "market_adjustment": market_adjustment,
        }

    def _calculate_market_adjustment(
        self,
        action: str,
        confidence: float,
        market_data: pd.DataFrame
    ) -> float:
        """
        Calculate market-based confidence adjustment.

        Args:
            action: Proposed action
            confidence: Current confidence
            market_data: Market data

        Returns:
            Adjustment factor
        """
        if len(market_data) < 20:
            return 1.0

        # Calculate market momentum
        recent_returns = market_data['close'].pct_change().tail(10)
        momentum = recent_returns.mean()

        # Calculate volatility
        volatility = recent_returns.std()

        adjustment = 1.0

        # Adjust based on action and market conditions
        if action.upper() in ['BUY', 'LONG']:
            if momentum > 0.002:  # Strong upward momentum
                adjustment = 1.1
            elif momentum < -0.002:  # Strong downward momentum
                adjustment = 0.8
            if volatility > 0.03:  # High volatility
                adjustment *= 0.9
        elif action.upper() in ['SELL', 'SHORT']:
            if momentum < -0.002:  # Strong downward momentum
                adjustment = 1.1
            elif momentum > 0.002:  # Strong upward momentum
                adjustment = 0.8
            if volatility > 0.03:  # High volatility
                adjustment *= 0.9

        return adjustment


class SACPerformanceMonitor:
    """
    Monitors performance of SAC integration with Action Signal Guide.
    """

    def __init__(self):
        """Initialize SAC performance monitor."""
        self.performance_history = []
        self.accuracy_window = 50  # Last 50 decisions
        self.metrics = {
            "signal_sac_agreement_rate": 0.0,
            "integrated_decision_accuracy": 0.0,
            "signal_guide_accuracy": 0.0,
            "sac_accuracy": 0.0,
        }

    def record_decision_outcome(
        self,
        integrated_decision: Dict[str, Any],
        actual_outcome: float,
        market_data: pd.DataFrame
    ) -> None:
        """
        Record the outcome of an integrated decision.

        Args:
            integrated_decision: The integrated decision made
            actual_outcome: Actual profit/loss outcome
            market_data: Market data at decision time
        """
        decision_record = {
            "timestamp": pd.Timestamp.now(),
            "action": integrated_decision["action"],
            "confidence": integrated_decision["confidence"],
            "source": integrated_decision.get("source", "unknown"),
            "signal_contribution": integrated_decision.get("signal_contribution", 0.0),
            "sac_contribution": integrated_decision.get("sac_contribution", 0.0),
            "outcome": actual_outcome,
            "market_volatility": market_data['close'].pct_change().tail(20).std() if len(market_data) >= 20 else 0.0,
        }

        self.performance_history.append(decision_record)

        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

        # Update metrics
        self._update_metrics()

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return self.metrics.copy()

    def _update_metrics(self) -> None:
        """Update performance metrics based on recent history."""
        if len(self.performance_history) < 10:
            return

        recent_decisions = self.performance_history[-self.accuracy_window:]

        # Calculate agreement rate (when both signal and SAC agree)
        agreement_decisions = [
            d for d in recent_decisions
            if d["source"] == "agreement"
        ]
        self.metrics["signal_sac_agreement_rate"] = (
            len(agreement_decisions) / len(recent_decisions)
            if recent_decisions else 0.0
        )

        # Calculate integrated decision accuracy
        profitable_decisions = sum(1 for d in recent_decisions if d["outcome"] > 0)
        self.metrics["integrated_decision_accuracy"] = profitable_decisions / len(recent_decisions)

        # Calculate component accuracies
        signal_guide_decisions = [
            d for d in recent_decisions
            if "signal_guide" in d["source"]
        ]
        if signal_guide_decisions:
            signal_profitable = sum(1 for d in signal_guide_decisions if d["outcome"] > 0)
            self.metrics["signal_guide_accuracy"] = signal_profitable / len(signal_guide_decisions)

        sac_decisions = [
            d for d in recent_decisions
            if "sac" in d["source"]
        ]
        if sac_decisions:
            sac_profitable = sum(1 for d in sac_decisions if d["outcome"] > 0)
            self.metrics["sac_accuracy"] = sac_profitable / len(sac_decisions)
