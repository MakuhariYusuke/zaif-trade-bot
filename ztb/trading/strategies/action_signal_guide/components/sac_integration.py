"""
SAC Integration Components for Action Signal Guide.

This module provides integration between Action Signal Guide and SAC (Soft Actor-Critic)
reinforcement learning system for enhanced signal validation and decision making.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

from .history_helpers import append_with_compaction

if TYPE_CHECKING:
    from ..action_signal_guide import ActionSignal

logger = logging.getLogger(__name__)


class SACDecisionPayload(TypedDict, total=False):
    action: str
    confidence: float
    timestamp: pd.Timestamp
    market_state: str


class SACValidationResult(TypedDict):
    is_valid: bool
    correlation_score: float
    action_alignment: float
    confidence_correlation: float
    timing_alignment: float
    market_alignment: float
    sac_action: str
    signal_action: str


class CorrelationHistoryEntry(TypedDict):
    timestamp: pd.Timestamp
    correlation_score: float
    pattern_type: str
    signal_confidence: float
    is_valid: bool


class IntegratedDecision(TypedDict, total=False):
    action: str
    confidence: float
    source: str
    reason: str
    signal_contribution: float
    sac_contribution: float
    market_adjustment: float


class IntegrationHistoryEntry(TypedDict):
    timestamp: pd.Timestamp
    signal_action: str
    sac_action: str
    integrated_action: str
    signal_confidence: float
    sac_confidence: float
    integrated_confidence: float


class PerformanceDecisionRecord(TypedDict):
    timestamp: pd.Timestamp
    action: str
    confidence: float
    source: str
    signal_contribution: float
    sac_contribution: float
    outcome: float
    market_volatility: float


class SACPerformanceMetrics(TypedDict):
    signal_sac_agreement_rate: float
    integrated_decision_accuracy: float
    signal_guide_accuracy: float
    sac_accuracy: float


def _coerce_float(value: object, default: float) -> float:
    """Coerce unknown numeric payloads to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_action(value: object, default: str = "") -> str:
    """Normalize action-like values (enum/string/object) into uppercase label."""
    candidate = getattr(value, "value", value)
    if candidate is None:
        return default
    return str(candidate).upper()


def _coerce_timestamp(value: object, default: pd.Timestamp) -> pd.Timestamp:
    """Coerce timestamp-like values to pandas Timestamp for stable ordering."""
    if isinstance(value, pd.Timestamp):
        return value
    try:
        converted = pd.to_datetime(value)
        if isinstance(converted, pd.Timestamp):
            return converted
    except Exception:
        pass
    return default


class SACSignalValidator:
    """
    Validates Action Signal Guide signals using SAC decisions.
    """

    def __init__(self) -> None:
        """Initialize SAC signal validator."""
        self.correlation_history: list[CorrelationHistoryEntry] = []
        self.validation_threshold = 0.6
        self.confidence_boost_factor = 1.3
        self.confidence_penalty_factor = 0.7

    def validate_with_sac(
        self,
        signals: list["ActionSignal"],
        sac_decisions: SACDecisionPayload,
        market_data: pd.DataFrame,
    ) -> list["ActionSignal"]:
        """
        Validate signals using SAC decisions.

        Args:
            signals: Action Signal Guide signals
            sac_decisions: SAC system decisions
            market_data: Current market data

        Returns:
            Validated signals with SAC correlation scores
        """
        validated_signals: list["ActionSignal"] = []

        for signal in signals:
            validation_result = self._validate_single_signal(
                signal,
                sac_decisions,
                market_data,
            )

            if validation_result["is_valid"]:
                # Apply confidence adjustment based on SAC correlation
                correlation_score = validation_result["correlation_score"]
                original_confidence = _coerce_float(getattr(signal, "confidence", 0.5), 0.5)

                if correlation_score >= self.validation_threshold:
                    signal.confidence = min(
                        1.0,
                        original_confidence * self.confidence_boost_factor,
                    )
                else:
                    signal.confidence = max(
                        0.1,
                        original_confidence * self.confidence_penalty_factor,
                    )

                # Add SAC validation metadata
                if not hasattr(signal, "sac_validation"):
                    signal.sac_validation = {}

                signal.sac_validation.update(validation_result)
                validated_signals.append(signal)

        # Store validation results for learning
        self._update_correlation_history(validated_signals)

        return validated_signals

    def _validate_single_signal(
        self,
        signal: "ActionSignal",
        sac_decisions: SACDecisionPayload,
        market_data: pd.DataFrame,
    ) -> SACValidationResult:
        """
        Validate a single signal against SAC decisions.

        Args:
            signal: Signal to validate
            sac_decisions: SAC decisions
            market_data: Market data

        Returns:
            Validation result dictionary
        """
        signal_action = _normalize_action(getattr(signal, "action", ""))
        sac_action = _normalize_action(sac_decisions.get("action", ""))

        # Calculate action alignment
        action_alignment = 1.0 if signal_action == sac_action else 0.0

        # Calculate confidence correlation
        signal_confidence = _coerce_float(getattr(signal, "confidence", 0.5), 0.5)
        sac_confidence = _coerce_float(sac_decisions.get("confidence", 0.5), 0.5)
        confidence_correlation = max(0.0, 1.0 - abs(signal_confidence - sac_confidence))

        # Calculate timing alignment
        signal_timing = getattr(signal, "timestamp", None)
        sac_timing = sac_decisions.get("timestamp", None)

        timing_alignment = 0.5  # Default neutral
        if isinstance(signal_timing, pd.Timestamp) and isinstance(sac_timing, pd.Timestamp):
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
            action_alignment * correlation_weights["action"]
            + confidence_correlation * correlation_weights["confidence"]
            + timing_alignment * correlation_weights["timing"]
            + market_alignment * correlation_weights["market_condition"]
        )

        return {
            "is_valid": overall_correlation >= self.validation_threshold,
            "correlation_score": float(overall_correlation),
            "action_alignment": float(action_alignment),
            "confidence_correlation": float(confidence_correlation),
            "timing_alignment": float(timing_alignment),
            "market_alignment": float(market_alignment),
            "sac_action": sac_action,
            "signal_action": signal_action,
        }

    def _calculate_market_alignment(
        self,
        signal: "ActionSignal",
        sac_decisions: SACDecisionPayload,
        market_data: pd.DataFrame,
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
        if len(market_data) < 10 or "close" not in market_data.columns:
            return 0.5

        # Get recent market metrics
        recent_data = market_data.tail(10)
        volatility = _coerce_float(recent_data["close"].pct_change().std(), 0.0)

        # SAC market state assessment
        sac_market_state = str(sac_decisions.get("market_state", "neutral")).lower()

        # Signal market sensitivity
        pattern_type = str(getattr(signal, "pattern_type", "unknown")).lower()

        # Adjust alignment based on market conditions and pattern type
        alignment_score = 0.5

        if sac_market_state == "high_volatility" and volatility > 0.03:
            if pattern_type in ["volume", "gann", "granville"]:
                alignment_score = 0.8
            else:
                alignment_score = 0.6
        elif sac_market_state == "low_volatility" and volatility < 0.01:
            if pattern_type in ["fibonacci", "harmonic", "candlestick"]:
                alignment_score = 0.8
            else:
                alignment_score = 0.6
        elif sac_market_state in ["trending_bullish", "trending_bearish"]:
            if pattern_type in ["fibonacci", "harmonic", "gann", "dow_theory"]:
                alignment_score = 0.8
            else:
                alignment_score = 0.6

        return alignment_score

    def _update_correlation_history(self, validated_signals: list["ActionSignal"]) -> None:
        """
        Update correlation history for learning.

        Args:
            validated_signals: Recently validated signals
        """
        for signal in validated_signals:
            if hasattr(signal, "sac_validation"):
                sac_validation = getattr(signal, "sac_validation", {})
                if not isinstance(sac_validation, dict):
                    continue

                entry: CorrelationHistoryEntry = {
                    "timestamp": pd.Timestamp.now(),
                    "correlation_score": _coerce_float(
                        sac_validation.get("correlation_score", 0.0),
                        0.0,
                    ),
                    "pattern_type": str(getattr(signal, "pattern_type", "unknown")),
                    "signal_confidence": _coerce_float(
                        getattr(signal, "confidence", 0.5),
                        0.5,
                    ),
                    "is_valid": bool(sac_validation.get("is_valid", False)),
                }
                append_with_compaction(
                    self.correlation_history,
                    entry,
                    high_water=1000,
                    retain=500,
                )


class SACDecisionIntegrator:
    """
    Integrates SAC decisions with Action Signal Guide for enhanced decision making.
    """

    def __init__(self) -> None:
        """Initialize SAC decision integrator."""
        self.decision_weights = {
            "signal_guide": 0.6,
            "sac_decision": 0.4,
        }
        self.confidence_threshold = 0.7
        self.integration_history: list[IntegrationHistoryEntry] = []

    def integrate_decisions(
        self,
        signal_guide_signals: list["ActionSignal"],
        sac_decisions: SACDecisionPayload,
        market_data: pd.DataFrame,
    ) -> IntegratedDecision:
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
                "action": _normalize_action(sac_decisions.get("action", "HOLD"), "HOLD"),
                "confidence": _coerce_float(sac_decisions.get("confidence", 0.5), 0.5)
                * 0.8,
                "source": "sac_only",
                "reason": "no_valid_signals",
            }

        # Calculate integrated decision
        integrated_decision = self._calculate_integrated_decision(
            best_signal,
            sac_decisions,
            market_data,
        )

        # Store integration result
        entry: IntegrationHistoryEntry = {
            "timestamp": pd.Timestamp.now(),
            "signal_action": _normalize_action(getattr(best_signal, "action", "UNKNOWN"), "UNKNOWN"),
            "sac_action": _normalize_action(sac_decisions.get("action", "UNKNOWN"), "UNKNOWN"),
            "integrated_action": _normalize_action(integrated_decision["action"], "HOLD"),
            "signal_confidence": _coerce_float(getattr(best_signal, "confidence", 0.5), 0.5),
            "sac_confidence": _coerce_float(sac_decisions.get("confidence", 0.5), 0.5),
            "integrated_confidence": _coerce_float(integrated_decision["confidence"], 0.0),
        }
        append_with_compaction(
            self.integration_history,
            entry,
            high_water=500,
            retain=250,
        )

        return integrated_decision

    def _select_best_signal(self, signals: list["ActionSignal"]) -> "ActionSignal | None":
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
            key=lambda signal: (
                _coerce_float(getattr(signal, "confidence", 0.0), 0.0),
                _coerce_timestamp(getattr(signal, "timestamp", pd.Timestamp.min), pd.Timestamp.min),
            ),
            reverse=True,
        )

        return sorted_signals[0] if sorted_signals else None

    def _calculate_integrated_decision(
        self,
        best_signal: "ActionSignal",
        sac_decisions: SACDecisionPayload,
        market_data: pd.DataFrame,
    ) -> IntegratedDecision:
        """
        Calculate integrated decision from signal and SAC.

        Args:
            best_signal: Best Action Signal Guide signal
            sac_decisions: SAC decisions
            market_data: Market data

        Returns:
            Integrated decision
        """
        signal_action = _normalize_action(getattr(best_signal, "action", "HOLD"), "HOLD")
        signal_confidence = _coerce_float(getattr(best_signal, "confidence", 0.5), 0.5)

        sac_action = _normalize_action(sac_decisions.get("action", "HOLD"), "HOLD")
        sac_confidence = _coerce_float(sac_decisions.get("confidence", 0.5), 0.5)

        # Check for agreement
        if signal_action == sac_action:
            # Agreement - boost confidence
            integrated_confidence = min(
                1.0,
                (signal_confidence + sac_confidence) / 2 * 1.2,
            )
            integrated_action = signal_action
            source = "agreement"
        else:
            # Disagreement - use higher confidence decision
            if signal_confidence >= sac_confidence:
                integrated_action = signal_action
                integrated_confidence = signal_confidence * 0.9
                source = "signal_guide_dominant"
            else:
                integrated_action = sac_action
                integrated_confidence = sac_confidence * 0.9
                source = "sac_dominant"

        # Apply market condition adjustments
        market_adjustment = self._calculate_market_adjustment(
            integrated_action,
            integrated_confidence,
            market_data,
        )

        integrated_confidence = min(1.0, integrated_confidence * market_adjustment)

        return {
            "action": integrated_action,
            "confidence": float(integrated_confidence),
            "source": source,
            "signal_contribution": float(signal_confidence),
            "sac_contribution": float(sac_confidence),
            "market_adjustment": float(market_adjustment),
        }

    def _calculate_market_adjustment(
        self,
        action: str,
        confidence: float,
        market_data: pd.DataFrame,
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
        _ = confidence  # kept for backward signature compatibility
        if len(market_data) < 20 or "close" not in market_data.columns:
            return 1.0

        # Calculate market momentum
        recent_returns = market_data["close"].pct_change().tail(10)
        momentum = _coerce_float(recent_returns.mean(), 0.0)

        # Calculate volatility
        volatility = _coerce_float(recent_returns.std(), 0.0)

        adjustment = 1.0

        # Adjust based on action and market conditions
        action_label = action.upper()
        if action_label in ["BUY", "LONG"]:
            if momentum > 0.002:  # Strong upward momentum
                adjustment = 1.1
            elif momentum < -0.002:  # Strong downward momentum
                adjustment = 0.8
            if volatility > 0.03:  # High volatility
                adjustment *= 0.9
        elif action_label in ["SELL", "SHORT"]:
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

    def __init__(self) -> None:
        """Initialize SAC performance monitor."""
        self.performance_history: list[PerformanceDecisionRecord] = []
        self.accuracy_window = 50  # Last 50 decisions
        self.metrics: SACPerformanceMetrics = {
            "signal_sac_agreement_rate": 0.0,
            "integrated_decision_accuracy": 0.0,
            "signal_guide_accuracy": 0.0,
            "sac_accuracy": 0.0,
        }

    def record_decision_outcome(
        self,
        integrated_decision: IntegratedDecision,
        actual_outcome: float,
        market_data: pd.DataFrame,
    ) -> None:
        """
        Record the outcome of an integrated decision.

        Args:
            integrated_decision: The integrated decision made
            actual_outcome: Actual profit/loss outcome
            market_data: Market data at decision time
        """
        market_volatility = 0.0
        if len(market_data) >= 20 and "close" in market_data.columns:
            market_volatility = _coerce_float(
                market_data["close"].pct_change().tail(20).std(),
                0.0,
            )

        decision_record: PerformanceDecisionRecord = {
            "timestamp": pd.Timestamp.now(),
            "action": _normalize_action(integrated_decision.get("action", "HOLD"), "HOLD"),
            "confidence": _coerce_float(integrated_decision.get("confidence", 0.0), 0.0),
            "source": str(integrated_decision.get("source", "unknown")),
            "signal_contribution": _coerce_float(
                integrated_decision.get("signal_contribution", 0.0),
                0.0,
            ),
            "sac_contribution": _coerce_float(
                integrated_decision.get("sac_contribution", 0.0),
                0.0,
            ),
            "outcome": float(actual_outcome),
            "market_volatility": float(market_volatility),
        }

        append_with_compaction(
            self.performance_history,
            decision_record,
            high_water=1000,
            retain=500,
        )

        # Update metrics
        self._update_metrics()

    def get_performance_metrics(self) -> dict[str, float]:
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

        recent_decisions = self.performance_history[-self.accuracy_window :]

        # Calculate agreement rate (when both signal and SAC agree)
        agreement_decisions = [
            decision
            for decision in recent_decisions
            if decision["source"] == "agreement"
        ]
        self.metrics["signal_sac_agreement_rate"] = (
            len(agreement_decisions) / len(recent_decisions)
            if recent_decisions
            else 0.0
        )

        # Calculate integrated decision accuracy
        profitable_decisions = sum(
            1 for decision in recent_decisions if decision["outcome"] > 0
        )
        self.metrics["integrated_decision_accuracy"] = (
            profitable_decisions / len(recent_decisions)
        )

        # Calculate component accuracies
        signal_guide_decisions = [
            decision
            for decision in recent_decisions
            if "signal_guide" in decision["source"]
        ]
        if signal_guide_decisions:
            signal_profitable = sum(
                1 for decision in signal_guide_decisions if decision["outcome"] > 0
            )
            self.metrics["signal_guide_accuracy"] = (
                signal_profitable / len(signal_guide_decisions)
            )

        sac_decisions = [
            decision
            for decision in recent_decisions
            if "sac" in decision["source"]
        ]
        if sac_decisions:
            sac_profitable = sum(
                1 for decision in sac_decisions if decision["outcome"] > 0
            )
            self.metrics["sac_accuracy"] = sac_profitable / len(sac_decisions)
