#!/usr/bin/env python3
"""
Trend Convergence Calculator for Multi-Timeframe Analysis

Phase 2: 時間軸間トレンド収束度計算システム
複数時間軸のトレンド整合性を定量的に評価

Features:
- 時間軸間トレンド強度比較
- 収束度スコアリング (0-100)
- トレンド継続確率計算
- ダイバージェンス検出
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict

from ztb.utils.logging_utils import get_logger
from ztb.trading.signal.multi_timeframe_analyzer import (
    Timeframe, TrendDirection, TrendAnalysis, ConvergenceAnalysis
)

logger = get_logger(__name__)


@dataclass
class ConvergenceMetrics:
    """Detailed convergence metrics"""
    alignment_score: float  # 0-100 (higher = better alignment)
    strength_consistency: float  # 0-100 (higher = more consistent strength)
    momentum_harmony: float  # 0-100 (higher = harmonious momentum)
    divergence_penalty: float  # 0-50 (penalty for conflicting signals)
    confidence_level: float  # 0-100 (overall confidence in convergence)


@dataclass
class TrendConvergenceResult:
    """Complete convergence analysis result"""
    overall_score: float  # 0-100
    metrics: ConvergenceMetrics
    trend_strength: float  # 0-100
    continuation_probability: float  # 0-100
    risk_adjusted_score: float  # 0-100 (adjusted for divergence risk)
    recommendation: str  # "strong_convergence", "moderate_convergence", "weak_convergence", "divergence"


class TrendConvergenceCalculator:
    """
    Advanced trend convergence calculator for multi-timeframe analysis

    Calculates detailed convergence metrics across timeframes to provide
    more nuanced signal quality assessment.
    """

    def __init__(self) -> None:
        # Convergence scoring weights
        self.weights = {
            'alignment': 0.4,
            'strength_consistency': 0.25,
            'momentum_harmony': 0.25,
            'divergence_penalty': 0.1
        }

        # Thresholds for recommendations
        self.recommendation_thresholds = {
            'strong_convergence': 85,
            'moderate_convergence': 70,
            'weak_convergence': 50
        }

        logger.info("TrendConvergenceCalculator initialized")

    def calculate_convergence(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> TrendConvergenceResult:
        """
        Calculate comprehensive convergence analysis

        Args:
            trend_analyses: Dictionary of trend analyses by timeframe

        Returns:
            TrendConvergenceResult with detailed metrics
        """
        if not trend_analyses:
            return self._create_empty_result()

        # Calculate individual metrics
        alignment_score = self._calculate_alignment_score(trend_analyses)
        strength_consistency = self._calculate_strength_consistency(trend_analyses)
        momentum_harmony = self._calculate_momentum_harmony(trend_analyses)
        divergence_penalty = self._calculate_divergence_penalty(trend_analyses)

        # Calculate overall score
        overall_score = (
            alignment_score * self.weights['alignment'] +
            strength_consistency * self.weights['strength_consistency'] +
            momentum_harmony * self.weights['momentum_harmony'] -
            divergence_penalty * self.weights['divergence_penalty']
        )
        overall_score = max(0.0, min(100.0, overall_score))

        # Create metrics object
        metrics = ConvergenceMetrics(
            alignment_score=alignment_score,
            strength_consistency=strength_consistency,
            momentum_harmony=momentum_harmony,
            divergence_penalty=divergence_penalty,
            confidence_level=self._calculate_confidence_level(trend_analyses)
        )

        # Calculate additional metrics
        trend_strength = self._calculate_overall_trend_strength(trend_analyses)
        continuation_probability = self._calculate_continuation_probability(trend_analyses, overall_score)
        risk_adjusted_score = self._calculate_risk_adjusted_score(overall_score, divergence_penalty)

        # Determine recommendation
        recommendation = self._determine_recommendation(overall_score, divergence_penalty)

        return TrendConvergenceResult(
            overall_score=overall_score,
            metrics=metrics,
            trend_strength=trend_strength,
            continuation_probability=continuation_probability,
            risk_adjusted_score=risk_adjusted_score,
            recommendation=recommendation
        )

    def _calculate_alignment_score(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate alignment score based on directional agreement

        Higher score = more timeframes agree on direction
        """
        if len(trend_analyses) < 2:
            return 50.0

        # Count directions
        direction_counts: Dict[TrendDirection, int] = defaultdict(int)
        for analysis in trend_analyses.values():
            direction_counts[analysis.direction] += 1

        # Find dominant direction
        dominant_count = max(direction_counts.values())
        total_analyses = len(trend_analyses)

        # Calculate agreement ratio
        agreement_ratio = dominant_count / total_analyses

        # Convert to score (0-100)
        # Perfect agreement = 100, random = ~33 for 3 timeframes
        alignment_score = float((agreement_ratio - 0.33) / (1.0 - 0.33) * 100)
        alignment_score = max(0.0, min(100.0, alignment_score))

        return alignment_score

    def _calculate_strength_consistency(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate consistency of trend strengths across timeframes

        Higher score = more consistent strength levels
        """
        if len(trend_analyses) < 2:
            return 50.0

        strengths = [analysis.strength for analysis in trend_analyses.values()]

        # Calculate coefficient of variation (lower = more consistent)
        mean_strength = np.mean(strengths)
        if mean_strength == 0:
            return 50.0

        cv = np.std(strengths) / mean_strength

        # Convert CV to consistency score (0-100)
        # CV of 0 = perfect consistency = 100
        # CV of 0.5 = moderate consistency = 50
        consistency_score = max(0.0, 100.0 - (float(cv) * 200))

        return consistency_score

    def _calculate_momentum_harmony(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate harmony of momentum across timeframes

        Higher score = momentum directions are aligned
        """
        if len(trend_analyses) < 2:
            return 50.0

        momentums = [analysis.momentum for analysis in trend_analyses.values()]

        # Calculate average momentum direction
        avg_momentum = np.mean(momentums)

        # Calculate harmony as inverse of momentum spread
        momentum_spread = np.std(momentums)

        # Perfect harmony = 0 spread = 100 score
        harmony_score = max(0.0, 100.0 - (float(momentum_spread) * 2))

        return harmony_score

    def _calculate_divergence_penalty(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate penalty for conflicting signals

        Higher penalty = more conflicting signals
        """
        if len(trend_analyses) < 2:
            return 0.0

        # Check for extreme conflicts (bullish vs bearish)
        bullish_count = sum(1 for analysis in trend_analyses.values()
                          if analysis.direction in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH])
        bearish_count = sum(1 for analysis in trend_analyses.values()
                          if analysis.direction in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH])

        total_analyses = len(trend_analyses)

        # Calculate conflict ratio
        if bullish_count > 0 and bearish_count > 0:
            conflict_ratio = min(bullish_count, bearish_count) / total_analyses
            divergence_penalty = conflict_ratio * 50  # Max penalty of 50
        else:
            divergence_penalty = 0.0

        return divergence_penalty

    def _calculate_confidence_level(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate overall confidence in the convergence analysis
        """
        if not trend_analyses:
            return 0.0

        # Base confidence on number of timeframes and data quality
        timeframe_count = len(trend_analyses)
        base_confidence = min(100.0, timeframe_count * 33.3)  # 33.3 per timeframe

        # Adjust for strength consistency
        strength_consistency = self._calculate_strength_consistency(trend_analyses)
        confidence_adjustment = (strength_consistency - 50) * 0.5  # ±25 adjustment

        confidence = base_confidence + confidence_adjustment
        confidence = max(0.0, min(100.0, confidence))

        return confidence

    def _calculate_overall_trend_strength(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> float:
        """
        Calculate overall trend strength across timeframes
        """
        if not trend_analyses:
            return 0.0

        # Weight by timeframe importance (shorter = more responsive)
        weights = {
            Timeframe.M1: 0.5,
            Timeframe.M5: 0.3,
            Timeframe.M15: 0.2
        }

        weighted_strength = 0.0
        total_weight = 0.0

        for timeframe, analysis in trend_analyses.items():
            weight = weights.get(timeframe, 0.33)
            weighted_strength += analysis.strength * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_strength / total_weight

    def _calculate_continuation_probability(self, trend_analyses: Dict[Timeframe, TrendAnalysis],
                                         convergence_score: float) -> float:
        """
        Calculate probability of trend continuation
        """
        if not trend_analyses:
            return 50.0

        # Base probability on convergence score
        base_probability = convergence_score

        # Adjust based on trend strength
        trend_strength = self._calculate_overall_trend_strength(trend_analyses)
        strength_bonus = (trend_strength - 50) * 0.3  # ±15 adjustment

        probability = base_probability + strength_bonus
        probability = max(0.0, min(100.0, probability))

        return probability

    def _calculate_risk_adjusted_score(self, overall_score: float, divergence_penalty: float) -> float:
        """
        Calculate risk-adjusted convergence score
        """
        # Reduce score based on divergence risk
        risk_adjustment = divergence_penalty * 0.5  # 0-25 point reduction
        risk_adjusted_score = overall_score - risk_adjustment

        return max(0.0, risk_adjusted_score)

    def _determine_recommendation(self, overall_score: float, divergence_penalty: float) -> str:
        """
        Determine convergence recommendation
        """
        # High divergence overrides high score
        if divergence_penalty > 20:
            return "divergence"
        elif overall_score >= self.recommendation_thresholds['strong_convergence']:
            return "strong_convergence"
        elif overall_score >= self.recommendation_thresholds['moderate_convergence']:
            return "moderate_convergence"
        elif overall_score >= self.recommendation_thresholds['weak_convergence']:
            return "weak_convergence"
        else:
            return "divergence"

    def _create_empty_result(self) -> TrendConvergenceResult:
        """
        Create empty result for insufficient data
        """
        metrics = ConvergenceMetrics(
            alignment_score=0.0,
            strength_consistency=0.0,
            momentum_harmony=0.0,
            divergence_penalty=0.0,
            confidence_level=0.0
        )

        return TrendConvergenceResult(
            overall_score=0.0,
            metrics=metrics,
            trend_strength=0.0,
            continuation_probability=50.0,
            risk_adjusted_score=0.0,
            recommendation="insufficient_data"
        )

    def get_convergence_report(self, trend_analyses: Dict[Timeframe, TrendAnalysis]) -> Dict[str, Union[float, str, Dict[str, float]]]:
        """
        Get detailed convergence report

        Returns:
            Dictionary with convergence analysis details
        """
        result = self.calculate_convergence(trend_analyses)

        return {
            "overall_score": result.overall_score,
            "recommendation": result.recommendation,
            "trend_strength": result.trend_strength,
            "continuation_probability": result.continuation_probability,
            "risk_adjusted_score": result.risk_adjusted_score,
            "metrics": {
                "alignment_score": result.metrics.alignment_score,
                "strength_consistency": result.metrics.strength_consistency,
                "momentum_harmony": result.metrics.momentum_harmony,
                "divergence_penalty": result.metrics.divergence_penalty,
                "confidence_level": result.metrics.confidence_level
            }
        }