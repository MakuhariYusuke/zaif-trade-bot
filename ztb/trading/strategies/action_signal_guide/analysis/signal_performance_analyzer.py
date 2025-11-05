"""
SignalPerformanceAnalyzer Component.

This component analyzes the performance correlation between Action Signal Guide
signals and SAC learning outcomes, providing quantitative metrics for signal quality.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ztb.utils.logging_utils import get_logger


class SignalPerformanceAnalyzer:
    """
    Analyzes performance correlation between Action Signal Guide signals and SAC learning.

    This class provides:
    - Signal quality scoring algorithms
    - SAC learning curve correlation analysis
    - Signal contribution quantification
    - Performance metrics dashboard
    """

    def __init__(self, performance_tracker: Any, pattern_statistics: Any, max_history_size: int = 10000):
        """
        Initialize SignalPerformanceAnalyzer.

        Args:
            performance_tracker: PerformanceTracker instance for metrics
            pattern_statistics: PatternStatistics instance for pattern data
            max_history_size: Maximum number of historical records to keep
        """
        self.max_history_size = max_history_size
        self.logger = get_logger("ztb.trading.strategies.signal_performance_analyzer")

        # Dependencies
        self.performance_tracker = performance_tracker
        self.pattern_statistics = pattern_statistics

        # Signal quality tracking
        self.signal_quality_history: List[Dict[str, Any]] = []
        self.signal_sac_correlations: List[Dict[str, Any]] = []

        # SAC learning metrics
        self.sac_learning_curves: List[Dict[str, Any]] = []
        self.sac_action_distributions: List[Dict[str, Any]] = []

        # Performance correlation data
        self.signal_contribution_scores: Dict[str, List[float]] = defaultdict(list)
        self.regime_signal_effectiveness: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Quality scoring parameters
        self.quality_weights = {
            'strength': 0.4,
            'confidence': 0.3,
            'success_rate': 0.2,
            'consistency': 0.1
        }

    def calculate_signal_quality_score(
        self,
        signal_strength: float,
        signal_confidence: float,
        pattern_type: str,
        historical_success_rate: float,
        consistency_score: float
    ) -> float:
        """
        Calculate comprehensive signal quality score.

        Args:
            signal_strength: Signal strength (0-1)
            signal_confidence: Signal confidence (0-1)
            pattern_type: Type of pattern
            historical_success_rate: Historical success rate for this pattern
            consistency_score: Signal consistency score

        Returns:
            Quality score (0-1)
        """
        # Weighted combination of quality factors
        quality_score = (
            self.quality_weights['strength'] * signal_strength +
            self.quality_weights['confidence'] * signal_confidence +
            self.quality_weights['success_rate'] * historical_success_rate +
            self.quality_weights['consistency'] * consistency_score
        )

        # Pattern type adjustment (some patterns are inherently more reliable)
        pattern_adjustments = {
            'fibonacci': 1.1,    # Fibonacci patterns are generally reliable
            'harmonic': 1.05,    # Harmonic patterns have good success rates
            'dow_theory': 1.0,   # Dow theory is solid but conservative
            'candlestick': 0.95, # Candlestick patterns can be noisy
            'oscillator': 0.9    # Oscillators can be false signals
        }

        adjustment = pattern_adjustments.get(pattern_type, 1.0)
        final_score = min(1.0, quality_score * adjustment)

        # Record quality metrics
        quality_record = {
            'timestamp': time.time(),
            'pattern_type': pattern_type,
            'signal_strength': signal_strength,
            'signal_confidence': signal_confidence,
            'historical_success_rate': historical_success_rate,
            'consistency_score': consistency_score,
            'quality_score': final_score
        }

        self.signal_quality_history.append(quality_record)
        if len(self.signal_quality_history) > self.max_history_size:
            self.signal_quality_history = self.signal_quality_history[-self.max_history_size:]

        return final_score

    def _calculate_trend(self, values: Union[List[float], pd.Series]) -> float:
        """
        Calculate trend direction and strength.

        Args:
            values: List of values to analyze trend

        Returns:
            Trend coefficient (-1 to 1, positive = improving)
        """
        if len(values) < 2:
            return 0.0

        # Convert to numpy array if pandas Series
        if isinstance(values, pd.Series):
            values = values.values

        # Calculate linear trend using numpy polyfit
        x = np.arange(len(values))
        try:
            slope, _ = np.polyfit(x, values, 1)
            # Normalize trend by dividing by mean absolute value
            mean_abs = np.mean(np.abs(values))
            if mean_abs > 0:
                normalized_trend = slope / mean_abs
                # Clamp to [-1, 1] range
                return max(-1.0, min(1.0, normalized_trend))
            return 0.0
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def analyze_sac_learning_correlation(
        self,
        sac_learning_logs: Optional[List[Dict[str, Any]]] = None,
        correlation_window: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze correlation between SAC learning performance and signal quality.

        Args:
            sac_learning_logs: SAC learning metrics (rewards, losses, etc.)
            correlation_window: Rolling window size for correlation analysis

        Returns:
            Correlation analysis results
        """
        # Get SAC learning data from logs or use stored data
        if sac_learning_logs:
            sac_rewards = [log.get('reward', 0) for log in sac_learning_logs]
            sac_losses = [log.get('loss', 0) for log in sac_learning_logs]
            sac_timesteps = [log.get('timestep', i) for i, log in enumerate(sac_learning_logs)]
        else:
            # Use stored SAC learning curves if no logs provided
            if not self.sac_learning_curves:
                return {"error": "No SAC learning data available"}
            sac_rewards = [curve.get('reward', 0) for curve in self.sac_learning_curves]
            sac_losses = [curve.get('loss', 0) for curve in self.sac_learning_curves]
            sac_timesteps = list(range(len(sac_rewards)))

        # Get signal quality data from history
        if not self.signal_quality_history:
            return {"error": "No signal quality data available"}

        signal_qualities = [record.get('quality_score', 0) for record in self.signal_quality_history]
        signal_timestamps = [record.get('timestamp', i) for i, record in enumerate(self.signal_quality_history)]

        # Align time series (simple approach: use available data points)
        min_length = min(len(sac_rewards), len(signal_qualities))
        sac_rewards = sac_rewards[-min_length:]
        signal_qualities = signal_qualities[-min_length:]

        if len(sac_rewards) < 2:
            return {"error": "Insufficient data for correlation analysis"}

        # Calculate Pearson correlation
        try:
            correlation, p_value = stats.pearsonr(sac_rewards, signal_qualities)
        except Exception as e:
            self.logger.error(f"Correlation calculation failed: {e}")
            correlation, p_value = 0.0, 1.0

        # Calculate rolling correlations
        rolling_correlations = {}
        for window_size in [correlation_window // 4, correlation_window // 2, correlation_window]:
            if len(sac_rewards) >= window_size:
                try:
                    rolling_corr = pd.Series(sac_rewards).rolling(window_size).corr(
                        pd.Series(signal_qualities)
                    ).dropna().mean()
                    rolling_correlations[f'rolling_{window_size}'] = rolling_corr
                except Exception as e:
                    self.logger.warning(f"Rolling correlation calculation failed for window {window_size}: {e}")
                    rolling_correlations[f'rolling_{window_size}'] = 0.0

        # Signal contribution analysis
        high_quality_threshold = np.percentile(signal_qualities, 75) if signal_qualities else 0.5
        high_quality_rewards = [
            reward for reward, quality in zip(sac_rewards, signal_qualities)
            if quality >= high_quality_threshold
        ]
        low_quality_rewards = [
            reward for reward, quality in zip(sac_rewards, signal_qualities)
            if quality < high_quality_threshold
        ]

        contribution_analysis = {
            'high_quality_avg_reward': np.mean(high_quality_rewards) if high_quality_rewards else 0,
            'low_quality_avg_reward': np.mean(low_quality_rewards) if low_quality_rewards else 0,
            'quality_threshold': high_quality_threshold,
            'high_quality_count': len(high_quality_rewards),
            'low_quality_count': len(low_quality_rewards)
        }

        # Store correlation results
        correlation_record = {
            'timestamp': pd.Timestamp.now(),
            'correlation': correlation,
            'p_value': p_value,
            'rolling_correlations': rolling_correlations,
            'contribution_analysis': contribution_analysis,
            'data_points': len(sac_rewards)
        }
        self.signal_sac_correlations.append(correlation_record)

        # Keep history size manageable
        if len(self.signal_sac_correlations) > self.max_history_size:
            self.signal_sac_correlations = self.signal_sac_correlations[-self.max_history_size:]

        return {
            'correlation_coefficient': correlation,
            'p_value': p_value,
            'correlation_strength': self._interpret_correlation_strength(correlation),
            'rolling_correlations': rolling_correlations,
            'contribution_analysis': contribution_analysis,
            'data_points': len(sac_rewards),
            'correlation_trend': self._calculate_trend([c['correlation'] for c in self.signal_sac_correlations[-10:]])
        }

        contribution_analysis = {
            'high_quality_avg_reward': np.mean(high_quality_rewards) if high_quality_rewards else 0,
            'low_quality_avg_reward': np.mean(low_quality_rewards) if low_quality_rewards else 0,
            'reward_improvement': (
                np.mean(high_quality_rewards) - np.mean(low_quality_rewards)
                if high_quality_rewards and low_quality_rewards else 0
            )
        }

        correlation_result = {
            'overall_correlation': correlation,
            'p_value': p_value,
            'correlation_strength': self._interpret_correlation_strength(correlation),
            'rolling_correlations': rolling_correlations,
            'contribution_analysis': contribution_analysis,
            'analysis_timestamp': time.time()
        }

        self.signal_sac_correlations.append(correlation_result)

        return correlation_result

    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation coefficient strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        elif abs_corr >= 0.1:
            return "weak"
        else:
            return "very_weak"

    def calculate_signal_contribution_score(
        self,
        signal_quality: float,
        sac_action_alignment: float,
        market_regime: str,
        pattern_type: str
    ) -> float:
        """
        Calculate how much a signal contributes to SAC decision making.

        Args:
            signal_quality: Signal quality score
            sac_action_alignment: How well signal aligns with SAC action
            market_regime: Current market regime
            pattern_type: Type of pattern

        Returns:
            Contribution score (0-1)
        """
        # Base contribution from quality and alignment
        base_contribution = (signal_quality + sac_action_alignment) / 2

        # Regime-specific adjustments
        regime_multipliers = {
            'trending_bullish': 1.2,   # Signals more valuable in trending markets
            'trending_bearish': 1.2,
            'high_volatility': 0.8,    # Signals less reliable in high volatility
            'ranging': 1.0,            # Neutral in ranging markets
            'low_volatility': 1.1      # Signals more reliable in low volatility
        }

        regime_multiplier = regime_multipliers.get(market_regime, 1.0)

        # Pattern type effectiveness in different regimes
        pattern_regime_effectiveness = {
            'fibonacci': {'trending_bullish': 1.1, 'trending_bearish': 1.1},
            'harmonic': {'trending_bullish': 1.05, 'trending_bearish': 1.05},
            'dow_theory': {'trending_bullish': 1.2, 'trending_bearish': 1.2},
            'candlestick': {'high_volatility': 0.9, 'low_volatility': 1.1}
        }

        pattern_multiplier = 1.0
        if pattern_type in pattern_regime_effectiveness:
            pattern_multiplier = pattern_regime_effectiveness[pattern_type].get(market_regime, 1.0)

        contribution_score = min(1.0, base_contribution * regime_multiplier * pattern_multiplier)

        # Record contribution data
        self.signal_contribution_scores[pattern_type].append(contribution_score)
        self.regime_signal_effectiveness[market_regime][pattern_type] = contribution_score

        return contribution_score

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report with all metrics
        """
        report = {
            'timestamp': time.time(),
            'signal_quality_metrics': self._analyze_signal_quality_metrics(),
            'sac_correlation_analysis': self._analyze_sac_correlations(),
            'pattern_effectiveness': self._analyze_pattern_effectiveness(),
            'regime_performance': dict(self.regime_signal_effectiveness),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _analyze_signal_quality_metrics(self) -> Dict[str, Any]:
        """Analyze signal quality metrics from history."""
        if not self.signal_quality_history:
            return {}

        df = pd.DataFrame(self.signal_quality_history)

        quality_metrics = {
            'average_quality_score': df['quality_score'].mean(),
            'quality_score_std': df['quality_score'].std(),
            'quality_score_trend': self._calculate_trend(df['quality_score']),
            'pattern_quality_ranking': df.groupby('pattern_type')['quality_score'].mean().to_dict(),
            'quality_distribution': {
                'excellent': (df['quality_score'] >= 0.8).sum(),
                'good': ((df['quality_score'] >= 0.6) & (df['quality_score'] < 0.8)).sum(),
                'fair': ((df['quality_score'] >= 0.4) & (df['quality_score'] < 0.6)).sum(),
                'poor': (df['quality_score'] < 0.4).sum()
            }
        }

        return quality_metrics

    def _analyze_sac_correlations(self) -> Dict[str, Any]:
        """Analyze SAC learning correlations."""
        if not self.signal_sac_correlations:
            return {}

        recent_correlations = self.signal_sac_correlations[-10:]  # Last 10 analyses

        correlation_metrics = {
            'average_correlation': np.mean([c.get('correlation', 0) for c in recent_correlations]),
            'correlation_trend': self._calculate_trend([c.get('correlation', 0) for c in recent_correlations]),
            'strongest_correlation': max(recent_correlations, key=lambda x: abs(x.get('correlation', 0))),
            'correlation_stability': np.std([c.get('correlation', 0) for c in recent_correlations])
        }

        return correlation_metrics

    def _analyze_pattern_effectiveness(self) -> Dict[str, Any]:
        """Analyze pattern effectiveness across different metrics."""
        effectiveness = {}

        for pattern_type, scores in self.signal_contribution_scores.items():
            if scores:
                effectiveness[pattern_type] = {
                    'average_contribution': np.mean(scores),
                    'contribution_consistency': np.std(scores),
                    'total_signals': len(scores),
                    'effectiveness_trend': self._calculate_trend(scores)
                }

        return effectiveness

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []

        # Signal quality recommendations
        quality_metrics = self._analyze_signal_quality_metrics()
        if quality_metrics.get('average_quality_score', 0) < 0.6:
            recommendations.append("Signal quality is below optimal threshold. Consider adjusting pattern recognition parameters.")

        # Correlation recommendations
        correlation_metrics = self._analyze_sac_correlations()
        if correlation_metrics.get('average_correlation', 0) < 0.3:
            recommendations.append("SAC-signal correlation is weak. Consider improving signal features or SAC observation space.")

        # Pattern effectiveness recommendations
        effectiveness = self._analyze_pattern_effectiveness()
        underperforming_patterns = [
            pattern for pattern, metrics in effectiveness.items()
            if metrics['average_contribution'] < 0.5
        ]
        if underperforming_patterns:
            recommendations.append(f"Consider reducing weight for underperforming patterns: {', '.join(underperforming_patterns)}")

        return recommendations

    def analyze_correlation(self) -> Dict[str, Any]:
        """
        Analyze correlation between signals and performance.

        Returns:
            Correlation analysis results
        """
        return self.analyze_sac_learning_correlation()