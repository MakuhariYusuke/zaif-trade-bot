#!/usr/bin/env python3
"""
Market Condition-Specific Performance Analysis for SAC v428.

This module provides detailed performance analysis segmented by market conditions:
- Performance metrics by regime (bull/bear/sideways)
- Volatility-adjusted performance analysis
- Trend strength correlation analysis
- Regime transition impact assessment
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.analysis.specialized.market.market_regime_classifier import (
    MarketCondition,
    MarketRegime,
    VolatilityLevel,
)
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegimePerformanceMetrics:
    """Performance metrics for a specific market regime."""

    regime: MarketRegime
    volatility: VolatilityLevel
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    confidence_interval: Tuple[float, float]
    sample_size: int


@dataclass
class PerformanceComparison:
    """Comparison between different market conditions."""

    metric_name: str
    bull_performance: float
    bear_performance: float
    sideways_performance: float
    statistical_significance: float
    effect_size: float
    confidence_level: float


class RegimePerformanceAnalyzer:
    """
    Advanced performance analysis system for different market conditions.

    Provides comprehensive analysis of trading performance across:
    - Market regimes (bull/bear/sideways)
    - Volatility levels
    - Trend strength categories
    - Regime transitions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize regime performance analyzer.

        Args:
            config: Configuration parameters
        """
        self.config = config or self._get_default_config()
        self.logger = get_logger(f"{self.__class__.__name__}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "min_regime_periods": 10,
            "confidence_level": 0.95,
            "bootstrap_samples": 1000,
            "risk_free_rate": 0.02,
            "benchmark_returns": None,
        }

    def analyze_performance_by_regime(
        self, backtest_results: Dict[str, Any], market_conditions: List[MarketCondition]
    ) -> Dict[str, Any]:
        """
        Analyze trading performance segmented by market conditions.

        Args:
            backtest_results: Backtest results dictionary
            market_conditions: List of market conditions

        Returns:
            Comprehensive performance analysis by regime
        """
        # Align backtest results with market conditions
        aligned_data = self._align_backtest_with_conditions(
            backtest_results, market_conditions
        )

        if not aligned_data:
            self.logger.warning("No aligned data available for regime analysis")
            return {}

        # Calculate performance by regime
        regime_performance = self._calculate_regime_performance(aligned_data)

        # Calculate performance by volatility
        volatility_performance = self._calculate_volatility_performance(aligned_data)

        # Calculate trend strength correlations
        trend_correlations = self._calculate_trend_correlations(aligned_data)

        # Calculate regime transition impacts
        transition_impacts = self._calculate_transition_impacts(aligned_data)

        # Statistical comparisons
        comparisons = self._calculate_statistical_comparisons(regime_performance)

        return {
            "regime_performance": regime_performance,
            "volatility_performance": volatility_performance,
            "trend_correlations": trend_correlations,
            "transition_impacts": transition_impacts,
            "statistical_comparisons": comparisons,
            "summary_statistics": self._calculate_summary_statistics(aligned_data),
            "recommendations": self._generate_recommendations(
                regime_performance, comparisons
            ),
        }

    def _align_backtest_with_conditions(
        self, backtest_results: Dict[str, Any], market_conditions: List[MarketCondition]
    ) -> List[Dict[str, Any]]:
        """
        Align backtest results with market conditions by timestamp.

        Args:
            backtest_results: Backtest results
            market_conditions: Market conditions

        Returns:
            List of aligned data points
        """
        aligned_data = []

        # Extract trade data from backtest results
        trades = backtest_results.get("trades", [])
        if not trades:
            self.logger.warning("No trades found in backtest results")
            return []

        self.logger.info(f"Processing {len(trades)} trades")
        if trades:
            trade_times = [
                pd.to_datetime(trade.get("timestamp", trade.get("entry_time")))
                for trade in trades
                if trade.get("timestamp") or trade.get("entry_time")
            ]
            if trade_times:
                self.logger.info(
                    f"Trade time range: {min(trade_times)} to {max(trade_times)}"
                )
                self.logger.info(f"Sample trade timestamps: {trade_times[:3]}")

        # Convert market conditions to dict by timestamp
        conditions_by_time = {}
        for condition in market_conditions:
            # Normalize timestamp to minute level for better alignment
            time_key = condition.timestamp.replace(second=0, microsecond=0)
            conditions_by_time[time_key] = condition

        self.logger.info(
            f"Market conditions: {len(conditions_by_time)} unique timestamps"
        )
        if conditions_by_time:
            time_keys = list(conditions_by_time.keys())
            self.logger.info(
                f"Market condition time range: {min(time_keys)} to {max(time_keys)}"
            )
            self.logger.info(f"Sample market condition timestamps: {time_keys[:3]}")

        # Align each trade with market condition
        for trade in trades:
            trade_time = pd.to_datetime(trade.get("timestamp", trade.get("entry_time")))
            if trade_time:
                # Normalize trade timestamp for better matching
                trade_time_normalized = trade_time.replace(second=0, microsecond=0)
                # Find closest market condition
                aligned_condition = self._find_closest_condition(
                    trade_time_normalized, conditions_by_time
                )
                if aligned_condition:
                    aligned_data.append(
                        {
                            "trade": trade,
                            "condition": aligned_condition,
                            "timestamp": trade_time,
                        }
                    )
                else:
                    self.logger.debug(
                        f"No condition found for trade at {trade_time_normalized}"
                    )

        self.logger.info(
            f"Successfully aligned {len(aligned_data)} out of {len(trades)} trades with market conditions"
        )

        self.logger.info(f"Aligned {len(aligned_data)} trades with market conditions")
        return aligned_data

    def _find_closest_condition(
        self,
        trade_time_normalized: pd.Timestamp,
        conditions_by_time: Dict[pd.Timestamp, MarketCondition],
    ) -> Optional[MarketCondition]:
        """Find the closest market condition by timestamp."""
        if not conditions_by_time:
            return None

        # First try exact match with normalized timestamp
        if trade_time_normalized in conditions_by_time:
            return conditions_by_time[trade_time_normalized]

        # Find closest match within reasonable time window (e.g., 1 hour)
        time_window = pd.Timedelta(hours=1)
        closest_condition = None
        min_diff = time_window

        for condition_time, condition in conditions_by_time.items():
            time_diff = abs(trade_time_normalized - condition_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_condition = condition

        return closest_condition

    def _calculate_regime_performance(
        self, aligned_data: List[Dict[str, Any]]
    ) -> Dict[str, RegimePerformanceMetrics]:
        """Calculate performance metrics for each market regime."""
        regime_data = defaultdict(list)

        # Group trades by regime
        for item in aligned_data:
            regime = item["condition"].regime.value
            regime_data[regime].append(item)

        regime_performance = {}

        for regime, trades in regime_data.items():
            if len(trades) < self.config["min_regime_periods"]:
                self.logger.warning(
                    f"Insufficient data for regime {regime}: {len(trades)} trades"
                )
                continue

            try:
                metrics = self._calculate_performance_metrics(trades)
                regime_performance[regime] = metrics
            except Exception as e:
                self.logger.error(
                    f"Failed to calculate metrics for regime {regime}: {e}"
                )

        return regime_performance

    def _calculate_performance_metrics(
        self, trades: List[Dict[str, Any]]
    ) -> RegimePerformanceMetrics:
        """Calculate comprehensive performance metrics for a set of trades."""
        if not trades:
            raise ValueError("No trades provided")

        # Extract returns and other metrics
        returns = []
        wins = 0
        total_trades = len(trades)
        total_return = 0
        max_drawdown = 0
        peak = 0
        current_drawdown = 0

        for trade in trades:
            trade_data = trade["trade"]
            pnl = trade_data.get("pnl", trade_data.get("profit_loss", 0))
            returns.append(pnl)
            total_return += pnl

            if pnl > 0:
                wins += 1

            # Calculate drawdown
            cumulative = sum(returns)
            if cumulative > peak:
                peak = cumulative
                current_drawdown = 0
            else:
                current_drawdown = peak - cumulative
                max_drawdown = max(max_drawdown, current_drawdown)

        # Calculate metrics
        returns_array = np.array(returns)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Sharpe ratio
        if len(returns_array) > 1:
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Profit factor
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r < 0]

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        profit_factor = (
            (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades))
            if avg_loss > 0
            else float("inf")
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = avg_return / downside_std if downside_std > 0 else 0

        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float("inf")

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._calculate_confidence_interval(returns_array)

        # Get regime and volatility from first trade
        regime = trades[0]["condition"].regime
        volatility = trades[0]["condition"].volatility

        return RegimePerformanceMetrics(
            regime=regime,
            volatility=volatility,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=0,  # TODO: Calculate from trade data
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            alpha=0,  # TODO: Calculate against benchmark
            beta=0,  # TODO: Calculate against benchmark
            confidence_interval=(ci_lower, ci_upper),
            sample_size=total_trades,
        )

    def _calculate_confidence_interval(
        self, returns: np.ndarray, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrapping."""
        if len(returns) < 2:
            return (np.mean(returns), np.mean(returns))

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))

        ci_lower = np.percentile(
            bootstrap_means, (1 - self.config["confidence_level"]) * 100 / 2
        )
        ci_upper = np.percentile(
            bootstrap_means, 100 - (1 - self.config["confidence_level"]) * 100 / 2
        )

        return (ci_lower, ci_upper)

    def _calculate_volatility_performance(
        self, aligned_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate performance by volatility levels."""
        volatility_data = defaultdict(list)

        for item in aligned_data:
            vol_level = item["condition"].volatility.value
            volatility_data[vol_level].append(item)

        volatility_performance = {}
        for vol_level, trades in volatility_data.items():
            if len(trades) >= self.config["min_regime_periods"]:
                try:
                    metrics = self._calculate_performance_metrics(trades)
                    volatility_performance[vol_level] = metrics
                except Exception as e:
                    self.logger.error(
                        f"Failed to calculate volatility metrics for {vol_level}: {e}"
                    )

        return volatility_performance

    def _calculate_trend_correlations(
        self, aligned_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate correlations between trend strength and performance."""
        trend_strengths = []
        performances = []

        for item in aligned_data:
            trend_strength = item["condition"].trend_strength
            pnl = item["trade"].get("pnl", item["trade"].get("profit_loss", 0))

            trend_strengths.append(trend_strength)
            performances.append(pnl)

        if len(trend_strengths) < 10:
            return {}

        correlations = {}
        try:
            correlations["trend_strength_pnl"] = np.corrcoef(
                trend_strengths, performances
            )[0, 1]
            correlations["trend_strength_abs_pnl"] = np.corrcoef(
                trend_strengths, np.abs(performances)
            )[0, 1]
        except Exception:
            correlations["trend_strength_pnl"] = 0
            correlations["trend_strength_abs_pnl"] = 0

        return correlations

    def _calculate_transition_impacts(
        self, aligned_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate performance impact of regime transitions."""
        if len(aligned_data) < 2:
            return {}

        transitions = []
        sorted_data = sorted(aligned_data, key=lambda x: x["timestamp"])

        for i in range(1, len(sorted_data)):
            prev_regime = sorted_data[i - 1]["condition"].regime.value
            curr_regime = sorted_data[i]["condition"].regime.value
            pnl = sorted_data[i]["trade"].get("pnl", 0)

            transitions.append(
                {
                    "transition": f"{prev_regime}_to_{curr_regime}",
                    "pnl": pnl,
                    "timestamp": sorted_data[i]["timestamp"],
                }
            )

        # Group by transition type
        transition_stats = defaultdict(list)
        for t in transitions:
            transition_stats[t["transition"]].append(t["pnl"])

        # Calculate statistics
        transition_impacts = {}
        for transition, pnls in transition_stats.items():
            if len(pnls) >= 3:  # Minimum sample size
                transition_impacts[transition] = {
                    "avg_pnl": np.mean(pnls),
                    "std_pnl": np.std(pnls),
                    "total_trades": len(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                }

        return dict(transition_impacts)

    def _calculate_statistical_comparisons(
        self, regime_performance: Dict[str, RegimePerformanceMetrics]
    ) -> List[PerformanceComparison]:
        """Calculate statistical comparisons between regimes."""
        comparisons = []

        if len(regime_performance) < 2:
            return comparisons

        metrics_to_compare = [
            "sharpe_ratio",
            "win_rate",
            "total_return",
            "profit_factor",
        ]

        for metric in metrics_to_compare:
            values = {}
            for regime, perf in regime_performance.items():
                values[regime] = getattr(perf, metric, 0)

            if len(values) >= 2:
                # Simple statistical significance test (t-test approximation)
                regimes_list = list(values.keys())
                if len(regimes_list) >= 2:
                    # For now, just calculate differences
                    # TODO: Implement proper statistical testing
                    comparison = PerformanceComparison(
                        metric_name=metric,
                        bull_performance=values.get("bull", 0),
                        bear_performance=values.get("bear", 0),
                        sideways_performance=values.get("sideways", 0),
                        statistical_significance=0.5,  # Placeholder
                        effect_size=0.0,  # Placeholder
                        confidence_level=self.config["confidence_level"],
                    )
                    comparisons.append(comparison)

        return comparisons

    def _calculate_summary_statistics(
        self, aligned_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        total_trades = len(aligned_data)
        total_pnl = sum(item["trade"].get("pnl", 0) for item in aligned_data)
        winning_trades = sum(
            1 for item in aligned_data if item["trade"].get("pnl", 0) > 0
        )

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "regime_coverage": len(
                set(item["condition"].regime.value for item in aligned_data)
            ),
            "volatility_coverage": len(
                set(item["condition"].volatility.value for item in aligned_data)
            ),
        }

    def _generate_recommendations(
        self,
        regime_performance: Dict[str, RegimePerformanceMetrics],
        comparisons: List[PerformanceComparison],
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Find best performing regime
        if regime_performance:
            best_regime = max(
                regime_performance.keys(),
                key=lambda r: regime_performance[r].sharpe_ratio,
            )
            recommendations.append(
                f"Best performance in {best_regime} market conditions"
            )

        # Check for significant performance differences
        for comparison in comparisons:
            diff = abs(comparison.bull_performance - comparison.bear_performance)
            if diff > 0.5:  # Significant difference threshold
                better_regime = (
                    "bull"
                    if comparison.bull_performance > comparison.bear_performance
                    else "bear"
                )
                recommendations.append(
                    f"Consider regime-specific strategies: {better_regime} markets show superior {comparison.metric_name}"
                )

        # Volatility recommendations
        if not recommendations:
            recommendations.append(
                "Implement adaptive strategies based on market regime detection"
            )

        return recommendations
