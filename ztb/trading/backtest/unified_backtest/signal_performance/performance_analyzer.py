"""
Backtest-Optimized Signal Performance Analyzer

Provides signal performance analysis specifically designed for backtest environments,
integrating with the unified backtest framework.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ztb.utils.logging_utils import get_logger
from ztb.trading.strategies.action_signal_guide.analysis.signal_performance_analyzer import (
    SignalPerformanceAnalyzer,
)

logger = get_logger(__name__)


class BacktestPerformanceAnalyzer(SignalPerformanceAnalyzer):
    """
    Backtest-optimized signal performance analyzer.

    Extends SignalPerformanceAnalyzer with backtest-specific functionality:
    - Trade outcome integration
    - Backtest-specific correlation analysis
    - Performance attribution by signal characteristics
    """

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize BacktestPerformanceAnalyzer.

        Args:
            max_history_size: Maximum number of historical records to keep
        """
        super().__init__(max_history_size=max_history_size)
        self.trade_outcomes: List[Dict[str, Union[str, int, float]]] = []
        self.backtest_correlations: List[Dict[str, Union[str, int, float]]] = []

    def record_trade_outcome(
        self,
        signal_timestamp: pd.Timestamp,
        trade_result: Dict[str, Union[str, int, float]],
        signal_data: Dict[str, Union[str, int, float, List[str]]]
    ) -> None:
        """
        Record trade outcome associated with a signal.

        Args:
            signal_timestamp: When the signal was generated
            trade_result: Trade execution result
            signal_data: Original signal data
        """
        try:
            outcome_record = {
                'signal_timestamp': signal_timestamp,
                'trade_timestamp': trade_result.get('timestamp'),
                'signal_type': signal_data.get('signal_type', 'unknown'),
                'signal_strength': signal_data.get('strength', 0.0),
                'signal_confidence': signal_data.get('confidence', 0.0),
                'signal_direction': signal_data.get('direction', 0.0),
                'source_patterns': signal_data.get('source_patterns', []),
                'trade_type': trade_result.get('type', 'unknown'),
                'entry_price': trade_result.get('entry_price', 0.0),
                'exit_price': trade_result.get('exit_price'),
                'quantity': trade_result.get('quantity', 0),
                'pnl': trade_result.get('pnl', 0.0),
                'pnl_pct': trade_result.get('pnl_pct', 0.0),
                'holding_period': trade_result.get('holding_period', 0),
                'market_regime': trade_result.get('market_regime', 'unknown'),
            }

            self.trade_outcomes.append(outcome_record)

            # Maintain history size
            if len(self.trade_outcomes) > self.max_history_size:
                self.trade_outcomes = self.trade_outcomes[-self.max_history_size:]

        except Exception as e:
            self.logger.warning(f"Failed to record trade outcome: {e}")

    def analyze_signal_trade_performance(self) -> Dict[str, Union[float, int, dict]]:
        """
        Analyze the relationship between signal characteristics and trade performance.

        Returns:
            Comprehensive analysis of signal-trade performance relationships
        """
        if not self.trade_outcomes:
            return {"error": "No trade outcome data available"}

        df = pd.DataFrame(self.trade_outcomes)

        analysis = {
            "total_trades": len(df),
            "profitable_trades": int((df['pnl'] > 0).sum()),
            "win_rate": float((df['pnl'] > 0).mean()),
            "average_pnl": float(df['pnl'].mean()),
            "average_pnl_pct": float(df['pnl_pct'].mean()),
            "total_pnl": float(df['pnl'].sum()),
        }

        # Signal strength vs performance correlation
        if len(df) > 5:
            strength_corr, strength_p = stats.pearsonr(df['signal_strength'], df['pnl_pct'])
            confidence_corr, confidence_p = stats.pearsonr(df['signal_confidence'], df['pnl_pct'])

            analysis["signal_strength_correlation"] = {
                "correlation": float(strength_corr),
                "p_value": float(strength_p),
                "significant": strength_p < 0.05
            }

            analysis["signal_confidence_correlation"] = {
                "correlation": float(confidence_corr),
                "p_value": float(confidence_p),
                "significant": confidence_p < 0.05
            }

        # Performance by signal strength quartiles
        strength_quartiles = pd.qcut(df['signal_strength'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_performance = df.groupby(strength_quartiles)['pnl_pct'].agg(['mean', 'count', 'std'])
        analysis["performance_by_strength_quartile"] = quartile_performance.to_dict()

        # Performance by signal type
        type_performance = df.groupby('signal_type')['pnl_pct'].agg(['mean', 'count', 'std'])
        analysis["performance_by_signal_type"] = type_performance.to_dict()

        # Pattern effectiveness analysis
        pattern_performance = self._analyze_pattern_effectiveness_in_trades(df)
        analysis["pattern_effectiveness"] = pattern_performance

        return analysis

    def _analyze_pattern_effectiveness_in_trades(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Analyze which patterns are most effective in trades."""
        pattern_stats = {}

        # Explode patterns to analyze individually
        pattern_df = df.explode('source_patterns')
        pattern_df = pattern_df.rename(columns={'source_patterns': 'pattern'})

        if not pattern_df.empty and 'pattern' in pattern_df.columns:
            pattern_performance = pattern_df.groupby('pattern')['pnl_pct'].agg([
                'mean', 'count', 'std', 'median'
            ]).round(4)

            # Sort by mean performance
            pattern_performance = pattern_performance.sort_values('mean', ascending=False)

            pattern_stats = pattern_performance.to_dict('index')

        return pattern_stats

    def analyze_backtest_signal_correlations(
        self,
        backtest_returns: List[float],
        signal_features: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Union[float, int, dict]]:
        """
        Analyze correlations between backtest performance and signal features.

        Args:
            backtest_returns: Backtest portfolio returns
            signal_features: Optional signal feature data

        Returns:
            Correlation analysis results
        """
        if not self.signal_quality_history:
            return {"error": "No signal quality data available"}

        # Get signal quality scores aligned with backtest returns
        signal_qualities = [
            record.get('quality_score', 0)
            for record in self.signal_quality_history[-len(backtest_returns):]
        ]

        # Ensure same length
        min_length = min(len(backtest_returns), len(signal_qualities))
        backtest_returns = backtest_returns[-min_length:]
        signal_qualities = signal_qualities[-min_length:]

        if len(backtest_returns) < 5:
            return {"error": "Insufficient data for correlation analysis"}

        # Calculate correlations
        correlation, p_value = stats.pearsonr(backtest_returns, signal_qualities)

        # Rolling correlations
        rolling_correlations = {}
        for window in [10, 25, 50]:
            if len(backtest_returns) >= window:
                try:
                    rolling_corr = pd.Series(backtest_returns).rolling(window).corr(
                        pd.Series(signal_qualities)
                    ).dropna().mean()
                    rolling_correlations[f'rolling_{window}'] = float(rolling_corr)
                except Exception as e:
                    self.logger.warning(f"Rolling correlation calculation failed for window {window}: {e}")

        # Store correlation result
        correlation_record = {
            'timestamp': pd.Timestamp.now(),
            'correlation': correlation,
            'p_value': p_value,
            'rolling_correlations': rolling_correlations,
            'data_points': len(backtest_returns),
            'backtest_return_mean': np.mean(backtest_returns),
            'signal_quality_mean': np.mean(signal_qualities),
        }

        self.backtest_correlations.append(correlation_record)

        return {
            'overall_correlation': correlation,
            'p_value': p_value,
            'correlation_strength': self._interpret_correlation_strength(correlation),
            'rolling_correlations': rolling_correlations,
            'data_points': len(backtest_returns),
            'correlation_trend': self._calculate_trend([c['correlation'] for c in self.backtest_correlations[-10:]]),
            'backtest_signal_quality_relationship': self._analyze_signal_quality_impact(backtest_returns, signal_qualities),
        }

    def _analyze_signal_quality_impact(
        self,
        backtest_returns: List[float],
        signal_qualities: List[float]
    ) -> Dict[str, Union[float, int]]:
        """Analyze the impact of signal quality on backtest performance."""
        # Create quality-based portfolios
        high_quality_threshold = np.percentile(signal_qualities, 75)
        low_quality_threshold = np.percentile(signal_qualities, 25)

        high_quality_returns = [
            ret for ret, qual in zip(backtest_returns, signal_qualities)
            if qual >= high_quality_threshold
        ]

        low_quality_returns = [
            ret for ret, qual in zip(backtest_returns, signal_qualities)
            if qual <= low_quality_threshold
        ]

        return {
            'high_quality_return_mean': np.mean(high_quality_returns) if high_quality_returns else 0.0,
            'low_quality_return_mean': np.mean(low_quality_returns) if low_quality_returns else 0.0,
            'high_quality_count': len(high_quality_returns),
            'low_quality_count': len(low_quality_returns),
            'quality_threshold_high': high_quality_threshold,
            'quality_threshold_low': low_quality_threshold,
            'return_improvement': (
                np.mean(high_quality_returns) - np.mean(low_quality_returns)
                if high_quality_returns and low_quality_returns else 0.0
            ),
        }

    def generate_backtest_performance_report(self) -> Dict[str, Union[float, int, str, dict, list]]:
        """
        Generate comprehensive backtest performance report.

        Returns:
            Complete backtest performance analysis report
        """
        base_report = self.generate_performance_report()

        # Add backtest-specific analysis
        backtest_analysis = {
            "trade_outcome_analysis": self.analyze_signal_trade_performance(),
            "backtest_correlations": self.backtest_correlations[-5:] if self.backtest_correlations else [],
            "signal_effectiveness_summary": self._generate_signal_effectiveness_summary(),
        }

        # Merge with base report
        base_report.update(backtest_analysis)

        return base_report

    def _generate_signal_effectiveness_summary(self) -> Dict[str, Union[float, int, str]]:
        """Generate summary of signal effectiveness in backtest context."""
        if not self.trade_outcomes:
            return {"status": "No trade data available"}

        df = pd.DataFrame(self.trade_outcomes)

        # Calculate key effectiveness metrics
        win_rate = (df['pnl'] > 0).mean()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if (df['pnl'] > 0).any() else 0
        avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if (df['pnl'] <= 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        return {
            "overall_win_rate": float(win_rate),
            "average_win": float(avg_win),
            "average_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "total_return": float(df['pnl'].sum()),
            "best_trade": float(df['pnl'].max()),
            "worst_trade": float(df['pnl'].min()),
            "trade_count": len(df),
            "effective_signals": int((df['pnl'] > 0).sum()),
        }