"""
Backtest Integration for Signal Performance Analysis

Integrates SignalPerformanceAnalyzer with the UnifiedBacktester framework,
providing comprehensive signal analysis during backtest execution.
"""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from ztb.utils.logging_utils import get_logger
from ztb.trading.backtest.unified_backtest.signal_performance.performance_analyzer import BacktestPerformanceAnalyzer
from ztb.trading.backtest.unified_backtest.signal_performance.signal_tracker import SignalTracker

logger = get_logger(__name__)


class BacktestSignalPerformanceAnalyzer:
    """
    Integrates signal performance analysis with unified backtest framework.

    Provides:
    - Real-time signal tracking during backtest
    - Trade outcome recording and analysis
    - Performance correlation analysis
    - Comprehensive reporting
    """

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize BacktestSignalPerformanceAnalyzer.

        Args:
            max_history_size: Maximum number of historical records to keep
        """
        self.signal_tracker = SignalTracker(max_history_size=max_history_size)
        self.performance_analyzer = BacktestPerformanceAnalyzer(max_history_size=max_history_size)
        self.logger = logger

    def initialize_backtest(self, strategy_name: str, config: Dict[str, Union[str, int, float, bool]]) -> None:
        """
        Initialize for a new backtest run.

        Args:
            strategy_name: Name of the strategy being tested
            config: Backtest configuration
        """
        self.signal_tracker.reset()
        self.performance_analyzer = BacktestPerformanceAnalyzer(max_history_size=self.signal_tracker.max_history_size)

        self.current_strategy = strategy_name
        self.backtest_config = config

        self.logger.info(f"Initialized signal performance analysis for strategy: {strategy_name}")

    def track_signal_generation(
        self,
        timestamp: pd.Timestamp,
        signal_data: Dict[str, Union[str, int, float, List[str]]],
        market_data: pd.Series,
        current_position: int
    ) -> None:
        """
        Track signal generation during backtest.

        Args:
            timestamp: Signal timestamp
            signal_data: Signal information from strategy
            market_data: Current market data
            current_position: Current portfolio position
        """
        # Calculate signal quality score if ActionSignalGuide data is available
        if 'source_patterns' in signal_data and signal_data.get('source_patterns'):
            try:
                quality_score = self.performance_analyzer.calculate_signal_quality_score(
                    signal_strength=signal_data.get('strength', 0.0),
                    signal_confidence=signal_data.get('confidence', 0.0),
                    pattern_type=signal_data.get('signal_type', 'unknown'),
                    historical_success_rate=0.5,  # Default, could be improved
                    consistency_score=0.5  # Default, could be improved
                )

                # Store quality score
                quality_record = {
                    'timestamp': timestamp,
                    'quality_score': quality_score,
                    'pattern_type': signal_data.get('signal_type', 'unknown'),
                    'signal_strength': signal_data.get('strength', 0.0),
                    'signal_confidence': signal_data.get('confidence', 0.0),
                    'source_patterns': signal_data.get('source_patterns', []),
                }

                self.performance_analyzer.signal_quality_history.append(quality_record)

            except Exception as e:
                self.logger.warning(f"Failed to calculate signal quality: {e}")

        # Track the signal
        self.signal_tracker.track_signal(
            timestamp=timestamp,
            signal_data=signal_data,
            market_data=market_data,
            position_before=current_position,
            position_after=current_position,  # Will be updated when trade executes
            trade_executed=False
        )

    def track_trade_execution(
        self,
        signal_timestamp: pd.Timestamp,
        trade_result: Dict[str, Union[str, int, float]],
        new_position: int
    ) -> None:
        """
        Track trade execution resulting from a signal.

        Args:
            signal_timestamp: Timestamp of the original signal
            trade_result: Trade execution details
            new_position: Position after trade execution
        """
        # Find the corresponding signal and update it
        for signal in reversed(self.signal_tracker.signals):
            if signal.timestamp == signal_timestamp and not signal.trade_executed:
                signal.position_after = new_position
                signal.trade_executed = True
                signal.trade_result = trade_result
                break

        # Record trade outcome for performance analysis
        signal_data = None
        for signal in self.signal_tracker.signals:
            if signal.timestamp == signal_timestamp:
                signal_data = {
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'direction': signal.direction,
                    'source_patterns': signal.source_patterns,
                }
                break

        if signal_data:
            self.performance_analyzer.record_trade_outcome(
                signal_timestamp=signal_timestamp,
                trade_result=trade_result,
                signal_data=signal_data
            )

    def analyze_backtest_performance(
        self,
        portfolio_returns: List[float],
        trade_history: List[Dict[str, Union[str, int, float]]]
    ) -> Dict[str, Union[float, int, str, dict, list]]:
        """
        Analyze signal performance for completed backtest.

        Args:
            portfolio_returns: Portfolio returns during backtest
            trade_history: Complete trade history

        Returns:
            Comprehensive signal performance analysis
        """
        analysis_results = {
            "signal_tracking_summary": self.signal_tracker.get_signal_statistics(),
            "trade_performance_analysis": self.performance_analyzer.analyze_signal_trade_performance(),
            "backtest_correlation_analysis": self.performance_analyzer.analyze_backtest_signal_correlations(
                backtest_returns=portfolio_returns
            ),
            "signal_effectiveness_report": self.performance_analyzer.generate_backtest_performance_report(),
        }

        # Add trade attribution analysis
        analysis_results["trade_attribution"] = self._analyze_trade_attribution(trade_history)

        # Add signal timing analysis
        analysis_results["signal_timing_analysis"] = self._analyze_signal_timing(portfolio_returns)

        return analysis_results

    def _analyze_trade_attribution(self, trade_history: List[Dict[str, Union[str, int, float]]]) -> Dict[str, Union[float, int, dict]]:
        """Analyze which signals led to successful trades."""
        if not trade_history or not self.signal_tracker.signals:
            return {"error": "Insufficient data for trade attribution analysis"}

        # Match trades with signals
        attributed_trades = []
        for trade in trade_history:
            trade_timestamp = pd.to_datetime(trade.get('timestamp'))

            # Find closest signal before the trade
            closest_signal = None
            min_time_diff = float('inf')

            for signal in self.signal_tracker.signals:
                if signal.trade_executed and signal.trade_result:
                    signal_trade_time = pd.to_datetime(signal.trade_result.get('timestamp'))
                    time_diff = abs((trade_timestamp - signal_trade_time).total_seconds())

                    if time_diff < min_time_diff and time_diff < 300:  # Within 5 minutes
                        min_time_diff = time_diff
                        closest_signal = signal

            if closest_signal:
                attributed_trades.append({
                    'trade_pnl': trade.get('pnl', 0),
                    'signal_strength': closest_signal.strength,
                    'signal_confidence': closest_signal.confidence,
                    'signal_type': closest_signal.signal_type,
                    'source_patterns': closest_signal.source_patterns,
                    'time_to_execution': min_time_diff,
                })

        if not attributed_trades:
            return {"error": "No trades could be attributed to signals"}

        df = pd.DataFrame(attributed_trades)

        return {
            "total_attributed_trades": len(df),
            "profitable_attributed_trades": int((df['trade_pnl'] > 0).sum()),
            "attribution_success_rate": float((df['trade_pnl'] > 0).mean()),
            "average_signal_strength_for_winners": float(df[df['trade_pnl'] > 0]['signal_strength'].mean()),
            "average_signal_strength_for_losers": float(df[df['trade_pnl'] <= 0]['signal_strength'].mean()),
            "signal_strength_attribution_correlation": float(
                pd.Series(df['signal_strength']).corr(pd.Series(df['trade_pnl']))
                if len(df) > 1 else 0.0
            ),
        }

    def _analyze_signal_timing(self, portfolio_returns: List[float]) -> Dict[str, Union[float, int, dict]]:
        """Analyze signal timing effectiveness."""
        signals = self.signal_tracker.get_trade_signals()

        if not signals or len(portfolio_returns) < len(signals):
            return {"error": "Insufficient data for signal timing analysis"}

        # Analyze returns around signal times
        timing_analysis = {
            "signals_after_positive_returns": 0,
            "signals_after_negative_returns": 0,
            "average_return_before_signal": 0.0,
            "average_return_after_signal": 0.0,
        }

        # Simple timing analysis - check returns before/after signals
        return_window = min(5, len(portfolio_returns) // 10)  # Look at last 5 periods or 10% of data

        recent_returns = portfolio_returns[-return_window:]

        for signal in signals[-min(len(signals), return_window):]:
            # This is a simplified analysis - in practice you'd align timestamps
            signal_return_context = np.mean(recent_returns) if recent_returns else 0.0

            if signal_return_context > 0:
                timing_analysis["signals_after_positive_returns"] += 1
            else:
                timing_analysis["signals_after_negative_returns"] += 1

        total_recent_signals = min(len(signals), return_window)
        if total_recent_signals > 0:
            timing_analysis["signals_after_positive_returns"] = (
                timing_analysis["signals_after_positive_returns"] / total_recent_signals
            )
            timing_analysis["signals_after_negative_returns"] = (
                timing_analysis["signals_after_negative_returns"] / total_recent_signals
            )

        return timing_analysis

    def get_signal_performance_summary(self) -> Dict[str, Union[float, int, str, dict]]:
        """Get summary of signal performance analysis."""
        return {
            "signal_tracking": self.signal_tracker.get_signal_statistics(),
            "performance_analysis": self.performance_analyzer.generate_backtest_performance_report(),
            "trade_outcomes": len(self.performance_analyzer.trade_outcomes),
            "total_signals_tracked": len(self.signal_tracker.signals),
            "trades_executed": len(self.signal_tracker.get_trade_signals()),
        }

    def reset(self) -> None:
        """Reset all analysis data."""
        self.signal_tracker.reset()
        self.performance_analyzer = BacktestPerformanceAnalyzer(max_history_size=self.signal_tracker.max_history_size)
        self.logger.info("BacktestSignalPerformanceAnalyzer reset")