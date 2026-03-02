"""
Backtest Integration for Signal Performance Analysis

Integrates SignalPerformanceAnalyzer with the UnifiedBacktester framework,
providing comprehensive signal analysis during backtest execution.
"""

from typing import Any

import pandas as pd

from ztb.utils.logging_utils import get_logger
from .performance_analyzer import BacktestPerformanceAnalyzer
from .signal_tracker import SignalTracker

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

    def initialize_backtest(self, strategy_name: str, config: dict[str, str | int | float | bool]) -> None:
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

    def track_signal(
        self,
        timestamp: pd.Timestamp,
        signal_data: dict[str, str | int | float | list[str]],
        market_data: pd.Series,
        position_before: int,
        position_after: int,
        trade_executed: bool,
        trade_result: dict[str, str | int | float] | None = None
    ) -> None:
        """
        Track a signal during backtest execution.

        Args:
            timestamp: Signal timestamp
            signal_data: Signal information from strategy
            market_data: Market data at signal time
            position_before: Position before signal
            position_after: Position after signal
            trade_executed: Whether a trade was executed
            trade_result: Trade execution result if applicable
        """
        try:
            # Track signal with signal tracker
            self.signal_tracker.track_signal(
                timestamp=timestamp,
                signal_data=signal_data,
                market_data=market_data,
                position_before=position_before,
                position_after=position_after,
                trade_executed=trade_executed,
                trade_result=trade_result
            )

            self.logger.debug(f"Tracked signal at {timestamp}: {signal_data.get('signal_type', 'unknown')}")

        except Exception as e:
            self.logger.error(f"Failed to track signal: {e}")

    def record_trade_outcome(
        self,
        signal_timestamp: pd.Timestamp,
        trade_result: dict[str, str | int | float],
        signal_data: dict[str, str | int | float | list[str]]
    ) -> None:
        """
        Record trade outcome for signal performance analysis.

        Args:
            signal_timestamp: Timestamp when signal was generated
            trade_result: Trade execution result
            signal_data: Original signal data
        """
        try:
            # Record trade outcome with performance analyzer
            self.performance_analyzer.record_trade_outcome(
                signal_timestamp=signal_timestamp,
                trade_result=trade_result,
                signal_data=signal_data
            )

            self.logger.debug(f"Recorded trade outcome for signal at {signal_timestamp}")

        except Exception as e:
            self.logger.error(f"Failed to record trade outcome: {e}")

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate comprehensive signal performance report.

        Returns:
            Dictionary containing signal tracking and performance analysis
        """
        try:
            # Get signal tracking summary
            signal_summary = self.signal_tracker.get_signal_statistics()

            # Get performance analysis summary directly
            performance_summary = self.performance_analyzer._generate_signal_effectiveness_summary()

            # Add signal quality score
            signal_quality_score = self.performance_analyzer._calculate_signal_quality_score()
            performance_summary["signal_quality_score"] = signal_quality_score

            return {
                "signal_tracking": signal_summary,
                "performance_analysis": performance_summary,  # Use summary directly
                "integration_status": "active",
                "report_timestamp": pd.Timestamp.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {
                "error": str(e),
                "status": "Report generation failed"
            }
