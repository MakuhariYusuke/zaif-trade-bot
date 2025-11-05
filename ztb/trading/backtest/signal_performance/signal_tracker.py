"""
Signal Tracker for Backtest Integration

Tracks Action Signal Guide signals during backtest execution,
collecting data for performance analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedSignal:
    """Represents a tracked signal during backtest."""
    timestamp: pd.Timestamp
    signal_type: str
    direction: float
    strength: float
    confidence: float
    source_patterns: List[str]
    market_data: Dict[str, float]
    position_before: int
    position_after: int
    trade_executed: bool
    trade_result: Optional[Dict[str, Union[str, int, float]]] = None


class SignalTracker:
    """
    Tracks Action Signal Guide signals during backtest execution.

    Collects comprehensive signal data including:
    - Signal characteristics (strength, confidence, patterns)
    - Market context at signal generation
    - Position changes and trade outcomes
    - Performance attribution data
    """

    def __init__(self, max_history_size: int = 10000):
        """
        Initialize SignalTracker.

        Args:
            max_history_size: Maximum number of signals to track
        """
        self.max_history_size = max_history_size
        self.signals: List[TrackedSignal] = []
        self.current_position = 0
        self.logger = logger

    def track_signal(
        self,
        timestamp: pd.Timestamp,
        signal_data: Dict[str, Union[str, int, float, List[str]]],
        market_data: pd.Series,
        position_before: int,
        position_after: int,
        trade_executed: bool,
        trade_result: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> None:
        """
        Track a signal during backtest execution.

        Args:
            timestamp: Signal timestamp
            signal_data: Signal information from ActionSignalGuide
            market_data: Market data at signal time
            position_before: Position before signal
            position_after: Position after signal
            trade_executed: Whether a trade was executed
            trade_result: Trade execution result if applicable
        """
        try:
            tracked_signal = TrackedSignal(
                timestamp=timestamp,
                signal_type=signal_data.get('signal_type', 'unknown'),
                direction=signal_data.get('direction', 0.0),
                strength=signal_data.get('strength', 0.0),
                confidence=signal_data.get('confidence', 0.0),
                source_patterns=signal_data.get('source_patterns', []),
                market_data={
                    'open': market_data.get('open', 0.0),
                    'high': market_data.get('high', 0.0),
                    'low': market_data.get('low', 0.0),
                    'close': market_data.get('close', 0.0),
                    'volume': market_data.get('volume', 0.0),
                    'returns': market_data.get('returns', 0.0),
                },
                position_before=position_before,
                position_after=position_after,
                trade_executed=trade_executed,
                trade_result=trade_result
            )

            self.signals.append(tracked_signal)
            self.current_position = position_after

            # Maintain history size
            if len(self.signals) > self.max_history_size:
                self.signals = self.signals[-self.max_history_size:]

        except Exception as e:
            self.logger.warning(f"Failed to track signal: {e}")

    def get_signal_history(self) -> List[TrackedSignal]:
        """Get complete signal tracking history."""
        return self.signals.copy()

    def get_recent_signals(self, n: int = 100) -> List[TrackedSignal]:
        """Get most recent n signals."""
        return self.signals[-n:] if len(self.signals) > n else self.signals.copy()

    def get_signals_by_pattern(self, pattern: str) -> List[TrackedSignal]:
        """Get signals that include a specific pattern."""
        return [
            signal for signal in self.signals
            if pattern in signal.source_patterns
        ]

    def get_signals_by_type(self, signal_type: str) -> List[TrackedSignal]:
        """Get signals of a specific type."""
        return [
            signal for signal in self.signals
            if signal.signal_type == signal_type
        ]

    def get_trade_signals(self) -> List[TrackedSignal]:
        """Get signals that resulted in trades."""
        return [
            signal for signal in self.signals
            if signal.trade_executed
        ]

    def get_signal_statistics(self) -> Dict[str, Union[int, float, dict]]:
        """Get comprehensive signal statistics."""
        if not self.signals:
            return {"total_signals": 0}

        df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'signal_type': s.signal_type,
                'direction': s.direction,
                'strength': s.strength,
                'confidence': s.confidence,
                'pattern_count': len(s.source_patterns),
                'trade_executed': s.trade_executed,
                'position_change': s.position_after - s.position_before,
            }
            for s in self.signals
        ])

        # Basic statistics
        stats = {
            "total_signals": len(df),
            "executed_trades": int(df['trade_executed'].sum()),
            "unique_signal_types": df['signal_type'].nunique(),
            "total_patterns_used": df['pattern_count'].sum(),
            "average_patterns_per_signal": df['pattern_count'].mean(),
            "trade_execution_rate": df['trade_executed'].mean(),
            "position_change_rate": (df['position_change'] != 0).mean(),
        }

        # Signal type distribution
        stats["signal_type_distribution"] = df['signal_type'].value_counts().to_dict()

        # Strength and confidence statistics
        stats["signal_strength"] = {
            "mean": df['strength'].mean(),
            "std": df['strength'].std(),
            "min": df['strength'].min(),
            "max": df['strength'].max(),
        }

        stats["signal_confidence"] = {
            "mean": df['confidence'].mean(),
            "std": df['confidence'].std(),
            "min": df['confidence'].min(),
            "max": df['confidence'].max(),
        }

        # Direction distribution
        direction_counts = pd.cut(df['direction'],
                                bins=[-1.1, -0.1, 0.1, 1.1],
                                labels=['sell', 'hold', 'buy']).value_counts()
        stats["direction_distribution"] = direction_counts.to_dict()

        return stats

    def get_signal_summary(self) -> Dict[str, Union[int, float, dict]]:
        """Get comprehensive signal summary for reporting."""
        return self.get_signal_statistics()

    def reset(self) -> None:
        """Reset signal tracking history."""
        self.signals.clear()
        self.current_position = 0
        self.logger.info("Signal tracker reset")