"""
Strategy adapters for backtesting.

Provides adapters to wrap different trading strategies for unified backtest interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod


class StrategyAdapter(Protocol):
    """Protocol for trading strategy adapters."""

    def generate_signal(self, data: pd.DataFrame, current_position: int) -> Dict[str, Any]:
        """
        Generate trading signal.

        Args:
            data: Market data with OHLCV and features
            current_position: Current position (-1, 0, 1 for short, flat, long)

        Returns:
            Signal dict with 'action' and optional parameters
        """
        ...


class RLPolicyAdapter:
    """Adapter for RL policy (PPO trained model)."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize with trained model path."""
        self.model_path = model_path
        self.model = None
        # TODO: Load actual PPO model when available
        # For now, implement a simple rule-based fallback

    def generate_signal(self, data: pd.DataFrame, current_position: int) -> Dict[str, Any]:
        """Generate signal using RL policy."""
        if self.model is None:
            # Fallback: Simple momentum strategy
            return self._momentum_signal(data, current_position)

        # TODO: Implement actual RL policy inference
        return {"action": "hold", "confidence": 0.5}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for backtest (returns DataFrame)."""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            # Assume no position for signal generation
            signal = self.generate_signal(current_data, 0)
            signals.append(signal['action'])

        # Convert actions to signals (-1, 0, 1)
        action_to_signal = {'sell': -1, 'hold': 0, 'buy': 1}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame({
            'timestamp': data['timestamp'] if 'timestamp' in data else data.index,
            'signal': signal_values
        })

    def _momentum_signal(self, data: pd.DataFrame, current_position: int) -> Dict[str, Any]:
        """Simple momentum-based signal as RL fallback."""
        if len(data) < 20:
            return {"action": "hold", "confidence": 0.5}

        # Simple RSI-based signal
        close = data['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest_rsi = rsi.iloc[-1] if not rsi.empty else 50

        if latest_rsi < 30 and current_position <= 0:
            return {"action": "buy", "confidence": 0.7}
        elif latest_rsi > 70 and current_position >= 0:
            return {"action": "sell", "confidence": 0.7}
        else:
            return {"action": "hold", "confidence": 0.5}


class SMACrossoverAdapter:
    """Simple Moving Average crossover strategy."""

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """Initialize with MA periods."""
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, data: pd.DataFrame, current_position: int) -> Dict[str, Any]:
        """Generate SMA crossover signal."""
        if len(data) < self.slow_period:
            return {"action": "hold", "confidence": 0.5}

        close = data['close']
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        # Check for crossover
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else fast_ma.iloc[-1]
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else slow_ma.iloc[-1]
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]

        # Bullish crossover
        if prev_fast <= prev_slow and curr_fast > curr_slow and current_position <= 0:
            return {"action": "buy", "confidence": 0.8}

        # Bearish crossover
        elif prev_fast >= prev_slow and curr_fast < curr_slow and current_position >= 0:
            return {"action": "sell", "confidence": 0.8}

        return {"action": "hold", "confidence": 0.5}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for backtest (returns DataFrame)."""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            # Assume no position for signal generation
            signal = self.generate_signal(current_data, 0)
            signals.append(signal['action'])

        # Convert actions to signals (-1, 0, 1)
        action_to_signal = {'sell': -1, 'hold': 0, 'buy': 1}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame({
            'timestamp': data['timestamp'] if 'timestamp' in data else data.index,
            'signal': signal_values
        })


class BuyAndHoldAdapter:
    """Buy and hold strategy (benchmark)."""

    def __init__(self):
        """Initialize buy and hold strategy."""
        self.initialized = False

    def generate_signal(self, data: pd.DataFrame, current_position: int) -> Dict[str, Any]:
        """Generate buy and hold signal."""
        if not self.initialized and len(data) > 0:
            self.initialized = True
            return {"action": "buy", "confidence": 1.0}

        return {"action": "hold", "confidence": 1.0}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for backtest (returns DataFrame)."""
        signals = []
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            # Assume no position for signal generation
            signal = self.generate_signal(current_data, 0)
            signals.append(signal['action'])

        # Convert actions to signals (-1, 0, 1)
        action_to_signal = {'sell': -1, 'hold': 0, 'buy': 1}
        signal_values = [action_to_signal.get(s, 0) for s in signals]

        return pd.DataFrame({
            'timestamp': data['timestamp'] if 'timestamp' in data else data.index,
            'signal': signal_values
        })


def create_adapter(strategy_name: str, **kwargs) -> StrategyAdapter:
    """Factory function to create strategy adapters."""

    if strategy_name == "rl":
        return RLPolicyAdapter(**kwargs)
    elif strategy_name == "sma_fast_slow":
        return SMACrossoverAdapter(**kwargs)
    elif strategy_name == "buy_hold":
        return BuyAndHoldAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")