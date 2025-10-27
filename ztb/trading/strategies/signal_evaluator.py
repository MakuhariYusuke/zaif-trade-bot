"""
Signal Evaluator - Backtesting and Validation for Technical Signals

This module provides tools to evaluate the effectiveness of technical signals
through backtesting and statistical analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.utils.logging_utils import get_logger

from .signal_definitions import SignalDefinitions, SignalType


@dataclass
class SignalPerformance:
    """Performance metrics for a signal."""

    signal_name: str
    total_signals: int
    profitable_signals: int
    win_rate: float
    avg_return: float
    max_return: float
    min_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float


@dataclass
class BacktestResult:
    """Results from signal backtesting."""

    signal_performances: Dict[str, SignalPerformance]
    overall_win_rate: float
    overall_return: float
    benchmark_return: float
    alpha: float
    start_date: datetime
    end_date: datetime
    num_trades: int


class SignalEvaluator:
    """
    Evaluates technical signals through backtesting and statistical analysis.

    This class provides comprehensive evaluation of signal effectiveness,
    including win rates, returns, risk metrics, and comparative analysis.
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
        benchmark_symbol: str = "BTC/JPY",
    ):
        """
        Initialize the signal evaluator.

        Args:
            commission_rate: Trading commission rate (0.001 = 0.1%)
            slippage: Estimated slippage cost
            benchmark_symbol: Symbol for benchmark comparison
        """
        self.logger = get_logger("SignalEvaluator")
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.benchmark_symbol = benchmark_symbol

        self.signal_definitions = SignalDefinitions()

        self.logger.info("Initialized SignalEvaluator")

    def backtest_signals(
        self,
        data: pd.DataFrame,
        signals: List[str],
        initial_capital: float = 100000.0,
        position_size: float = 0.1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> BacktestResult:
        """
        Backtest a set of signals on historical data.

        Args:
            data: Historical OHLCV data with features
            signals: List of signal names to test
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            stop_loss: Stop loss percentage (None for no stop loss)
            take_profit: Take profit percentage (None for no take profit)

        Returns:
            BacktestResult with performance metrics
        """
        self.logger.info(
            f"Starting backtest with {len(signals)} signals on {len(data)} data points"
        )

        # Initialize portfolio
        capital = initial_capital
        position = 0.0  # Position size in base currency
        entry_price = 0.0
        trades = []
        signal_performances = {}

        # Initialize signal tracking
        for signal in signals:
            signal_performances[signal] = {
                "signals": 0,
                "profitable": 0,
                "returns": [],
                "entry_times": [],
                "exit_times": [],
            }

        # Process each time step
        for idx, row in data.iterrows():
            current_price = row["close"]

            # Check for signal exits first
            if position != 0.0:
                exit_signal = self._check_exit_conditions(
                    position, entry_price, current_price, stop_loss, take_profit
                )

                if exit_signal:
                    # Exit position
                    exit_value = abs(position) * current_price
                    commission = exit_value * self.commission_rate

                    if position > 0:  # Long position
                        pnl = (current_price - entry_price) * abs(position) - commission
                    else:  # Short position
                        pnl = (entry_price - current_price) * abs(position) - commission

                    capital += pnl
                    trades.append(
                        {
                            "entry_time": entry_price_time,
                            "exit_time": idx,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "position": position,
                            "pnl": pnl,
                            "return_pct": pnl / (abs(position) * entry_price),
                        }
                    )

                    position = 0.0
                    entry_price = 0.0

            # Check for new signals
            if position == 0.0:  # Only enter if no position
                for signal in signals:
                    sig_type, strength = self.signal_definitions.evaluate_signal(
                        signal, row.values, data.columns.tolist()
                    )

                    if strength > 0.5:  # Only act on strong signals
                        # Determine action based on signal type
                        if sig_type == SignalType.BUY:
                            action = 1  # BUY
                        elif sig_type == SignalType.SELL:
                            action = -1  # SELL (short)
                        else:
                            continue

                        # Calculate position size
                        position_value = capital * position_size
                        trade_size = position_value / current_price

                        # Apply commission and slippage
                        commission = position_value * self.commission_rate
                        effective_price = current_price * (1 + self.slippage * action)

                        # Enter position
                        position = trade_size * action
                        entry_price = effective_price
                        entry_price_time = idx
                        capital -= commission

                        # Track signal performance
                        signal_performances[signal]["signals"] += 1
                        signal_performances[signal]["entry_times"].append(idx)

                        break  # Only take one signal per timestep

        # Calculate final performance metrics
        total_return = (capital - initial_capital) / initial_capital

        # Calculate benchmark return (buy and hold)
        benchmark_return = (data["close"].iloc[-1] - data["close"].iloc[0]) / data[
            "close"
        ].iloc[0]

        # Convert signal performances to SignalPerformance objects
        signal_perf_objects = {}
        overall_win_rate = 0.0
        total_signals = 0

        for signal_name, perf in signal_performances.items():
            if perf["signals"] > 0:
                returns = (
                    perf["returns"]
                    if "returns" in perf
                    else [
                        t["return_pct"]
                        for t in trades
                        if t.get("signal") == signal_name
                    ]
                )
                profitable = sum(1 for r in returns if r > 0)

                signal_perf_objects[signal_name] = SignalPerformance(
                    signal_name=signal_name,
                    total_signals=perf["signals"],
                    profitable_signals=profitable,
                    win_rate=profitable / perf["signals"]
                    if perf["signals"] > 0
                    else 0.0,
                    avg_return=np.mean(returns) if returns else 0.0,
                    max_return=np.max(returns) if returns else 0.0,
                    min_return=np.min(returns) if returns else 0.0,
                    sharpe_ratio=self._calculate_sharpe_ratio(returns),
                    max_drawdown=self._calculate_max_drawdown(returns),
                    total_return=np.sum(returns) if returns else 0.0,
                )

                overall_win_rate += (
                    signal_perf_objects[signal_name].win_rate * perf["signals"]
                )
                total_signals += perf["signals"]

        overall_win_rate = (
            overall_win_rate / total_signals if total_signals > 0 else 0.0
        )

        result = BacktestResult(
            signal_performances=signal_perf_objects,
            overall_win_rate=overall_win_rate,
            overall_return=total_return,
            benchmark_return=benchmark_return,
            alpha=total_return - benchmark_return,
            start_date=data.index[0],
            end_date=data.index[-1],
            num_trades=len(trades),
        )

        self.logger.info(
            f"Backtest completed: {result.num_trades} trades, "
            f"return: {result.overall_return:.2%}, "
            f"win rate: {result.overall_win_rate:.2%}"
        )

        return result

    def _check_exit_conditions(
        self,
        position: float,
        entry_price: float,
        current_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> bool:
        """Check if position should be exited based on stop loss/take profit."""
        if position > 0:  # Long position
            if stop_loss and current_price <= entry_price * (1 - stop_loss):
                return True
            if take_profit and current_price >= entry_price * (1 + take_profit):
                return True
        else:  # Short position
            if stop_loss and current_price >= entry_price * (1 + stop_loss):
                return True
            if take_profit and current_price <= entry_price * (1 - take_profit):
                return True
        return False

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for a series of returns."""
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe ratio (assuming daily returns)
        return (avg_return / std_return) * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from a series of returns."""
        if not returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def compare_signals(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Compare multiple backtest results.

        Args:
            results: Dictionary of backtest results by strategy name

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for strategy_name, result in results.items():
            comparison_data.append(
                {
                    "Strategy": strategy_name,
                    "Total Return": result.overall_return,
                    "Win Rate": result.overall_win_rate,
                    "Benchmark Alpha": result.alpha,
                    "Num Trades": result.num_trades,
                    "Start Date": result.start_date,
                    "End Date": result.end_date,
                }
            )

        return pd.DataFrame(comparison_data)

    def get_signal_correlation(
        self, data: pd.DataFrame, signals: List[str]
    ) -> pd.DataFrame:
        """
        Calculate correlation between signals and future returns.

        Args:
            data: Historical data with features
            signals: List of signal names

        Returns:
            DataFrame with signal correlations
        """
        correlations = []

        for signal in signals:
            signal_strengths = []
            future_returns = []

            for i in range(len(data) - 1):
                row = data.iloc[i]
                sig_type, strength = self.signal_definitions.evaluate_signal(
                    signal, row.values, data.columns.tolist()
                )
                signal_strengths.append(strength)

                # Calculate future return (next period)
                future_price = data.iloc[i + 1]["close"]
                current_price = row["close"]
                future_return = (future_price - current_price) / current_price
                future_returns.append(future_return)

            if signal_strengths:
                corr = np.corrcoef(signal_strengths, future_returns)[0, 1]
                correlations.append(
                    {
                        "Signal": signal,
                        "Correlation": corr,
                        "Avg Strength": np.mean(signal_strengths),
                    }
                )

        return pd.DataFrame(correlations)
