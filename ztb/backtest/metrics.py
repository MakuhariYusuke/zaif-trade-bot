"""
Backtest metrics calculation module.

Provides comprehensive performance metrics for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Return metrics
    total_return: float
    cagr: float
    annualized_return: float

    # Risk metrics
    max_drawdown: float
    volatility: float

    # Trade metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    turnover: float

    # Slippage estimate
    estimated_slippage_bps: float

    # Statistical significance (optional)
    deflated_sharpe_ratio: Optional[float] = None
    pvalue_bootstrap: Optional[float] = None


class MetricsCalculator:
    """Calculates comprehensive trading performance metrics."""

    @staticmethod
    def calculate_returns(equity_curve: pd.Series, freq: str = 'D') -> pd.Series:
        """Calculate periodic returns from equity curve."""
        if freq == 'D':
            # Daily returns
            returns = equity_curve.pct_change().fillna(0)
        else:
            # For other frequencies, resample
            returns = equity_curve.resample(freq).last().pct_change().fillna(0)

        return returns

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return float(sharpe)

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return float(sortino)

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min()
        return float(max_dd)

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_dd: float) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if max_dd >= 0:  # No drawdown
            return 0.0

        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        if years == 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1
        calmar = annualized_return / abs(max_dd)
        return float(calmar)

    @staticmethod
    def calculate_cagr(equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(equity_curve) < 2:
            return 0.0

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        years = len(equity_curve) / 252  # Assuming daily data

        if years <= 0 or total_return <= -1:
            return 0.0

        cagr = (1 + total_return) ** (1 / years) - 1
        return float(cagr)

    @staticmethod
    def calculate_trade_metrics(orders: pd.DataFrame) -> Tuple[int, float, float, float, float]:
        """
        Calculate trade-level metrics.

        Returns: (total_trades, win_rate, avg_win, avg_loss, profit_factor)
        """
        if orders.empty:
            return 0, 0.0, 0.0, 0.0, 0.0

        # Assume orders has 'pnl' column with profit/loss per trade
        if 'pnl' not in orders.columns:
            return len(orders), 0.0, 0.0, 0.0, 0.0

        pnls = orders['pnl'].dropna()
        if len(pnls) == 0:
            return len(orders), 0.0, 0.0, 0.0, 0.0

        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]

        total_trades = len(pnls)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0.0

        total_win = winning_trades.sum()
        total_loss = abs(losing_trades.sum())
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

        return total_trades, win_rate, avg_win, avg_loss, profit_factor

    @staticmethod
    def estimate_turnover(orders: pd.DataFrame, initial_capital: float = 10000) -> float:
        """Estimate portfolio turnover (annualized)."""
        if orders.empty or 'notional' not in orders.columns:
            return 0.0

        # Sum of absolute notional values
        total_turnover = orders['notional'].abs().sum()

        # Annualize (assuming daily data)
        days = len(orders) if len(orders) > 0 else 1
        annualized_turnover = total_turnover / initial_capital * (252 / days)

        return float(annualized_turnover)

    @staticmethod
    def estimate_slippage_bps(orders: pd.DataFrame, slippage_bps: float = 5.0) -> float:
        """Estimate slippage impact in basis points."""
        # For now, return the configured slippage
        # Could be enhanced to calculate based on order book data
        return slippage_bps

    @classmethod
    def calculate_all_metrics(
        cls,
        equity_curve: pd.Series,
        orders: pd.DataFrame,
        initial_capital: float = 10000,
        risk_free_rate: float = 0.02,
        slippage_bps: float = 5.0
    ) -> BacktestMetrics:
        """Calculate all performance metrics."""

        returns = cls.calculate_returns(equity_curve)

        sharpe = cls.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = cls.calculate_sortino_ratio(returns, risk_free_rate)
        max_dd = cls.calculate_max_drawdown(equity_curve)
        calmar = cls.calculate_calmar_ratio(returns, max_dd)
        cagr = cls.calculate_cagr(equity_curve)

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0
        volatility = returns.std() * np.sqrt(252)  # Annualized

        total_trades, win_rate, avg_win, avg_loss, profit_factor = cls.calculate_trade_metrics(orders)
        turnover = cls.estimate_turnover(orders, initial_capital)
        estimated_slippage = cls.estimate_slippage_bps(orders, slippage_bps)

        deflated_sharpe = cls.calculate_deflated_sharpe_ratio(returns)
        pvalue_bootstrap = cls.calculate_bootstrap_pvalue(returns, benchmark_returns=returns * 0.5)

        return BacktestMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_return=total_return,
            cagr=cagr,
            annualized_return=cagr,  # Same as CAGR for consistency
            max_drawdown=max_dd,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            turnover=turnover,
            estimated_slippage_bps=estimated_slippage,
            deflated_sharpe_ratio=deflated_sharpe,
            pvalue_bootstrap=pvalue_bootstrap
        )

    @staticmethod
    def calculate_deflated_sharpe_ratio(returns: pd.Series, num_strategies: int = 1000) -> float:
        """Calculate deflated Sharpe ratio to account for multiple testing."""
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)

        # Deflate by number of strategies tested (simplified)
        # In practice, this would be more sophisticated
        deflation_factor = 1.0 / np.sqrt(num_strategies)
        return sharpe * deflation_factor

    @staticmethod
    def calculate_bootstrap_pvalue(strategy_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 num_bootstrap: int = 1000) -> float:
        """Calculate bootstrap p-value for strategy vs benchmark comparison."""
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Strategy and benchmark returns must have same length")

        # Calculate observed difference
        observed_diff = strategy_returns.mean() - benchmark_returns.mean()

        # Bootstrap resampling
        combined = pd.concat([strategy_returns, benchmark_returns], ignore_index=True)
        bootstrap_diffs = []

        for _ in range(num_bootstrap):
            # Resample with replacement
            strat_sample = combined.sample(n=len(strategy_returns), replace=True)
            bench_sample = combined.sample(n=len(benchmark_returns), replace=True)

            diff = strat_sample.mean() - bench_sample.mean()
            bootstrap_diffs.append(diff)

        # Calculate p-value (two-tailed)
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        return float(p_value)