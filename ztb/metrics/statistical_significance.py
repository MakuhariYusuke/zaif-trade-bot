"""
Statistical significance tests for trading strategies.

Includes Deflated Sharpe Ratio (DSR), Probabilistic Sharpe Ratio (PSR),
and Bootstrap p-value calculations.
"""

from typing import List, Union

import numpy as np
import pandas as pd

from ztb.metrics.metrics import sharpe_ratio
from ztb.trading.constants import TRADING_DAYS_PER_YEAR


def calculate_deflated_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    num_strategies: int = 1000,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR) to account for multiple testing.

    This is a simplified version that adjusts the Sharpe Ratio based on the
    number of trials (strategies tested).

    Args:
        returns: Return series
        num_strategies: Number of independent strategies tested (trials)
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Deflated Sharpe Ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)

    # Calculate standard Sharpe Ratio
    sr = sharpe_ratio(returns, rf=risk_free_rate, period_per_year=TRADING_DAYS_PER_YEAR)

    # Deflate by number of strategies tested (simplified Bonferroni-like adjustment)
    # A more rigorous DSR requires track record length and skewness/kurtosis
    # For now, we use the simplified version from the existing codebase
    deflation_factor = 1.0 / np.sqrt(num_strategies) if num_strategies > 0 else 1.0

    return float(sr * deflation_factor)


def calculate_bootstrap_pvalue(
    strategy_returns: Union[pd.Series, np.ndarray, List[float]],
    benchmark_returns: Union[pd.Series, np.ndarray, List[float]],
    n_bootstrap: int = 1000,
) -> float:
    """
    Calculate bootstrap p-value for strategy vs benchmark comparison.

    Tests the null hypothesis that the strategy returns are not significantly
    different from the benchmark returns.

    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series
        n_bootstrap: Number of bootstrap samples

    Returns:
        p-value (probability that difference is due to chance)
    """
    if isinstance(strategy_returns, list):
        strategy_returns = np.array(strategy_returns)
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)

    if len(strategy_returns) != len(benchmark_returns):
        # Truncate to shorter length if necessary, or raise error
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

    # Calculate observed difference in means
    observed_diff = np.mean(strategy_returns) - np.mean(benchmark_returns)

    # Bootstrap resampling
    # We combine the distributions under the null hypothesis that they are the same
    combined = np.concatenate([strategy_returns, benchmark_returns])
    n = len(strategy_returns)

    bootstrap_diffs = []

    # Vectorized bootstrap if possible, but loop is safer for memory with large n_bootstrap
    for _ in range(n_bootstrap):
        # Resample from combined distribution
        strat_sample = np.random.choice(combined, size=n, replace=True)
        bench_sample = np.random.choice(combined, size=n, replace=True)

        diff = np.mean(strat_sample) - np.mean(bench_sample)
        bootstrap_diffs.append(diff)

    # Calculate p-value (two-tailed)
    # Proportion of bootstrap differences that are more extreme than observed difference
    bootstrap_array = np.array(bootstrap_diffs)
    p_value = np.mean(np.abs(bootstrap_array) >= np.abs(observed_diff))

    return float(p_value)
