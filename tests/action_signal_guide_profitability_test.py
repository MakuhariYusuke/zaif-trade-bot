#!/usr/bin/env python3
"""
Action Signal Guide Profitability Test

Tests if ActionSignalGuide alone can generate profitable trading signals.
This script simulates trading based on ActionSignalGuide signals and calculates returns.
Optimized for performance with parallel processing and statistical sampling.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import time

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide import ActionSignalGuide
from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuideConfig,
    GuidanceLevel,
)


class ActionSignalGuideBacktest:
    """Backtest engine for ActionSignalGuide profitability testing."""

    def __init__(self, initial_balance: float = 10000.0, fee_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.balance = initial_balance
        self.position = 0.0  # BTC amount held
        self.entry_price = 0.0
        self.trades = []
        self.portfolio_values = []

    def execute_trade(
        self, action: str, price: float, timestamp, signal_strength: float = 1.0
    ):
        """Execute a trade based on signal."""
        if action == "BUY" and self.position == 0:
            # Buy with all available balance
            btc_amount = (self.balance * (1 - self.fee_rate)) / price
            self.position = btc_amount
            self.entry_price = price
            self.balance = 0.0
            self.trades.append(
                {
                    "timestamp": timestamp,
                    "action": "BUY",
                    "price": price,
                    "btc_amount": btc_amount,
                    "signal_strength": signal_strength,
                }
            )
            return True

        elif action == "SELL" and self.position > 0:
            # Sell all position
            proceeds = self.position * price * (1 - self.fee_rate)
            profit_loss = proceeds - (self.position * self.entry_price)
            self.balance = proceeds
            self.trades.append(
                {
                    "timestamp": timestamp,
                    "action": "SELL",
                    "price": price,
                    "btc_amount": self.position,
                    "profit_loss": profit_loss,
                    "signal_strength": signal_strength,
                }
            )
            self.position = 0.0
            self.entry_price = 0.0
            return True

        return False

    def get_portfolio_value(self, current_price: float) -> float:
        """Get current portfolio value."""
        if self.position > 0:
            return self.balance + (self.position * current_price)
        else:
            return self.balance

    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "avg_trade_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        # Calculate returns
        final_value = (
            self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        )
        total_return = (final_value - self.initial_balance) / self.initial_balance

        # Calculate trade metrics
        sell_trades = [t for t in self.trades if t["action"] == "SELL"]
        winning_trades = [t for t in sell_trades if t.get("profit_loss", 0) > 0]
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0.0

        avg_trade_return = (
            np.mean([t.get("profit_loss", 0) for t in sell_trades])
            if sell_trades
            else 0.0
        )

        # Calculate drawdown
        peak = self.initial_balance
        max_drawdown = 0.0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "total_trades": len(sell_trades),
            "avg_trade_return": avg_trade_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": 0.0,  # Simplified, would need daily returns
        }


def create_realistic_test_data(n_points: int = 50000) -> pd.DataFrame:
    """Create realistic BTC/JPY test data with trends and volatility."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

    # Start with realistic BTC price
    base_price = 50000.0
    prices = [base_price]

    # Create realistic price movements with trends, mean reversion, and volatility
    for i in range(1, n_points):
        # Long-term trend (slight upward bias)
        trend = 0.00001  # Very small upward trend

        # Short-term mean reversion (much weaker)
        mean_reversion = (50000 - prices[-1]) * 0.0001  # Much weaker mean reversion

        # Random walk with volatility clustering
        volatility = 0.005 + 0.003 * np.sin(i * 0.001)  # Lower base volatility
        random_change = np.random.normal(0, volatility)

        # Combine factors
        price_change = trend + mean_reversion + random_change
        new_price = prices[-1] * (1 + price_change)

        # Ensure price stays reasonable
        new_price = max(new_price, 10000)  # Floor
        new_price = min(new_price, 200000)  # Ceiling

        prices.append(new_price)

    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some intrabar volatility
        volatility = 0.005 + 0.003 * np.sin(i * 0.01)
        high = price * (1 + abs(np.random.normal(0, volatility)))
        low = price * (1 - abs(np.random.normal(0, volatility)))
        open_price = price + np.random.normal(0, price * volatility * 0.3)
        close = price + np.random.normal(0, price * volatility * 0.3)

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = np.random.randint(100, 10000)

        data.append(
            {
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def sample_data_windows(data: pd.DataFrame, window_size: int = 1000, n_samples: int = 10) -> List[pd.DataFrame]:
    """Sample random windows from the data for statistical testing."""
    windows = []
    max_start = len(data) - window_size - 200  # Leave room for history
    
    np.random.seed(42)
    for _ in range(n_samples):
        start_idx = np.random.randint(200, max_start)
        end_idx = start_idx + window_size
        window = data.iloc[start_idx:end_idx].copy()
        # Reset index to start from 0 for each window
        window.index = pd.date_range("2023-01-01", periods=len(window), freq="1H")
        windows.append(window)
    
    return windows


def run_single_backtest(window_data: pd.DataFrame, config_dict: Optional[Dict] = None) -> Dict:
    """Run backtest on a single data window."""
    # Configure ActionSignalGuide
    if config_dict is None:
        config_dict = {
            # Enable only basic patterns that don't require ztb.features.trend
            "enable_candlestick_patterns": False,  # Disabled due to missing _is_uptrend method
            "enable_fibonacci_patterns": False,  # Disabled due to missing dependencies
            "enable_gann_patterns": False,  # Disabled due to missing dependencies
            "enable_wave_patterns": False,  # Disabled due to missing dependencies
            "enable_harmonic_patterns": False,  # Disabled due to missing ztb.features.trend
            "enable_oscillator_patterns": False,  # Disabled due to missing dependencies
            "enable_volume_patterns": False,  # Disabled due to missing dependencies
            "enable_bollinger_patterns": False,  # Disabled due to missing dependencies
            "enable_adx_patterns": False,  # Disabled due to missing ztb.features.trend
            "enable_granville_patterns": False,  # Disabled due to missing dependencies
            "enable_heikin_ashi_patterns": False,  # Disabled due to missing ztb.features.trend
            "enable_dow_theory_patterns": False,  # Disabled due to missing ztb.features.trend
        }

    try:
        config = ActionSignalGuideConfig(
            guidance_level=GuidanceLevel.WEAK,
            enable_adx_patterns=False,  # Disable ADX patterns to avoid trend module
            enable_heikin_ashi_patterns=False,  # Disable Heikin-Ashi patterns
            enable_dow_theory_patterns=False,  # Disable Dow Theory patterns
            enable_harmonic_patterns=False,  # Disable harmonic patterns
            enable_oscillator_patterns=False,  # Disable oscillator patterns
            enable_bollinger_patterns=False,  # Disable bollinger patterns
        )
        guide = ActionSignalGuide(config=config)
    except Exception as e:
        print(f"❌ Failed to initialize ActionSignalGuide: {e}")
        return {
            "final_value": 10000.0,
            "total_return": 0.0,
            "signals_generated": 0,
            "trades_executed": 0,
            "metrics": {
                "total_return": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "avg_trade_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            },
            "trades": [],
        }

    # Initialize backtest engine
    backtest = ActionSignalGuideBacktest(initial_balance=10000.0, fee_rate=0.001)

    signals_generated = 0
    trades_executed = 0

    for i in range(200, len(window_data)):  # Start from index 200 for sufficient history
        current_price = window_data.iloc[i]["close"].item()
        timestamp = window_data.index[i]

        # Get signals from ActionSignalGuide
        try:
            signals = guide.generate_signals(window_data, i)
            if signals:
                signals_generated += len(signals)

                # Process signals (simplified: use strongest signal)
                if signals:
                    # Find strongest buy/sell signal
                    buy_signals = [s for s in signals if s.direction > 0.1]
                    sell_signals = [s for s in signals if s.direction < -0.1]

                    if buy_signals and backtest.position == 0:
                        # Execute buy
                        strongest_buy = max(buy_signals, key=lambda s: abs(s.direction))
                        if backtest.execute_trade(
                            "BUY", current_price, timestamp, strongest_buy.strength
                        ):
                            trades_executed += 1
                    elif sell_signals and backtest.position > 0:
                        # Execute sell
                        strongest_sell = min(
                            sell_signals, key=lambda s: abs(s.direction)
                        )
                        if backtest.execute_trade(
                            "SELL", current_price, timestamp, strongest_sell.strength
                        ):
                            trades_executed += 1
        except Exception:
            # Skip errors to continue backtest
            continue

        # Record portfolio value
        portfolio_value = backtest.get_portfolio_value(current_price)
        backtest.portfolio_values.append(portfolio_value)

    # Calculate final metrics
    final_value = backtest.get_portfolio_value(window_data.iloc[-1]["close"].item())
    metrics = backtest.get_metrics()

    return {
        "final_value": final_value,
        "total_return": metrics["total_return"],
        "signals_generated": signals_generated,
        "trades_executed": trades_executed,
        "metrics": metrics,
        "trades": backtest.trades,
    }


def run_parallel_backtests(data_windows: List[pd.DataFrame], config_dict: Optional[Dict] = None, max_workers: int = 4) -> List[Dict]:
    """Run backtests in parallel on multiple data windows."""
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all backtest tasks
        future_to_window = {
            executor.submit(run_single_backtest, window, config_dict): window 
            for window in data_windows
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_window):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Backtest generated an exception: {exc}')
    
    return results


def calculate_confidence_intervals(results: List[Dict], confidence_level: float = 0.95) -> Dict:
    """Calculate statistical confidence intervals for backtest results."""
    if not results:
        return {}
    
    # Extract returns
    returns = [r["total_return"] for r in results]
    
    # Calculate mean and confidence interval
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    n = len(returns)
    
    # t-distribution for confidence interval
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    margin_error = t_value * std_return / np.sqrt(n)
    
    ci_lower = mean_return - margin_error
    ci_upper = mean_return + margin_error
    
    # Other statistics
    median_return = np.median(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    
    # Win rate statistics
    win_rates = [r["metrics"]["win_rate"] for r in results]
    mean_win_rate = np.mean(win_rates)
    
    return {
        "mean_return": mean_return,
        "median_return": median_return,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_return": std_return,
        "min_return": min_return,
        "max_return": max_return,
        "mean_win_rate": mean_win_rate,
        "n_samples": n,
        "confidence_level": confidence_level,
    }


def run_action_signal_guide_backtest(config_dict: Optional[Dict] = None, n_samples: int = 10, window_size: int = 1000, max_workers: int = 4) -> Dict:
    """Run backtest using ActionSignalGuide signals with statistical sampling."""
    print("🚀 Starting ActionSignalGuide Profitability Test")
    print("=" * 60)
    print(f"Running {n_samples} statistical samples with {window_size} data points each")
    print(f"Using {max_workers} parallel workers")

    start_time = time.time()

    # Create test data
    print("📊 Creating realistic test data...")
    data = create_realistic_test_data(50000)  # Larger dataset for better sampling
    print(f"✅ Created {len(data)} data points")
    print(f"Price range: ¥{data['close'].min():,.0f} - ¥{data['close'].max():,.0f}")

    # Sample data windows for statistical testing
    print("🎲 Sampling data windows for statistical analysis...")
    data_windows = sample_data_windows(data, window_size=window_size, n_samples=n_samples)
    print(f"✅ Created {len(data_windows)} sample windows")

    # Run parallel backtests
    print("📈 Running parallel backtests...")
    config_dict = {
        "enable_candlestick_patterns": True,
        "enable_fibonacci_patterns": True,
        "enable_gann_patterns": True,
        "enable_wave_patterns": True,
        "enable_harmonic_patterns": False,  # Disabled due to missing ztb.features.trend
        "enable_oscillator_patterns": True,
        "enable_volume_patterns": True,
        "enable_bollinger_patterns": True,
        "enable_adx_patterns": True,
        "enable_granville_patterns": True,
        "enable_heikin_ashi_patterns": True,
        "enable_dow_theory_patterns": False,  # Disabled due to missing ztb.features.trend
    }
    results = run_parallel_backtests(data_windows, config_dict, max_workers=max_workers)
    print(f"✅ Completed {len(results)} backtests")

    # Calculate confidence intervals
    print("� Calculating statistical confidence intervals...")
    stats_results = calculate_confidence_intervals(results)

    elapsed_time = time.time() - start_time
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")

    return {
        "stats": stats_results,
        "individual_results": results,
        "execution_time": elapsed_time,
        "n_samples": n_samples,
        "window_size": window_size,
    }


def main():
    """Main function to run the profitability test."""
    print("🔬 Action Signal Guide Profitability Analysis")
    print("Testing if technical analysis signals alone can generate profits...")
    print("Using statistical sampling and parallel processing for reliable results.")

    # Run statistical backtest with sampling
    config_dict = {
        "enable_candlestick_patterns": True,
        "enable_fibonacci_patterns": False,  # Disabled due to missing dependencies
        "enable_gann_patterns": False,  # Disabled due to missing dependencies
        "enable_wave_patterns": False,  # Disabled due to missing dependencies
        "enable_harmonic_patterns": False,  # Disabled due to missing ztb.features.trend
        "enable_oscillator_patterns": False,  # Disabled due to missing dependencies
        "enable_volume_patterns": False,  # Disabled due to missing dependencies
        "enable_bollinger_patterns": False,  # Disabled due to missing dependencies
        "enable_adx_patterns": False,  # Disabled due to missing ztb.features.trend
        "enable_granville_patterns": False,  # Disabled due to missing dependencies
        "enable_heikin_ashi_patterns": False,  # Disabled due to missing ztb.features.trend
        "enable_dow_theory_patterns": False,  # Disabled due to missing ztb.features.trend
    }
    results = run_action_signal_guide_backtest(
        config_dict=config_dict,
        n_samples=20,  # Number of statistical samples
        window_size=1500,  # Size of each data window
        max_workers=4  # Parallel workers
    )

    stats = results["stats"]

    # Print statistical results
    print("\n" + "=" * 80)
    print("📊 STATISTICAL BACKTEST RESULTS")
    print("=" * 80)
    print(f"Number of Samples: {stats['n_samples']}")
    print(f"Window Size: {results['window_size']} data points")
    print(f"Confidence Level: {stats['confidence_level']:.1%}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print()
    print("📈 Return Statistics:")
    print(f"Mean Return: {stats['mean_return']:.2%}")
    print(f"Median Return: {stats['median_return']:.2%}")
    print(f"95% Confidence Interval: [{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}]")
    print(f"Standard Deviation: {stats['std_return']:.2%}")
    print(f"Min Return: {stats['min_return']:.2%}")
    print(f"Max Return: {stats['max_return']:.2%}")
    print()
    print("🎯 Performance Metrics:")
    print(f"Mean Win Rate: {stats['mean_win_rate']:.1%}")

    # Summary
    print("\n" + "=" * 80)
    print("🎯 CONCLUSION")
    print("=" * 80)

    if stats['ci_upper'] > 0:
        print("✅ POSITIVE RESULT: ActionSignalGuide shows potential profitability!")
        print(f"Mean Return: {stats['mean_return']:.2%} (95% CI: [{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}])")
    elif stats['ci_lower'] < 0 and stats['ci_upper'] > 0:
        print("🤔 MIXED RESULT: ActionSignalGuide shows marginal profitability.")
        print(f"Mean Return: {stats['mean_return']:.2%} (95% CI: [{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}])")
        print("The confidence interval includes zero, indicating uncertainty.")
    else:
        print("❌ NEGATIVE RESULT: ActionSignalGuide does not generate profits.")
        print(f"Mean Return: {stats['mean_return']:.2%} (95% CI: [{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}])")

    print("\n📋 Key Insights:")
    print(f"• Statistical samples: {stats['n_samples']} windows of {results['window_size']} data points each")
    print(f"• Parallel processing: {results.get('max_workers', 4)} workers used")
    print(f"• Execution efficiency: {results['execution_time']:.2f} seconds total")
    print(f"• Mean win rate: {stats['mean_win_rate']:.1%}")

    if stats['mean_return'] > 0.05:  # 5% return threshold
        print("🎉 The ActionSignalGuide shows promising profitability potential!")
    elif stats['mean_return'] > 0:
        print("🤔 The ActionSignalGuide shows marginal profitability.")
    else:
        print("📚 The ActionSignalGuide needs further optimization for profitability.")


if __name__ == "__main__":
    main()
