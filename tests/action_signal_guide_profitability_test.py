#!/usr/bin/env python3
"""
Action Signal Guide Profitability Test

Tests if ActionSignalGuide alone can generate profitable trading signals.
This script simulates trading based on ActionSignalGuide signals and calculates returns.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
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


def create_realistic_test_data(n_points: int = 10000) -> pd.DataFrame:
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


def run_action_signal_guide_backtest(config_dict: Optional[Dict] = None) -> Dict:
    """Run backtest using ActionSignalGuide signals."""
    print("🚀 Starting ActionSignalGuide Profitability Test")
    print("=" * 60)

    # Create test data
    print("📊 Creating realistic test data...")
    data = create_realistic_test_data(5000)
    print(f"✅ Created {len(data)} data points")
    print(f"Price range: ¥{data['close'].min():,.0f} - ¥{data['close'].max():,.0f}")
    # Configure ActionSignalGuide
    if config_dict is None:
        config_dict = {
            # Enable all patterns for comprehensive testing
            "enable_candlestick_patterns": True,
            "enable_fibonacci_patterns": True,
            "enable_gann_patterns": True,
            "enable_wave_patterns": True,
            "enable_harmonic_patterns": True,
            "enable_oscillator_patterns": True,
            "enable_volume_patterns": True,
            "enable_bollinger_patterns": True,
            "enable_adx_patterns": True,
            "enable_granville_patterns": True,
            "enable_heikin_ashi_patterns": True,
            "enable_dow_theory_patterns": True,
        }

    config = ActionSignalGuideConfig(
        debug_short_mode=False,
        guidance_level=GuidanceLevel.WEAK,
        error_suppression_threshold=0,
        **config_dict,
    )

    print("🎯 Initializing ActionSignalGuide...")
    guide = ActionSignalGuide(config=config)
    print(f"✅ Initialized with {len(guide.all_recognizers)} recognizers")

    # Initialize backtest engine
    backtest = ActionSignalGuideBacktest(initial_balance=10000.0, fee_rate=0.001)

    # Run backtest
    print("📈 Running backtest...")
    signals_generated = 0
    trades_executed = 0

    for i in range(200, len(data)):  # Start from index 200 for sufficient history
        current_price = data.iloc[i]["close"]
        timestamp = data.index[i]

        # Get signals from ActionSignalGuide
        try:
            signals = guide.generate_signals(data, i)
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
                            print(
                                f"🛒 BUY at ¥{current_price:,.0f} (strength: {strongest_buy.strength:.3f})"
                            )
                    elif sell_signals and backtest.position > 0:
                        # Execute sell
                        strongest_sell = min(
                            sell_signals, key=lambda s: abs(s.direction)
                        )
                        if backtest.execute_trade(
                            "SELL", current_price, timestamp, strongest_sell.strength
                        ):
                            trades_executed += 1
                            print(
                                f"📤 SELL at ¥{current_price:,.0f} (strength: {strongest_sell.strength:.3f})"
                            )
        except Exception:
            # Skip errors to continue backtest
            continue

        # Record portfolio value
        portfolio_value = backtest.get_portfolio_value(current_price)
        backtest.portfolio_values.append(portfolio_value)

        # Progress indicator
        if i % 500 == 0:
            progress = (i - 200) / (len(data) - 200) * 100
            current_value = backtest.get_portfolio_value(current_price)
            print(f"Progress: {progress:.1f}% | Portfolio: ¥{current_value:,.0f}")
    # Calculate final metrics
    final_value = backtest.get_portfolio_value(data.iloc[-1]["close"])
    metrics = backtest.get_metrics()

    # Print results
    print("\n" + "=" * 60)
    print("📊 ACTION SIGNAL GUIDE BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Balance: ¥{backtest.initial_balance:,.0f}")
    print(f"Final Value: ¥{final_value:,.0f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"Total Signals Generated: {signals_generated}")
    print(f"Trades Executed: {trades_executed}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Average Trade Return: ¥{metrics['avg_trade_return']:,.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    # Trade analysis
    if backtest.trades:
        sell_trades = [t for t in backtest.trades if t["action"] == "SELL"]
        if sell_trades:
            profits = [t.get("profit_loss", 0) for t in sell_trades]
            print("\n💼 Trade Analysis:")
            print(f"Total Trades: {len(sell_trades)}")
            print(f"Winning Trades: {len([p for p in profits if p > 0])}")
            print(f"Losing Trades: {len([p for p in profits if p <= 0])}")
            print(f"Win Rate: {len([p for p in profits if p > 0]) / len(profits):.1%}")
            if profits:
                print(f"Largest Win: ¥{max(profits):,.2f}")
                print(f"Largest Loss: ¥{min(profits):,.2f}")
                print(f"Average Win: ¥{np.mean([p for p in profits if p > 0]):,.2f}")
                print(f"Average Loss: ¥{np.mean([p for p in profits if p <= 0]):,.2f}")
    return {
        "final_value": final_value,
        "total_return": metrics["total_return"],
        "signals_generated": signals_generated,
        "trades_executed": trades_executed,
        "metrics": metrics,
        "trades": backtest.trades,
    }


def main():
    """Main function to run the profitability test."""
    print("🔬 Action Signal Guide Profitability Analysis")
    print("Testing if technical analysis signals alone can generate profits...")

    # Run backtest with all patterns enabled
    results = run_action_signal_guide_backtest()

    # Summary
    print("\n" + "=" * 80)
    print("🎯 CONCLUSION")
    print("=" * 80)

    if results["total_return"] > 0:
        print("✅ POSITIVE RESULT: ActionSignalGuide generated profitable signals!")
        print(f"Total Return: {results['total_return']:.1%}")
    else:
        print("❌ NEGATIVE RESULT: ActionSignalGuide did not generate profits.")
        print(f"Total Return: {results['total_return']:.1%}")
    print("\n📋 Key Insights:")
    print(
        f"• Signals generated per data point: {results['signals_generated'] / 4800:.3f}"
    )
    print(
        f"• Trading frequency: {results['trades_executed'] / 4800:.3f} trades per data point"
    )
    print(f"• Win rate: {results['metrics']['win_rate']:.1%}")

    if results["total_return"] > 0.05:  # 5% return threshold
        print("🎉 The ActionSignalGuide shows promising profitability potential!")
    elif results["total_return"] > 0:
        print("🤔 The ActionSignalGuide shows marginal profitability.")
    else:
        print("📚 The ActionSignalGuide needs further optimization for profitability.")


if __name__ == "__main__":
    main()
