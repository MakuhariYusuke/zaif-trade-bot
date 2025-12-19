#!/usr/bin/env python3
"""
Test improved SignalQualityScorer with backtest simulation
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from ztb.trading.signal.quality_scorer import SignalQualityScorer

def create_test_market_data():
    """Create realistic market data for testing"""
    np.random.seed(42)

    # Generate 5000 periods of 1-hour data (about 7 months) for more comprehensive testing
    dates = pd.date_range('2024-01-01', periods=5000, freq='1H')

    # Create realistic price movements with trends and volatility
    base_price = 100.0

    # Add trend component - MIXED trends (up and down periods)
    # Create random trend segments instead of linear upward trend
    trend_segments = []
    segment_length = 200  # 200 periods per segment
    for i in range(25):  # 25 segments for 5000 periods
        trend_direction = np.random.choice([-1, 1])  # Random up or down
        trend_magnitude = np.random.uniform(0.02, 0.08)  # Random magnitude
        segment_trend = np.linspace(0, trend_direction * trend_magnitude, segment_length)
        trend_segments.extend(segment_trend)
    trend = np.array(trend_segments)

    # Add cyclical component (market cycles) - multiple frequencies
    weekly_cycle = 0.03 * np.sin(2 * np.pi * np.arange(5000) / 168)  # Weekly cycle
    monthly_cycle = 0.02 * np.sin(2 * np.pi * np.arange(5000) / 720)  # Monthly cycle
    cycle = weekly_cycle + monthly_cycle

    # Add random noise with volatility clustering (more realistic)
    # Volatility changes over time
    vol_trend = 0.01 + 0.005 * np.sin(2 * np.pi * np.arange(5000) / 500)  # Changing volatility
    volatility = vol_trend + 0.005 * np.random.exponential(1, 5000)
    noise = np.random.normal(0, volatility)

    # Combine components
    price_changes = trend + cycle + noise
    prices = base_price * np.cumprod(1 + price_changes)

    # Create OHLCV data with more realistic spreads
    spread_mult = np.random.uniform(0.001, 0.005, 5000)  # Realistic spreads
    high_mult = 1 + spread_mult
    low_mult = 1 - spread_mult

    # Add some gaps and jumps (market events)
    gap_indices = np.random.choice(5000, size=50, replace=False)  # 50 random gaps
    for idx in gap_indices:
        if idx > 0:
            gap_size = np.random.normal(0, 0.02)  # 2% average gap
            prices[idx:] *= (1 + gap_size)

    volume_base = 100000
    volume = volume_base + np.random.normal(0, 20000, 5000)
    volume = np.maximum(volume, 10000)  # Minimum volume

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, 5000)),  # Small gap between open/close
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': volume
    })

    return df

def run_signal_quality_backtest(df: pd.DataFrame, scorer: SignalQualityScorer):
    """Run backtest using SignalQualityScorer"""
    capital = 1000000  # 1M JPY
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    trades = []
    equity_curve = [capital]

    # Risk management parameters - More realistic settings
    max_position_size = 0.05  # 5% of capital per trade (increased for more impact)
    stop_loss_pct = 0.005  # 0.5% stop loss (very tight)
    take_profit_pct = 0.01  # 1% take profit (tight)

    # Debug: Track score distribution
    score_counts = {'buy_signals': 0, 'sell_signals': 0, 'hold_signals': 0}
    score_ranges = {'0-25': 0, '26-45': 0, '46-55': 0, '56-75': 0, '76-100': 0}

    portfolio = {
        'btc_balance': 0.0,
        'jpy_balance': capital,
        'current_price': 0.0
    }

    # Wait for sufficient data for technical indicators (50 periods minimum)
    start_index = 50

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        current_price = row['close']
        portfolio['current_price'] = current_price

        # Calculate BTC balance for position sizing
        if position == 1:
            portfolio['btc_balance'] = (capital * max_position_size) / current_price
        elif position == -1:
            portfolio['btc_balance'] = (capital * max_position_size) / current_price
        else:
            portfolio['btc_balance'] = 0.0

        # Get signal quality score (use data up to current point)
        current_data = df.iloc[:i+1]
        continuous_action = 0.0  # Neutral baseline
        discrete_action, quality_score = scorer.calculate_signal_quality(
            current_data, continuous_action, portfolio
        )

        # Debug: Count score distribution
        if discrete_action == 1:
            score_counts['buy_signals'] += 1
        elif discrete_action == -1:
            score_counts['sell_signals'] += 1
        else:
            score_counts['hold_signals'] += 1

        if quality_score <= 25:
            score_ranges['0-25'] += 1
        elif quality_score <= 45:
            score_ranges['26-45'] += 1
        elif quality_score <= 55:
            score_ranges['46-55'] += 1
        elif quality_score <= 75:
            score_ranges['56-75'] += 1
        else:
            score_ranges['76-100'] += 1

        # Trading logic - only trade if score is strong enough
        signal_threshold = 70  # Slightly lower threshold for more signals

        if discrete_action == 1 and position == 0 and quality_score >= signal_threshold:  # BUY signal
            position = 1
            entry_price = current_price
            trade_size = capital * max_position_size

        elif discrete_action == -1 and position == 0 and quality_score <= (100 - signal_threshold):  # SELL signal
            position = -1
            entry_price = current_price
            trade_size = capital * max_position_size

        elif ((discrete_action == -1 and position == 1) or
              (discrete_action == 1 and position == -1)):  # Close position on opposite signal
            # Calculate P&L
            if position == 1:  # Closing long
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Closing short
                pnl_pct = (entry_price - current_price) / entry_price

            capital *= (1 + pnl_pct * max_position_size)

            trades.append({
                'entry_time': df.iloc[i-1]['timestamp'] if i > 0 else df.iloc[i]['timestamp'],
                'exit_time': row['timestamp'],
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_pct': pnl_pct,
                'type': 'long' if position == 1 else 'short',
                'exit_reason': 'signal'
            })

            position = 0
            entry_price = 0

        # Check stop loss / take profit
        if position != 0:
            if position == 1:  # Long position
                current_pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                current_pnl_pct = (entry_price - current_price) / entry_price

            if current_pnl_pct <= -stop_loss_pct or current_pnl_pct >= take_profit_pct:
                capital *= (1 + current_pnl_pct * max_position_size)

                trades.append({
                    'entry_time': df.iloc[i-1]['timestamp'] if i > 0 else df.iloc[i]['timestamp'],
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': current_pnl_pct,
                    'type': 'long' if position == 1 else 'short',
                    'exit_reason': 'stop_loss' if current_pnl_pct <= -stop_loss_pct else 'take_profit'
                })

                position = 0
                entry_price = 0

        equity_curve.append(capital)

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_capital': capital,
        'total_return': (capital - 1000000) / 1000000,
        'score_counts': score_counts,
        'score_ranges': score_ranges
    }

def calculate_metrics(backtest_result):
    """Calculate trading metrics"""
    trades = backtest_result['trades']
    equity_curve = backtest_result['equity_curve']

    if not trades:
        return {
            'total_return': 0,
            'win_rate': 0,
            'total_trades': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0
        }

    # Basic metrics
    total_return = backtest_result['total_return']
    winning_trades = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(winning_trades) / len(trades)

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe ratio (simplified)
    returns = equity_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0

    # Profit factor
    gross_profit = sum(t['pnl_pct'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'total_trades': len(trades),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor
    }

def main():
    """Main test function"""
    print("🧪 Testing Improved SignalQualityScorer Backtest")
    print("=" * 60)

    # Create test data
    print("📊 Generating test market data...")
    df = create_test_market_data()
    print(f"Generated {len(df)} data points")

    # Initialize improved scorer
    print("🎯 Initializing improved SignalQualityScorer...")
    config = {
        'buy_threshold': 75,   # Lower threshold for BUY signals
        'sell_threshold': 25,  # Higher threshold for SELL signals
        'hold_threshold': 45,
        'trend_window': 10     # Add missing trend_window parameter
    }
    scorer = SignalQualityScorer(config)

    # Run backtest
    print("🚀 Running backtest simulation...")
    result = run_signal_quality_backtest(df, scorer)

    # Calculate metrics
    metrics = calculate_metrics(result)

    # Print results
    print("\n📈 BACKTEST RESULTS")
    print("-" * 40)
    print(".2%")
    print(".1%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(".2%")
    print(".2f")
    print(".2f")

    # Trade analysis
    trades = result['trades']
    if trades:
        print(f"\n📊 TRADE ANALYSIS ({len(trades)} trades)")
        print("-" * 40)

        long_trades = [t for t in trades if t['type'] == 'long']
        short_trades = [t for t in trades if t['type'] == 'short']

        print(f"Long trades: {len(long_trades)}")
        print(f"Short trades: {len(short_trades)}")

        if long_trades:
            long_win_rate = len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades)
            print(".1%")

        if short_trades:
            short_win_rate = len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades)
            print(".1%")

        # Exit reasons
        stop_losses = len([t for t in trades if t['exit_reason'] == 'stop_loss'])
        take_profits = len([t for t in trades if t['exit_reason'] == 'take_profit'])
        signals = len([t for t in trades if t['exit_reason'] == 'signal'])

        print(f"Exits by stop loss: {stop_losses}")
        print(f"Exits by take profit: {take_profits}")
        print(f"Exits by signal: {signals}")

    print("\n✅ Backtest completed successfully!")
    print(f"Final capital: ¥{result['final_capital']:,.0f}")

def main():
    """Main test function"""
    print("🧪 Testing Improved SignalQualityScorer Backtest")
    print("=" * 60)

    # Create test data
    print("📊 Generating test market data...")
    df = create_test_market_data()
    print(f"Generated {len(df)} data points")

    # Initialize improved scorer with adjusted thresholds for more signals
    config = {
        'buy_threshold': 65,   # Lower threshold for BUY signals (more signals)
        'sell_threshold': 35,  # Higher threshold for SELL signals (more signals)
        'hold_threshold': 45
    }
    print("🎯 Initializing improved SignalQualityScorer...")
    scorer = SignalQualityScorer(config)

    # Run backtest
    print("🚀 Running backtest simulation...")
    result = run_signal_quality_backtest(df, scorer)

    # Calculate metrics
    metrics = calculate_metrics(result)

    # Print results
    print("\n📈 BACKTEST RESULTS")
    print("-" * 40)
    print(".2%")
    print(".1%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(".2%")
    print(".2f")
    print(".2f")

    # Trade analysis
    trades = result['trades']
    if trades:
        print(f"\n📊 TRADE ANALYSIS ({len(trades)} trades)")
        print("-" * 40)

        long_trades = [t for t in trades if t['type'] == 'long']
        short_trades = [t for t in trades if t['type'] == 'short']

        print(f"Long trades: {len(long_trades)}")
        print(f"Short trades: {len(short_trades)}")

        if long_trades:
            long_win_rate = len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades)
            print(".1%")

        if short_trades:
            short_win_rate = len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades)
            print(".1%")

        # Exit reasons
        stop_losses = len([t for t in trades if t['exit_reason'] == 'stop_loss'])
        take_profits = len([t for t in trades if t['exit_reason'] == 'take_profit'])
        signals = len([t for t in trades if t['exit_reason'] == 'signal'])

        print(f"Exits by stop loss: {stop_losses}")
        print(f"Exits by take profit: {take_profits}")
        print(f"Exits by signal: {signals}")

    # Debug: Print score distribution
    if 'score_counts' in result:
        print("\n🎯 SIGNAL DISTRIBUTION")
        print("-" * 40)
        score_counts = result['score_counts']
        total_signals = sum(score_counts.values())
        print(f"Buy signals: {score_counts['buy_signals']} ({score_counts['buy_signals']/total_signals*100:.1f}%)")
        print(f"Sell signals: {score_counts['sell_signals']} ({score_counts['sell_signals']/total_signals*100:.1f}%)")
        print(f"Hold signals: {score_counts['hold_signals']} ({score_counts['hold_signals']/total_signals*100:.1f}%)")

        print("\n📊 SCORE RANGE DISTRIBUTION")
        print("-" * 40)
        score_ranges = result['score_ranges']
        for range_name, count in score_ranges.items():
            print(f"{range_name}: {count} ({count/total_signals*100:.1f}%)")

    print("\n✅ Backtest completed successfully!")
    print(f"Final capital: ¥{result['final_capital']:,.0f}")

if __name__ == "__main__":
    main()