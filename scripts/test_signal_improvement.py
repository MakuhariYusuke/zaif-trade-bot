#!/usr/bin/env python3
"""
Signal Guidance Improvement Backtest Validation Script

このスクリプトは、改善された信号ガイダンスシステムのバックテスト検証を実行します。
目標: 信号頻度を2.9 signals/dayから20-50 signals/dayに改善

検証内容:
- 信号頻度の測定
- 信号品質の評価
- バックテスト結果の比較
- パフォーマンス指標の計算
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem
from ztb.utils.constants import DEFAULT_SEED
from ztb.utils.logging_utils import get_logger, setup_logging

# Enable debug logging for all loggers
setup_logging()

logger = get_logger(__name__)


class SignalGuidanceBacktestValidator:
    """
    Backtest validator for signal guidance improvements

    Validates frequency improvements and signal quality enhancements
    """

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "configs/backtest_config.json"
        self.improved_system = SignalGuidanceSystem()
        self.results_dir = Path("results/signal_guidance_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_market_data(self, symbol: str = "BTC/JPY", days: int = 30) -> pd.DataFrame:
        """
        Load market data for backtesting

        Args:
            symbol: Trading symbol
            days: Number of days of data to load

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading {days} days of {symbol} data")

        # For this validation, we'll generate synthetic but realistic data
        # In production, this would load from actual market data sources
        np.random.seed(DEFAULT_SEED)

        # Generate data for the specified number of days (5-minute intervals)
        intervals_per_day = 288  # 24 hours * 12 (5-min intervals)
        total_points = days * intervals_per_day

        # Start from a recent date
        start_time = datetime.now() - timedelta(days=days)

        # Generate realistic price series
        base_price = 50000
        prices = [base_price]
        timestamps = [start_time]

        for i in range(total_points - 1):
            # Realistic price movement model
            trend = 0.00005 * np.sin(i / 1000)  # Long-term trend
            mean_reversion = (base_price - prices[-1]) * 0.0001  # Mean reversion
            noise = np.random.normal(0, 0.002)  # Random noise
            volatility = np.random.choice([0.001, 0.004], p=[0.8, 0.2])  # Variable volatility

            change = trend + mean_reversion + noise * volatility
            new_price = prices[-1] * (1 + change)
            new_price = max(new_price, 10000)  # Floor price

            prices.append(new_price)
            timestamps.append(start_time + timedelta(minutes=5*(i+1)))

        # Create OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate OHLC from close price
            volatility_factor = np.random.uniform(0.001, 0.008)
            high = close * (1 + abs(np.random.normal(0, volatility_factor)))
            low = close * (1 - abs(np.random.normal(0, volatility_factor)))
            open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.0005))
            volume = np.random.lognormal(12, 0.8)  # Realistic volume

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        logger.info(f"Loaded {len(df)} data points")
        return df

    def simulate_model_actions(self, market_data: pd.DataFrame) -> List[float]:
        """
        Simulate continuous actions from a trading model

        Args:
            market_data: Market data DataFrame

        Returns:
            List of continuous actions (-1 to 1)
        """
        logger.info("Simulating model continuous actions")

        actions = []
        np.random.seed(123)  # Different seed for variety

        for _, row in market_data.iterrows():
            # Simulate realistic model output distribution with balanced buy/sell signals
            # Use a distribution that can generate both positive and negative signals
            base_action = np.random.normal(0, 0.5)  # Increased std dev for more variation

            # Add some market-responsive behavior
            price_change = (row['close'] - row['open']) / row['open']
            market_influence = price_change * 3  # Amplify market movement effect

            # Add occasional strong signals in both directions
            if np.random.random() < 0.1:  # 10% chance of strong signal
                if np.random.random() < 0.5:
                    base_action = np.random.uniform(0.5, 1.0)  # Strong BUY
                else:
                    base_action = np.random.uniform(-1.0, -0.5)  # Strong SELL

            action = base_action + market_influence
            action = np.clip(action, -1, 1)  # Clip to valid range

            actions.append(action)

        logger.info(f"Generated {len(actions)} actions. "
                   f"Buy signals: {sum(1 for a in actions if a > 0.2)}, "
                   f"Sell signals: {sum(1 for a in actions if a < -0.2)}, "
                   f"Neutral: {sum(1 for a in actions if -0.2 <= a <= 0.2)}")

        return actions

    def run_signal_guidance_backtest(self, market_data: pd.DataFrame,
                                   continuous_actions: List[float],
                                   initial_portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest with improved signal guidance

        Args:
            market_data: Market data for backtesting
            continuous_actions: Model continuous actions
            initial_portfolio: Initial portfolio state

        Returns:
            Backtest results dictionary
        """
        logger.info("Running signal guidance backtest")

        signals = []
        portfolio_history = [initial_portfolio.copy()]
        quality_scores = []

        current_portfolio = initial_portfolio.copy()

        for i, (_, row) in enumerate(market_data.iterrows()):
            continuous_action = continuous_actions[i] if i < len(continuous_actions) else 0.0

            # Apply signal guidance using the full system
            guided_action = self.improved_system.apply_guidance(
                continuous_action, row, current_portfolio
            )

            # Debug: Print action details
            print(f"Row {i}: continuous_action={continuous_action:.3f}, guided_action={guided_action}")

            # Get quality score from the scorer (for metrics)
            market_df = self.improved_system._create_market_dataframe(row, current_portfolio)
            _, quality_score = self.improved_system.quality_scorer.calculate_signal_quality(
                market_df, continuous_action, current_portfolio
            )

            signals.append(guided_action)
            quality_scores.append(quality_score)

            # Update portfolio (simplified - just track balance)
            current_price = row['close']
            current_portfolio['current_price'] = current_price
            current_portfolio['portfolio_value'] = (
                current_portfolio['btc_balance'] * current_price +
                current_portfolio['jpy_balance']
            )

            portfolio_history.append(current_portfolio.copy())

        # Calculate metrics
        results = self._calculate_backtest_metrics(signals, quality_scores, market_data, portfolio_history)

        return results

    def _calculate_backtest_metrics(self, signals: List[int], quality_scores: List[float],
                                  market_data: pd.DataFrame, portfolio_history: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics"""

        # Basic signal statistics
        buy_signals = sum(1 for s in signals if s == 1)
        sell_signals = sum(1 for s in signals if s == -1)
        hold_signals = sum(1 for s in signals if s == 0)
        total_signals = len(signals)

        # Signal frequency (signals per day)
        days = len(market_data) / 288  # Assuming 288 5-min bars per day
        signals_per_day = (buy_signals + sell_signals) / days if days > 0 else 0

        # Signal quality metrics
        avg_quality_score = np.mean(quality_scores)
        quality_score_std = np.std(quality_scores)

        # Signal distribution analysis
        signal_changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
        signal_persistence = 1 - (signal_changes / max(1, len(signals) - 1))

        # Portfolio performance (simplified)
        initial_value = portfolio_history[0]['portfolio_value']
        final_value = portfolio_history[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100

        # Market return for comparison
        market_initial = market_data.iloc[0]['close']
        market_final = market_data.iloc[-1]['close']
        market_return = (market_final - market_initial) / market_initial * 100

        # Sharpe-like ratio (simplified)
        returns = [h['portfolio_value'] for h in portfolio_history]
        if len(returns) > 1:
            portfolio_returns = np.diff(returns) / returns[:-1]
            sharpe_ratio = np.mean(portfolio_returns) / max(0.0001, np.std(portfolio_returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        metrics = {
            'signal_frequency': {
                'signals_per_day': signals_per_day,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'total_signals': total_signals,
                'signal_ratio': (buy_signals + sell_signals) / total_signals if total_signals > 0 else 0
            },
            'signal_quality': {
                'avg_quality_score': avg_quality_score,
                'quality_score_std': quality_score_std,
                'min_quality_score': min(quality_scores) if quality_scores else 0,
                'max_quality_score': max(quality_scores) if quality_scores else 0,
                'win_rate': 0.0,  # Placeholder - would need actual trade results
                'profit_factor': 1.0,  # Placeholder - would need actual P&L data
                'max_drawdown': 0.0  # Placeholder - would need drawdown calculation
            },
            'signal_behavior': {
                'signal_changes': signal_changes,
                'signal_persistence': signal_persistence,
                'avg_signal_streak': self._calculate_avg_streak(signals)
            },
            'performance': {
                'total_return_pct': total_return,
                'market_return_pct': market_return,
                'excess_return': total_return - market_return,
                'sharpe_ratio': sharpe_ratio
            },
            'metadata': {
                'backtest_days': days,
                'data_points': len(market_data),
                'timestamp': datetime.now().isoformat()
            }
        }

        return metrics

    def _calculate_avg_streak(self, signals: List[int]) -> float:
        """Calculate average signal streak length"""
        if not signals:
            return 0

        streaks = []
        current_streak = 1

        for i in range(1, len(signals)):
            if signals[i] == signals[i-1] and signals[i] != 0:
                current_streak += 1
            else:
                if current_streak > 1:
                    streaks.append(current_streak)
                current_streak = 1

        if current_streak > 1:
            streaks.append(current_streak)

        return np.mean(streaks) if streaks else 1.0

    def compare_with_baseline(self, improved_results: Dict, baseline_frequency: float = 2.9) -> Dict:
        """Compare improved results with baseline performance"""

        improved_freq = improved_results['signal_frequency']['signals_per_day']

        comparison = {
            'frequency_improvement': {
                'baseline_signals_per_day': baseline_frequency,
                'improved_signals_per_day': improved_freq,
                'improvement_factor': improved_freq / baseline_frequency if baseline_frequency > 0 else 0,
                'improvement_pct': ((improved_freq - baseline_frequency) / baseline_frequency * 100) if baseline_frequency > 0 else 0
            },
            'target_achievement': {
                'target_min': 20,
                'target_max': 50,
                'achieved': improved_freq,
                'target_met': 20 <= improved_freq <= 50,
                'within_range': improved_freq >= 20
            }
        }

        return comparison

    def save_results(self, results: Dict, comparison: Dict, filename: str = None) -> str:
        """Save backtest results to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signal_guidance_backtest_{timestamp}.json"

        filepath = self.results_dir / filename

        output = {
            'backtest_results': results,
            'baseline_comparison': comparison,
            'summary': {
                'signals_per_day': results['signal_frequency']['signals_per_day'],
                'avg_quality_score': results['signal_quality']['avg_quality_score'],
                'target_achieved': comparison['target_achievement']['target_met']
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {filepath}")
        return str(filepath)

    def print_summary(self, results: Dict, comparison: Dict):
        """Print human-readable summary of results"""

        print("\n" + "="*60)
        print("SIGNAL GUIDANCE IMPROVEMENT BACKTEST RESULTS")
        print("="*60)

        # Signal Frequency
        freq = results['signal_frequency']
        print("\n📊 SIGNAL FREQUENCY:")
        print(f"   Total Signals/Day: {freq['signals_per_day']:.1f}")
        print(f"   Buy Signals: {freq['buy_signals']}")
        print(f"   Sell Signals: {freq['sell_signals']}")
        print(f"   Hold Signals: {freq['hold_signals']}")
        print(f"   Signal Ratio: {freq['signal_ratio']:.1f}")

        # Signal Quality
        quality = results['signal_quality']
        print("\n🎯 SIGNAL QUALITY:")
        print(f"   Win Rate: {quality['win_rate']:.1f}%")
        print(f"   Profit Factor: {quality['profit_factor']:.1f}")
        print(f"   Max Drawdown: {quality['max_drawdown']:.1f}%")
        print(f"   Avg Quality Score: {quality['avg_quality_score']:.1f}")

        # Performance
        perf = results['performance']
        print("\n💰 PERFORMANCE:")
        print(f"   Total Return: {perf['total_return_pct']:.2f}%")
        print(f"   Market Return: {perf['market_return_pct']:.2f}%")
        print(f"   Excess Return: {perf['excess_return']:.2f}%")
        print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        # Comparison
        comp = comparison['frequency_improvement']
        target = comparison['target_achievement']
        print("\n🏆 COMPARISON WITH BASELINE:")
        print(f"   Frequency Improvement: {comp['improvement_factor']:.1f}x")
        print(f"   Baseline Signals/Day: {comp['baseline_signals_per_day']:.1f}")
        print(f"   Improved Signals/Day: {comp['improved_signals_per_day']:.1f}")
        print(f"   Target Range: {target['target_min']}-{target['target_max']} signals/day")
        print(f"   Target Achieved: {'✅ YES' if target['target_met'] else '❌ NO'}")

        print("\n" + "="*60)

    def run_validation(self, days: int = 30) -> Tuple[Dict, Dict]:
        """
        Run complete validation suite

        Args:
            days: Number of days to backtest

        Returns:
            Tuple of (results, comparison)
        """
        logger.info(f"Starting signal guidance validation for {days} days")

        # Load market data
        market_data = self.load_market_data(days=days)

        # Simulate model actions
        continuous_actions = self.simulate_model_actions(market_data)

        # Initial portfolio
        initial_portfolio = {
            'btc_balance': 0.5,
            'jpy_balance': 100000,
            'current_price': market_data.iloc[0]['close'],
            'portfolio_value': 0.5 * market_data.iloc[0]['close'] + 100000
        }

        # Run backtest
        results = self.run_signal_guidance_backtest(
            market_data, continuous_actions, initial_portfolio
        )

        # Compare with baseline
        comparison = self.compare_with_baseline(results)

        # Print summary
        self.print_summary(results, comparison)

        # Save results
        self.save_results(results, comparison)

        return results, comparison


def main():
    """Main validation function"""
    validator = SignalGuidanceBacktestValidator()

    # Run validation
    results, comparison = validator.run_validation(days=30)

    # Check if targets met
    signals_per_day = results['signal_frequency']['signals_per_day']
    target_met = 20 <= signals_per_day <= 50

    if target_met:
        logger.info("🎉 TARGET ACHIEVED: Signal frequency improved to target range!")
        return 0
    else:
        logger.warning(f"⚠️  TARGET NOT MET: {signals_per_day:.1f} signals/day (target: 20-50)")
        return 1


if __name__ == "__main__":
    exit(main())
