#!/usr/bin/env python3
"""
SAC v445.3 Paper Trading Experiment
Phase 3-3 Live Trading Integration Test
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.live_trading.trading_api import TradingAPI
from ztb.live_trading.live_trader import LiveTrader
from ztb.training.scripts.paper_trade import detect_algorithm
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.env_config import get_trading_env_config
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem, GuidanceConfig


class PaperTradingExperiment:

    def __init__(self, model_path: str = "models/sac_v444_2_final_model.zip"):
        self.model_path = Path(model_path)
        self.results_dir = Path("results/paper_trading")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize signal guidance system
        self.signal_guidance = SignalGuidanceSystem(guidance_level="adaptive")

        # Detect algorithm from model path
        self.algorithm = detect_algorithm(self.model_path)
        print(f"Detected algorithm: {self.algorithm}")

        # Load model based on algorithm
        print(f"Loading {self.algorithm.upper()} model from {self.model_path}")
        try:
            if self.algorithm == "sac":
                self.model = SAC.load(str(self.model_path))  # type: ignore
            else:
                # For PPO or other algorithms, we'd need to import and use appropriate class
                raise ValueError(f"Algorithm {self.algorithm} not supported for paper trading experiment")
            print(f"{self.algorithm.upper()} model loaded successfully")
        except Exception as e:
            print(f"Failed to load {self.algorithm.upper()} model: {e}")
            raise

        # Initialize trading environment for observation generation
        self.env = None
        self._setup_environment()

        # Initialize components with mock credentials for paper trading
        self.trading_api = TradingAPI(
            api_key="mock_api_key",
            api_secret="mock_api_secret",
            test_mode=True
        )

        # Simple signal generator for paper trading (placeholder)
        def simple_signal_generator():
            return []

        self.live_trader = LiveTrader(
            trading_api=self.trading_api,
            signal_generator=simple_signal_generator,
            max_position_size=1.0
        )

        self.experiment_data = []

    def _setup_environment(self) -> None:
        """Setup trading environment for observation generation"""
        try:
            # Create a minimal config for observation generation
            config = {
                "reward_scaling": 1.0,
                "transaction_cost": 0.001,
                "max_position_size": 1.0,
                "risk_free_rate": 0.0,
                "initial_portfolio_value": 10000.0,
                "verbose": 0,
                "correlation_reduction": True,
                "enable_correlation_reduction": True,
            }

            # Create environment with dummy data for observation space
            dummy_df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
                'open': [50000.0] * 100,
                'high': [51000.0] * 100,
                'low': [49000.0] * 100,
                'close': [50000.0] * 100,
                'volume': [1000.0] * 100
            })

            self.env = DummyVecEnv([lambda: HeavyTradingEnv(df=dummy_df, config=config)])
            print("Trading environment setup completed")
        except Exception as e:
            print(f"Failed to setup environment: {e}")
            raise

    def load_market_data(self, days: int = 30) -> pd.DataFrame:
        """Load recent market data for paper trading"""
        try:
            # Load BTC/JPY data
            data_path = Path("data/btc_jpy_real_dataset.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Market data not found: {data_path}")

            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Get recent data
            end_date = df['timestamp'].max()
            start_date = end_date - timedelta(days=days)

            recent_df = df[df['timestamp'] >= start_date].copy()
            recent_df = recent_df.sort_values('timestamp').reset_index(drop=True)

            print(f"Loaded {len(recent_df)} data points from {start_date.date()} to {end_date.date()}")
            return recent_df

        except Exception as e:
            print(f"Error loading market data: {e}")
            raise

    def simulate_paper_trading(self, df: pd.DataFrame, initial_balance: float = 10000.0) -> Dict[str, Any]:
        """Simulate paper trading with real market data"""

        print("Starting paper trading simulation...")

        # Initialize portfolio
        portfolio = {
            'jpy_balance': initial_balance,
            'btc_balance': 0.0,
            'portfolio_value': initial_balance,
            'trades': [],
            'performance_history': []
        }

        # Trading parameters
        position_size = 0.1  # 10% of portfolio per trade
        min_trade_amount = 1000  # Minimum JPY per trade

        # Simulate trading
        for i, row in df.iterrows():
            current_price = row['close']
            timestamp = row['timestamp']

            # Update portfolio value
            btc_value = portfolio['btc_balance'] * current_price
            portfolio['portfolio_value'] = portfolio['jpy_balance'] + btc_value

            # Record performance
            portfolio['performance_history'].append({
                'timestamp': timestamp.isoformat(),
                'portfolio_value': portfolio['portfolio_value'],
                'btc_price': current_price,
                'btc_balance': portfolio['btc_balance'],
                'jpy_balance': portfolio['jpy_balance']
            })

            # Generate trading signal (simplified logic)
            # In real implementation, this would use the SAC model
            signal = self._generate_signal(row, portfolio)

            if signal == 'BUY' and portfolio['jpy_balance'] >= min_trade_amount:
                # Calculate position size
                trade_amount = min(
                    portfolio['jpy_balance'] * position_size,
                    portfolio['jpy_balance']
                )

                if trade_amount >= min_trade_amount:
                    btc_amount = trade_amount / current_price * (1 - 0.001)  # Subtract fee

                    portfolio['jpy_balance'] -= trade_amount
                    portfolio['btc_balance'] += btc_amount

                    portfolio['trades'].append({
                        'timestamp': timestamp.isoformat(),
                        'type': 'BUY',
                        'price': current_price,
                        'btc_amount': btc_amount,
                        'jpy_amount': trade_amount,
                        'portfolio_value': portfolio['portfolio_value']
                    })

                    print(f"BUY: {btc_amount:.6f} BTC @ {current_price:,.0f} JPY (Portfolio: {portfolio['portfolio_value']:,.0f} JPY)")

            elif signal == 'SELL' and portfolio['btc_balance'] > 0.0001:
                # Sell all BTC
                btc_amount = portfolio['btc_balance']
                jpy_amount = btc_amount * current_price * (1 - 0.001)  # Subtract fee

                portfolio['jpy_balance'] += jpy_amount
                portfolio['btc_balance'] = 0.0

                portfolio['trades'].append({
                    'timestamp': timestamp.isoformat(),
                    'type': 'SELL',
                    'price': current_price,
                    'btc_amount': btc_amount,
                    'jpy_amount': jpy_amount,
                    'portfolio_value': portfolio['portfolio_value']
                })

                print(f"SELL: {btc_amount:.6f} BTC @ {current_price:,.0f} JPY (Portfolio: {portfolio['portfolio_value']:,.0f} JPY)")

        # Final portfolio update
        final_btc_value = portfolio['btc_balance'] * df.iloc[-1]['close']
        portfolio['portfolio_value'] = portfolio['jpy_balance'] + final_btc_value

        return portfolio

    def _generate_signal(self, row: pd.Series, portfolio: Dict[str, Any]) -> str:
        """Generate trading signal using SAC v445.3 model"""
        try:
            # Create observation from current market data and portfolio state
            # This is a simplified observation - in practice, you'd use the full feature set
            current_price = float(row['close'])  # type: ignore

            # Create a basic observation vector (simplified for paper trading)
            # In real implementation, this should match the model's expected observation space
            obs = np.array([
                current_price / 100000.0,  # Normalized price
                portfolio['btc_balance'],    # Position size
                portfolio['jpy_balance'] / 10000.0,  # Normalized cash
                0.0,  # Placeholder for other features
                0.0,  # Placeholder for other features
            ], dtype=np.float32)

            # Ensure observation matches expected shape
            expected_shape = self.model.observation_space.shape[0]
            if len(obs) < expected_shape:
                # Pad with zeros if needed
                obs = np.pad(obs, (0, expected_shape - len(obs)), 'constant')
            elif len(obs) > expected_shape:
                # Truncate if needed
                obs = obs[:expected_shape]

            # Get action from SAC model
            action, _ = self.model.predict(obs, deterministic=True)

            # Apply signal guidance to convert continuous action to discrete signal
            discrete_action = self.signal_guidance.apply_guidance(action, row, portfolio)

            if discrete_action == 1:  # BUY
                return 'BUY'
            elif discrete_action == -1:  # SELL
                return 'SELL'
            else:  # HOLD (0)
                return 'HOLD'

        except Exception as e:
            print(f"Error generating signal with SAC model: {e}")
            return 'HOLD'

    def calculate_performance_metrics(self, portfolio: Dict[str, Any], initial_balance: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""

        trades = portfolio['trades']
        performance_history = portfolio['performance_history']

        # Basic metrics
        final_value = portfolio['portfolio_value']
        total_return = (final_value - initial_balance) / initial_balance * 100
        total_trades = len(trades)

        # BTC-related metrics
        final_btc_balance = portfolio['btc_balance']
        initial_btc_balance = 0.0  # Always start with 0 BTC
        btc_change = final_btc_balance - initial_btc_balance

        # Get final BTC price for valuation
        if performance_history:
            final_btc_price = performance_history[-1]['btc_price']
            final_btc_value = final_btc_balance * final_btc_price
            final_jpy_balance = portfolio['jpy_balance']
        else:
            final_btc_price = 0.0
            final_btc_value = 0.0
            final_jpy_balance = initial_balance

        # Trading metrics
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']

        win_trades = 0
        total_pnl = 0

        # Calculate P&L for completed trade pairs
        for sell_trade in sell_trades:
            # Find corresponding buy trade (simplified - assumes FIFO)
            buy_trade = None
            for bt in buy_trades:
                if bt['timestamp'] < sell_trade['timestamp']:
                    buy_trade = bt
                    break

            if buy_trade:
                pnl = sell_trade['jpy_amount'] - buy_trade['jpy_amount']
                total_pnl += pnl
                if pnl > 0:
                    win_trades += 1

        win_rate = win_trades / max(1, len(sell_trades)) * 100

        # Risk metrics
        portfolio_values = [p['portfolio_value'] for p in performance_history]
        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        # Signal guidance statistics
        signal_stats = self.signal_guidance._calculate_signal_statistics(trades)

        return {
            'initial_balance': initial_balance,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate_pct': win_rate,
            'total_pnl': total_pnl,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(performance_history),
            # BTC-specific metrics
            'initial_btc_balance': initial_btc_balance,
            'final_btc_balance': final_btc_balance,
            'btc_change': btc_change,
            'final_btc_price': final_btc_price,
            'final_btc_value': final_btc_value,
            'final_jpy_balance': final_jpy_balance,
            'btc_value_change_pct': ((final_btc_value - (initial_btc_balance * final_btc_price)) / max(initial_btc_balance * final_btc_price, 0.01)) * 100 if initial_btc_balance > 0 else 0.0,
            # Signal guidance metrics
            'signal_stats': signal_stats
        }

    def _calculate_max_drawdown(self, portfolio_values: list) -> float:
        """Calculate maximum drawdown percentage"""
        if not portfolio_values:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_sharpe_ratio(self, performance_history: list) -> float:
        """Calculate Sharpe ratio"""
        if len(performance_history) < 2:
            return 0.0

        # Calculate daily returns
        values = [p['portfolio_value'] for p in performance_history]
        returns = []

        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            returns.append(daily_return)

        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5

        # Assume risk-free rate of 0 for simplicity
        return avg_return / max(std_return, 0.0001) * (252 ** 0.5)  # Annualized

    def save_results(self, portfolio: Dict[str, Any], metrics: Dict[str, Any], experiment_name: str):
        """Save experiment results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"paper_trading_{experiment_name}_{timestamp}.json"

        results = {
            'experiment_info': {
                'name': experiment_name,
                'timestamp': timestamp,
                'model_path': str(self.model_path),
                'phase': '3-3 Live Trading Integration'
            },
            'portfolio': portfolio,
            'metrics': metrics
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {result_file}")
        return result_file

    def run_experiment(self, days: int = 30, initial_balance: float = 10000.0):
        """Run the complete paper trading experiment"""

        print("=== SAC v445.3 Paper Trading Experiment ===")
        print("Phase 3-3 Live Trading Integration Test")
        print(f"Model: {self.model_path}")
        print(f"Initial Balance: {initial_balance:,.0f} JPY")
        print(f"Test Period: {days} days")
        print()

        try:
            # Load market data
            df = self.load_market_data(days)

            # Run paper trading simulation
            portfolio = self.simulate_paper_trading(df, initial_balance)

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(portfolio, initial_balance)

            # Display results
            print("\n" + "="*60)
            print("PAPER TRADING EXPERIMENT RESULTS")
            print("="*60)

            print("\n💰 PORTFOLIO SUMMARY:")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")

            print("\n📊 TRADING PERFORMANCE:")
            print(".2f")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Buy Trades: {metrics['buy_trades']}")
            print(f"Sell Trades: {metrics['sell_trades']}")
            print(".2f")
            print(".2f")

            print("\n📈 RISK METRICS:")
            print(".2f")
            print(".2f")

            print("\n₿ BTC POSITION ANALYSIS:")
            print(".8f")
            print(".8f")
            print(".8f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")

            print("\n🎯 SIGNAL GUIDANCE ANALYSIS:")
            signal_stats = metrics.get('signal_stats', {})
            print(f"Total Signals: {signal_stats.get('total_signals', 0)}")
            print(f"Buy Signals: {signal_stats.get('buy_signals', 0)}")
            print(f"Sell Signals: {signal_stats.get('sell_signals', 0)}")
            print(f"Hold Signals: {signal_stats.get('hold_signals', 0)}")
            print(".2f")
            print(".1f")

            print("\n" + "="*60)

            # Save results
            experiment_name = f"sac_v445_3_paper_trading_{days}days"
            result_file = self.save_results(portfolio, metrics, experiment_name)

            print(f"\nExperiment completed successfully!")
            print(f"Results saved to: {result_file}")

            return metrics

        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SAC v445.3 Paper Trading Experiment')
    parser.add_argument('--model-path', type=str, default='models/sac_v445_3_final.zip',
                       help='Path to the SAC model file')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to test')
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                       help='Initial portfolio balance in JPY')

    args = parser.parse_args()

    experiment = PaperTradingExperiment(model_path=args.model_path)
    results = experiment.run_experiment(days=args.days, initial_balance=args.initial_balance)

    if results:
        print("\n🎯 Experiment Summary:")
        print(f"• Return: {results['total_return_pct']:.2f}%")
        print(f"• Trades: {results['total_trades']}")
        print(f"• Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"• Max DD: {results['max_drawdown_pct']:.2f}%")
    else:
        print("❌ Experiment failed")


if __name__ == "__main__":
    main()