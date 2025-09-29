#!/usr/bin/env python3
"""
Backtest runner CLI.

Executes trading strategy backtests with comprehensive metrics and reporting.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from .adapters import create_adapter, StrategyAdapter
from .metrics import MetricsCalculator, BacktestMetrics
from .report import ReportGenerator
from ..utils.observability import setup_observability, generate_correlation_id
from ..risk.position_sizing import PositionSizer, SizingMethod


class BacktestEngine:
    """Core backtest execution engine."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 0.0,
        enable_risk: bool = False,
        risk_profile: str = 'balanced',
        kill_file: str = '/tmp/ztb.stop',
        target_vol: Optional[float] = None,
        correlation_id: Optional[str] = None
    ):
        """Initialize backtest engine."""
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.enable_risk = enable_risk
        self.risk_profile = risk_profile
        self.kill_file = kill_file
        self.target_vol = target_vol
        self.correlation_id = correlation_id or generate_correlation_id()

        # Initialize position sizer
        if target_vol:
            self.position_sizer = PositionSizer(target_volatility=target_vol)
        else:
            self.position_sizer = None

        # Initialize kill switch if risk management enabled
        self.kill_switch = get_global_kill_switch() if enable_risk else None

    def load_data(self, dataset_path: str) -> pd.DataFrame:
        """Load market data from cache or file."""
        # TODO: Implement actual data loading from repository cache
        # For now, generate synthetic data
        np.random.seed(42)  # Deterministic for testing

        # Generate 1 year of daily BTC data (synthetic)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_points = len(dates)

        # Generate realistic BTC price series
        base_price = 30000
        returns = np.random.normal(0.001, 0.03, n_points)  # Mean 0.1%, vol 3%
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        high_mult = 1 + np.random.uniform(0, 0.02, n_points)
        low_mult = 1 - np.random.uniform(0, 0.02, n_points)
        volume = np.random.uniform(100, 1000, n_points)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, n_points)),
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices,
            'volume': volume
        })

        data.set_index('timestamp', inplace=True)
        return data

    def run_backtest(
        self,
        strategy: StrategyAdapter,
        data: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Run backtest simulation."""

        capital = self.initial_capital
        position = 0  # -1, 0, 1 for short, flat, long
        equity_curve = []
        orders = []

        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_data = data.iloc[:i+1]  # All data up to current point

            # Check kill switch if risk management enabled
            if self.enable_risk and self.kill_switch and self.kill_switch.is_killed():
                print(f"Kill switch activated at {timestamp}. Stopping new orders.")
                break  # Stop processing new signals

            # Generate signal
            signal = strategy.generate_signal(current_data, position)

            # Execute trade if signal
            if signal['action'] in ['buy', 'sell']:
                price = row['close'] * (1 + self.slippage_bps / 10000)  # Apply slippage

                # Calculate position size
                if self.position_sizer:
                    # Use position sizer
                    signals = {'BTC_JPY': 1.0 if signal['action'] == 'buy' else -1.0}
                    current_prices = {'BTC_JPY': price}
                    asset_vols = {'BTC_JPY': 0.5}  # Simplified volatility assumption

                    sizes = self.position_sizer.calculate_position_sizes(
                        signals, current_prices, capital, asset_vols
                    )

                    if sizes:
                        size = sizes[0]
                        shares = size.quantity
                        sizing_reason = size.sizing_reason
                    else:
                        shares = capital / price
                        sizing_reason = "Fallback: full capital"
                else:
                    # Original logic: all-in
                    shares = capital / price
                    sizing_reason = "All-in position sizing"

                if signal['action'] == 'buy' and position <= 0:
                    # Buy to long
                    order = {
                        'timestamp': timestamp,
                        'action': 'buy',
                        'price': price,
                        'shares': shares,
                        'notional': shares * price,
                        'position_before': position,
                        'position_after': 1,
                        'sizing_reason': sizing_reason,
                        'pnl': 0.0  # Will be calculated on close
                    }
                    position = 1
                    capital = 0  # All in (simplified)
                    orders.append(order)

                elif signal['action'] == 'sell' and position >= 0:
                    # Sell to short or close long
                    if position == 1:
                        # Close long position
                        entry_order = next((o for o in reversed(orders) if o['action'] == 'buy'), None)
                        if entry_order:
                            pnl = (price - entry_order['price']) * entry_order['shares']
                            capital = pnl + entry_order['notional']

                    order = {
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': price,
                        'shares': shares if 'shares' in locals() else 0,
                        'notional': abs(capital) if capital != 0 else 0,
                        'position_before': position,
                        'position_after': -1 if position == 0 else 0,
                        'sizing_reason': sizing_reason,
                        'pnl': pnl if 'pnl' in locals() else 0.0
                    }
                    position = -1 if position == 0 else 0
                    orders.append(order)

            # Record equity
            current_equity = capital if position == 0 else capital + (position * row['close'] * (shares if 'shares' in locals() else 0))
            timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
            equity_curve.append({
                'timestamp': timestamp_str,
                'equity': max(current_equity, 0)  # Prevent negative equity
            })

        # Convert to pandas objects
        equity_df = pd.DataFrame(equity_curve)
        equity_series = pd.Series(
            [p['equity'] for p in equity_curve],
            index=data.index[:len(equity_curve)]
        )

        orders_df = pd.DataFrame(orders)

        return equity_series, orders_df


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Run trading strategy backtest')
    parser.add_argument('--policy', required=True,
                       choices=['rl', 'sma_fast_slow', 'buy_hold'],
                       help='Trading strategy to test')
    parser.add_argument('--dataset', default='btc_usd_1m',
                       help='Dataset to use (default: btc_usd_1m)')
    parser.add_argument('--slippage-bps', type=float, default=5.0,
                       help='Slippage in basis points (default: 5.0)')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                       help='Initial capital (default: 10000.0)')
    parser.add_argument('--output-dir', default='results/backtest',
                       help='Output directory (default: results/backtest)')
    parser.add_argument('--enable-risk', action='store_true',
                       help='Enable risk management features (kill switches, circuit breakers)')
    parser.add_argument('--risk-profile', default='balanced',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Risk profile (default: balanced)')
    parser.add_argument('--kill-file', default='/tmp/ztb.stop',
                       help='Kill switch file path (default: /tmp/ztb.stop)')
    parser.add_argument('--target-vol', type=float,
                       help='Target volatility for position sizing (enables vol targeting)')

    args = parser.parse_args()

    # Generate correlation ID for this run
    correlation_id = generate_correlation_id()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.policy}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running backtest for {args.policy} strategy...")
    print(f"Output directory: {output_dir}")
    print(f"Correlation ID: {correlation_id}")

    # Setup observability
    obs_client = setup_observability("backtest", output_dir, correlation_id)

    try:
        # Initialize components
        engine = BacktestEngine(
            initial_capital=args.initial_capital,
            slippage_bps=args.slippage_bps,
            enable_risk=args.enable_risk,
            risk_profile=args.risk_profile,
            kill_file=args.kill_file,
            target_vol=args.target_vol,
            correlation_id=correlation_id
        )

        strategy = create_adapter(args.policy)
        data = engine.load_data(args.dataset)

        # Run backtest
        equity_curve, orders = engine.run_backtest(strategy, data)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            equity_curve=equity_curve,
            orders=orders,
            initial_capital=args.initial_capital,
            slippage_bps=args.slippage_bps
        )

        # Create run metadata
        import platform
        import hashlib
        run_metadata = {
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": f"backtest_{args.policy}_{timestamp}",
            "type": "backtest",
            "config": {
                "policy": args.policy,
                "dataset": args.dataset,
                "slippage_bps": args.slippage_bps,
                "initial_capital": args.initial_capital,
                "enable_risk": args.enable_risk,
                "risk_profile": args.risk_profile,
                "target_vol": args.target_vol
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": platform.os.cpu_count()
            },
            "seeds": {
                "numpy": 42,  # From load_data
                "random": None
            },
            "package_hashes": {}  # TODO: Add package hashes
        }

        # Save run metadata
        obs_client.export_artifact("run_metadata", run_metadata)

        # Generate reports
        metadata = {
            'strategy': args.policy,
            'dataset': args.dataset,
            'slippage_bps': args.slippage_bps,
            'initial_capital': args.initial_capital
        }

        equity_list = [{'timestamp': ts, 'equity': eq}
                      for ts, eq in equity_curve.items()]
        orders_list = orders.to_dict('records') if not orders.empty else []

        # Generate outputs
        ReportGenerator.generate_json_report(
            metrics, equity_list, orders_list, metadata,
            output_dir / 'metrics.json'
        )

        ReportGenerator.generate_markdown_report(
            metrics, metadata,
            output_dir / 'report.md'
        )

        ReportGenerator.generate_equity_csv(
            equity_list,
            output_dir / 'equity_curve.csv'
        )

        ReportGenerator.generate_orders_csv(
            orders_list,
            output_dir / 'orders.csv'
        )

        print("Backtest completed successfully!")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Win Rate: {metrics.win_rate:.1%}")
        print(f"Total Trades: {metrics.total_trades}")

    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()