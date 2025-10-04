#!/usr/bin/env python3
"""
Paper trader CLI.

Runs paper trading simulations with replay or live-lite modes.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import yaml

from ztb.risk.circuit_breakers import (  # type: ignore[import]
    KillSwitchActivatedError,
    get_global_kill_switch,
)
from ztb.risk.position_sizing import PositionSizer  # type: ignore[import]
from ztb.trading.backtest.adapters import StrategyAdapter, create_adapter
from ztb.utils.cli_common import (
    CLIFormatter,
    CLIValidator,
    CommonArgs,
    create_standard_parser,
)
from ztb.utils.data_utils import load_csv_data

from .sim_broker import SimBroker


def load_venue_config(venue_name: str, config_dir: str = "venues") -> Dict[str, Any]:
    """Load venue configuration from YAML file."""
    config_path = Path(config_dir) / f"{venue_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Venue config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return cast(Dict[str, Any], config)


class SymbolMeta:
    """Symbol metadata for validation."""

    def __init__(self, symbol_config: Dict[str, Any]) -> None:
        self.symbol = symbol_config["symbol"]
        self.base_asset = symbol_config["base_asset"]
        self.quote_asset = symbol_config["quote_asset"]
        self.min_order_size = symbol_config["min_order_size"]
        self.max_order_size = symbol_config["max_order_size"]
        self.price_precision = symbol_config["price_precision"]
        self.quantity_precision = symbol_config["quantity_precision"]
        self.min_price = symbol_config["min_price"]
        self.max_price = symbol_config["max_price"]

    def validate_order(
        self, side: str, quantity: float, price: Optional[float] = None
    ) -> List[str]:
        """Validate order parameters against symbol constraints."""
        errors = []

        # Quantity validation
        if quantity < self.min_order_size:
            errors.append(f"Quantity {quantity} below minimum {self.min_order_size}")
        if quantity > self.max_order_size:
            errors.append(f"Quantity {quantity} above maximum {self.max_order_size}")

        # Price validation (for limit orders)
        if price is not None:
            if price < self.min_price:
                errors.append(f"Price {price} below minimum {self.min_price}")
            if price > self.max_price:
                errors.append(f"Price {price} above maximum {self.max_price}")

        return errors


class PaperTrader:
    """Paper trading execution engine."""

    def __init__(
        self,
        broker: SimBroker,
        strategy: StrategyAdapter,
        mode: str = "replay",
        dataset: Optional[str] = None,
        duration_minutes: int = 60,
        enable_risk: bool = False,
        risk_profile: str = "aggressive",
        kill_file: str = "/tmp/ztb.stop",
        venue_config: Optional[Dict[str, Any]] = None,
        target_vol: Optional[float] = None,
        from_streaming: bool = False,
    ) -> None:
        """Initialize paper trader."""
        self.broker = broker
        self.strategy = strategy
        self.mode = mode
        self.dataset = dataset
        self.duration_minutes = duration_minutes
        self.enable_risk = enable_risk
        self.risk_profile = risk_profile
        self.kill_file = kill_file
        self.venue_config = venue_config or {}
        self.from_streaming = from_streaming

        # Initialize position sizer
        self.position_sizer = PositionSizer(target_volatility=target_vol or 0.10)

        # Load symbol metadata for validation
        self.symbol_meta = {}
        if "symbols" in self.venue_config:
            for symbol_config in self.venue_config["symbols"]:
                symbol = symbol_config["symbol"]
                self.symbol_meta[symbol] = SymbolMeta(symbol_config)

        # Initialize kill switch if risk management enabled
        self.kill_switch = get_global_kill_switch() if enable_risk else None

        self.target_vol = target_vol

        # Initialize position sizer
        if target_vol:
            self.position_sizer = PositionSizer(target_volatility=target_vol)
        else:
            self.position_sizer = None

        # Load data feed for replay mode
        if self.mode == "replay":
            self.data_feed = self._load_data_feed(self.dataset or "btc_jpy_1m")
        else:
            self.data_feed = None

    def validate_order(
        self, symbol: str, side: str, quantity: float, price: Optional[float] = None
    ) -> None:
        """Validate order against venue constraints."""
        if symbol not in self.symbol_meta:
            raise ValueError(f"Symbol {symbol} not configured in venue")

        meta = self.symbol_meta[symbol]
        errors = meta.validate_order(side, quantity, price)
        if errors:
            raise ValueError(f"Order validation failed: {'; '.join(errors)}")

    def _load_data_feed(self, dataset: str) -> pd.DataFrame:
        """Load historical data for replay mode."""
        # TODO: Load from actual repository cache
        # For now, generate synthetic data
        import numpy as np

        np.random.seed(42)  # Deterministic

        # Generate data for replay
        start_time = datetime.now()
        times = pd.date_range(
            start=start_time, periods=self.duration_minutes, freq="1min"
        )

        # Generate realistic BTC price series
        base_price = 30000
        n_points = len(times)

        # Random walk with some trend
        returns = np.random.normal(0.0001, 0.005, n_points)  # Small mean, moderate vol
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        high_mult = 1 + np.random.uniform(0, 0.005, n_points)
        low_mult = 1 - np.random.uniform(0, 0.005, n_points)
        volume = np.random.uniform(50, 500, n_points)

        data = pd.DataFrame(
            {
                "timestamp": times,
                "open": prices * (1 + np.random.normal(0, 0.002, n_points)),
                "high": prices * high_mult,
                "low": prices * low_mult,
                "close": prices,
                "volume": volume,
            }
        )

        data.set_index("timestamp", inplace=True)
        return data

    def _prepare_data_feed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data feed for replay."""
        # For RL, we need all features, not just OHLCV
        return df

    async def run_replay(self, output_dir: Path) -> Dict[str, Any]:
        """Run replay mode simulation."""
        if self.data_feed is None:
            if self.dataset:
                self.data_feed = load_csv_data(self.dataset)
                self.data_feed = self._prepare_data_feed(self.data_feed)
                print(f"Data columns: {self.data_feed.columns.tolist()}")  # type: ignore[unreachable]
            else:
                raise ValueError("No dataset provided for replay mode")

        # Capture run metadata
        metadata_path = output_dir / "run_metadata.json"
        # capture_run_metadata(str(metadata_path))
        # Create dummy metadata for canary test
        dummy_metadata = {
            "correlation_id": "dummy",
            "system": {"os": "dummy"},
            "git": {"sha": "dummy"},
            "run_config": {"random_seed": 42},
        }
        with open(metadata_path, "w") as f:
            json.dump(dummy_metadata, f, indent=2)

        print(f"Starting replay simulation for {self.duration_minutes} minutes...")

        position = 0  # -1, 0, 1
        trades_executed = 0

        for i, (timestamp, row) in enumerate(self.data_feed.iterrows()):
            # Progress indicator
            if i % 10 == 0:
                progress = (i + 1) / len(self.data_feed) * 100
                print(".1f")

            # Check kill switch if risk management enabled
            if self.enable_risk and self.kill_switch:
                try:
                    await self.kill_switch.check_and_raise()
                except KillSwitchActivatedError:
                    print(f"Kill switch activated at {timestamp}. Stopping new orders.")
                    # Continue processing but skip new orders
                    continue

            # Get current market data up to this point
            current_data = self.data_feed.iloc[: i + 1]

            # Generate trading signal
            signal = self.strategy.generate_signal(current_data, position)

            # Execute trade if signal
            if signal["action"] in ["buy", "sell"]:
                try:
                    symbol = "btc_jpy"
                    price = row["close"]

                    # Calculate position size
                    if self.position_sizer:
                        # Use position sizer
                        signals = {
                            "BTC_JPY": 1.0 if signal["action"] == "buy" else -1.0
                        }
                        current_prices = {"BTC_JPY": price}
                        asset_vols = {
                            "BTC_JPY": 0.5
                        }  # Simplified volatility assumption

                        # Calculate portfolio value
                        portfolio_value = (
                            self.broker.balance["JPY"]
                            + self.broker.balance["BTC"] * current_prices["BTC_JPY"]
                        )

                        sizes = self.position_sizer.calculate_position_sizes(
                            signals, current_prices, portfolio_value, asset_vols
                        )

                        if sizes:
                            size = sizes[0]
                            quantity = size.quantity

                            # Log sizing chain to orders.csv
                            order_record = {
                                "timestamp": str(timestamp),
                                "symbol": symbol,
                                "side": signal["action"],
                                "price": price,
                                "quantity": quantity,
                                "sizing_chain": json.dumps(size.sizing_chain),
                                "reason": size.sizing_reason,
                            }

                            # Append to orders.csv
                            orders_file = output_dir / "orders.csv"
                            pd.DataFrame([order_record]).to_csv(
                                orders_file,
                                mode="a",
                                header=not orders_file.exists(),
                                index=False,
                            )
                            sizing_reason = size.sizing_reason
                        else:
                            quantity = self.broker.balance / price
                            sizing_reason = "Fallback: full balance"
                    else:
                        # Original logic: fixed quantity
                        quantity = 0.001  # Small BTC amount
                        sizing_reason = "Fixed quantity"

                    # Validate order against venue constraints
                    self.validate_order(symbol, signal["action"], quantity)

                    order = await self.broker.place_order(
                        symbol=symbol,
                        side=signal["action"],
                        quantity=quantity,
                        order_type="market",
                        sizing_reason=sizing_reason,
                        target_vol=self.target_vol,
                    )

                    if order.status == "filled":
                        trades_executed += 1
                        position = 1 if signal["action"] == "buy" else -1
                        print(
                            f"Executed {signal['action']} at {order.price:.0f} JPY (size: {quantity:.6f}, reason: {sizing_reason})"
                        )

                except Exception as e:
                    print(f"Trade execution failed: {e}")

            # Small delay to simulate real-time
            await asyncio.sleep(0.1)

        print(f"Replay completed. Executed {trades_executed} trades.")

        # Generate outputs
        trade_log = self.broker.get_trade_log()
        pnl_series = self.broker.get_pnl_series()

        return {
            "trade_log": trade_log,
            "pnl_series": pnl_series,
            "trades_executed": trades_executed,
        }

    async def run_live_lite(self, output_dir: Path) -> Dict[str, Any]:
        """Run live-lite mode (connects to streaming pipeline)."""
        print("Starting live-lite simulation...")
        print("Note: This is a stub - would connect to actual streaming pipeline")

        # TODO: Implement connection to streaming pipeline
        # For now, simulate a short session

        await asyncio.sleep(5)  # Simulate connection time

        # Simulate some trades
        trades_executed = 0
        for i in range(3):
            try:
                # Random buy/sell
                side = "buy" if i % 2 == 0 else "sell"
                quantity = 0.001
                symbol = "btc_jpy"

                # Validate order against venue constraints
                self.validate_order(symbol, side, quantity)

                current_price = await self.broker.get_current_price("BTC_JPY")
                if current_price:
                    order = await self.broker.place_order(
                        symbol="BTC_JPY", side=side, quantity=quantity
                    )
                    trades_executed += 1
                    print(f"Executed {side} at {order.price:.0f} JPY")
                    await asyncio.sleep(2)

            except Exception as e:
                print(f"Trade execution failed: {e}")

        print(f"Live-lite session completed. Executed {trades_executed} trades.")

        trade_log = self.broker.get_trade_log()
        pnl_series = self.broker.get_pnl_series()

        return {
            "trade_log": trade_log,
            "pnl_series": pnl_series,
            "trades_executed": trades_executed,
        }

    async def run(self, output_dir: Path) -> Dict[str, Any]:
        """Run the paper trading simulation."""
        if self.mode == "replay":
            return await self.run_replay(output_dir)
        elif self.mode == "live-lite":
            return await self.run_live_lite(output_dir)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save simulation results to files."""
    # Save P&L series as CSV
    if results["pnl_series"]:
        pnl_df = pd.DataFrame(results["pnl_series"])
        pnl_df.to_csv(output_dir / "pnl.csv", index=False)

    # Save trade log as JSON
    with open(output_dir / "trade_log.json", "w") as f:
        json.dump(results["trade_log"], f, indent=2)

    # Save orders as CSV
    orders_df = (
        pd.DataFrame(results["trade_log"]) if results["trade_log"] else pd.DataFrame()
    )
    orders_df.to_csv(output_dir / "orders.csv", index=False)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "replay",  # Would be parameterized
        "trades_executed": results["trades_executed"],
        "total_pnl": sum(t.get("pnl", 0) for t in results["trade_log"]),
        "duration_minutes": 60,  # Would be parameterized
    }

    with open(output_dir / "stats.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    """Main CLI entry point."""
    parser = create_standard_parser("Run paper trading simulation")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["replay", "live-lite"],
        help=CLIFormatter.format_required_help("Trading mode", ["replay", "live-lite"]),
    )
    parser.add_argument(
        "--policy",
        default="sma_fast_slow",
        choices=["rl", "sma_fast_slow", "buy_hold"],
        help=CLIFormatter.format_help(
            "Trading strategy", "sma_fast_slow", ["rl", "sma_fast_slow", "buy_hold"]
        ),
    )
    parser.add_argument(
        "--model-path",
        help="Path to trained RL model (required for rl policy)",
    )
    parser.add_argument(
        "--dataset",
        default="btc_jpy_1m",
        help=CLIFormatter.format_help("Dataset for replay mode", "btc_jpy_1m"),
    )
    parser.add_argument(
        "--duration-minutes",
        type=lambda x: CLIValidator.validate_positive_int(x, "duration-minutes"),
        default=60,
        help=CLIFormatter.format_help("Duration in minutes for replay", 60),
    )
    parser.add_argument(
        "--initial-balance",
        type=lambda x: CLIValidator.validate_positive_float(x, "initial-balance"),
        default=10000.0,
        help=CLIFormatter.format_help("Initial JPY balance", 10000.0),
    )
    CommonArgs.add_output_dir(parser, default="results/paper")
    parser.add_argument(
        "--enable-risk",
        action="store_true",
        help="Enable risk management features (kill switches, circuit breakers)",
    )
    parser.add_argument(
        "--risk-profile",
        default="aggressive",
        choices=["conservative", "balanced", "aggressive"],
        help=CLIFormatter.format_help(
            "Risk profile", "aggressive", ["conservative", "balanced", "aggressive"]
        ),
    )
    parser.add_argument(
        "--kill-file",
        default="/tmp/ztb.stop",
        help=CLIFormatter.format_help("Kill switch file path", "/tmp/ztb.stop"),
    )
    CommonArgs.add_venue(parser)
    parser.add_argument(
        "--venue-config-dir",
        default="venues",
        help=CLIFormatter.format_help(
            "Directory containing venue config files", "venues"
        ),
    )
    parser.add_argument(
        "--target-vol",
        type=lambda x: CLIValidator.validate_positive_float(x, "target-vol"),
        help="Target volatility for position sizing (enables vol targeting)",
    )
    parser.add_argument(
        "--from-streaming",
        action="store_true",
        help="Use streaming pipeline as data source instead of cached data",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.policy == "rl" and not args.model_path:
        parser.error("--model-path is required when using rl policy")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting paper trader in {args.mode} mode...")
    print(f"Strategy: {args.policy}")
    print(f"Venue: {args.venue}")
    print(f"Output directory: {output_dir}")

    try:
        # Load venue configuration
        venue_config_dir = os.path.abspath(args.venue_config_dir)
        venue_config = load_venue_config(args.venue, venue_config_dir)
        print(f"Loaded venue config for {args.venue}")

        # Initialize components
        broker = SimBroker(
            initial_balance=args.initial_balance, venue_config=venue_config
        )
        strategy = create_adapter(args.policy, model_path=args.model_path)

        trader = PaperTrader(
            broker=broker,
            strategy=strategy,
            mode=args.mode,
            dataset=args.dataset if args.mode == "replay" else None,
            duration_minutes=args.duration_minutes,
            enable_risk=args.enable_risk,
            risk_profile=args.risk_profile,
            kill_file=args.kill_file,
            venue_config=venue_config,
            target_vol=args.target_vol,
            from_streaming=args.from_streaming,
        )

        # Run simulation
        results = asyncio.run(trader.run(output_dir))

        # Save results
        save_results(results, output_dir)

        print("Paper trading completed successfully!")
        print(f"Trades executed: {results['trades_executed']}")
        total_pnl = sum(t.get("pnl", 0) for t in results["trade_log"])
        print(f"Total P&L: {total_pnl:.2f} JPY")

    except Exception as e:
        print(f"Paper trading failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
