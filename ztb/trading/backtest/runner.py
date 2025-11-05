#!/usr/bin/env python3
"""
Backtest runner CLI.

Executes trading strategy backtests with comprehensive metrics and reporting.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from ztb.utils.observability import generate_correlation_id, setup_observability
from ztb.utils.path_utils import ensure_dir
from ztb.utils.run_metadata import RunMetadata

# Import adapters

# Import risk management components (optional)
try:
    from ..risk.circuit_breakers import get_global_kill_switch  # type: ignore[import-not-found]
    from ..risk.position_sizing import PositionSizer  # type: ignore[import-not-found]

    RISK_AVAILABLE = True
except ImportError:
    RISK_AVAILABLE = False
    get_global_kill_switch = None
    PositionSizer = None
from .adapters import StrategyAdapter, create_adapter
from .metrics import MetricsCalculator
from .report import ReportGenerator

# Import adaptation system
try:
    from ...adaptation import HyperparameterAdaptationSystem

    ADAPTATION_AVAILABLE = True
except ImportError:
    ADAPTATION_AVAILABLE = False
    HyperparameterAdaptationSystem = None


class BacktestEngine:
    """Core backtest execution engine."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 0.0,
        enable_risk: bool = False,
        risk_profile: str = "balanced",
        kill_file: str = "/tmp/ztb.stop",
        target_vol: Optional[float] = None,
        correlation_id: Optional[str] = None,
        enable_adaptation: bool = False,
        adaptation_config: Optional[Dict[str, Any]] = None,
        max_position_size: float = 1.0,
        signal_performance_analyzer: Optional[Any] = None,
    ) -> None:
        """Initialize backtest engine."""
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.enable_risk = enable_risk
        self.risk_profile = risk_profile
        self.kill_file = kill_file
        self.target_vol = target_vol
        self.correlation_id = correlation_id or generate_correlation_id()
        self.enable_adaptation = enable_adaptation and ADAPTATION_AVAILABLE
        self.max_position_size = max_position_size
        self.signal_performance_analyzer = signal_performance_analyzer

        # Initialize position sizer
        if target_vol and RISK_AVAILABLE and PositionSizer:
            self.position_sizer = PositionSizer(target_volatility=target_vol)
        else:
            self.position_sizer = None

        # Initialize kill switch if risk management enabled
        self.kill_switch = None
        if enable_risk and RISK_AVAILABLE and get_global_kill_switch:
            self.kill_switch = get_global_kill_switch()

        # Initialize hyperparameter adaptation system
        self.adaptation_system = None
        if self.enable_adaptation and HyperparameterAdaptationSystem:
            try:
                # Create mock online learning and evaluation components for backtest

                # Mock components for backtest environment
                mock_online_learning = MockOnlineLearningPipeline()
                mock_evaluation_manager = MockEvaluationManager()

                self.adaptation_system = HyperparameterAdaptationSystem(
                    mock_online_learning, mock_evaluation_manager
                )

                # Apply custom configuration if provided
                if adaptation_config:
                    self.adaptation_system.update_config(adaptation_config)

                # Start adaptation system
                if not self.adaptation_system.start():
                    print("Warning: Failed to start hyperparameter adaptation system")
                    self.adaptation_system = None
                    self.enable_adaptation = False
                else:
                    print("Hyperparameter adaptation system started for backtest")

            except Exception as e:
                print(f"Warning: Failed to initialize adaptation system: {e}")
                self.adaptation_system = None
                self.enable_adaptation = False

    def load_data(self, dataset_path: str) -> pd.DataFrame:
        """Load market data from CSV file."""
        try:
            # Load data from CSV file
            data_path = Path(dataset_path)
            if not data_path.exists():
                # Try relative to project root
                data_path = Path(__file__).parent.parent.parent.parent / dataset_path
                if not data_path.exists():
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            print(f"Loading data from: {data_path}")
            data = pd.read_csv(data_path)

            # Ensure timestamp column exists and is datetime
            if "timestamp" not in data.columns:
                # Try common timestamp column names
                timestamp_cols = ["date", "datetime", "time"]
                timestamp_col = None
                for col in timestamp_cols:
                    if col in data.columns:
                        timestamp_col = col
                        break
                if timestamp_col:
                    data = data.rename(columns={timestamp_col: "timestamp"})
                else:
                    raise ValueError("No timestamp column found in dataset")

            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
                data["timestamp"] = pd.to_datetime(data["timestamp"])

            # Filter to 2023 data only for debugging
            # data = data[data["timestamp"].dt.year == 2023].copy()

            print(f"Loaded {len(data)} data points")

            # Ensure required OHLCV columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Set timestamp as index
            data.set_index("timestamp", inplace=True)
            return data

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def run_backtest(
        self, strategy: StrategyAdapter, data: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame, Optional[Dict[str, Any]]]:
        """Run backtest simulation."""

        capital = self.initial_capital
        position = 0  # -1, 0, 1 for short, flat, long
        equity_curve = []
        orders = []
        adaptation_history = [] if self.enable_adaptation else None

        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Progress reporting
            if i % 100 == 0 and i > 0:
                progress = (i / len(data)) * 100
                print(f"Backtest progress: {progress:.1f}% ({i}/{len(data)} steps)")

            current_data = data.iloc[: i + 1]  # All data up to current point

            # Hyperparameter adaptation step
            adaptation_result = None
            if (
                self.enable_adaptation and self.adaptation_system and i > 10
            ):  # Need some history
                try:
                    # Pass market data to adaptation system
                    market_data = current_data.tail(
                        50
                    )  # Last 50 periods for adaptation
                    adaptation_result = self.adaptation_system.adapt_hyperparameters(
                        market_data
                    )

                    # Update strategy parameters if available
                    if hasattr(strategy, "update_hyperparameters"):
                        current_params = (
                            self.adaptation_system.get_current_hyperparameters()
                        )
                        strategy.update_hyperparameters(current_params)

                    # Log adaptation if parameters changed
                    if adaptation_result and adaptation_result["adaptations"]:
                        print(
                            f"[{timestamp}] Adapted {len(adaptation_result['adaptations'])} hyperparameters. "
                            f"Performance improvement: {adaptation_result['performance_improvement']:.4f}"
                        )

                        # Store adaptation result
                        if adaptation_history is not None:
                            adaptation_history.append(
                                {
                                    "timestamp": timestamp,
                                    "adaptations": adaptation_result["adaptations"],
                                    "performance_improvement": adaptation_result[
                                        "performance_improvement"
                                    ],
                                    "confidence": adaptation_result["confidence"],
                                    "market_conditions": adaptation_result.get(
                                        "market_conditions", {}
                                    ),
                                }
                            )

                except Exception as e:
                    print(f"Warning: Adaptation failed at {timestamp}: {e}")

            # Check kill switch if risk management enabled
            if self.enable_risk and self.kill_switch and self.kill_switch.is_killed():
                print(f"Kill switch activated at {timestamp}. Stopping new orders.")
                break  # Stop processing new signals

            # Generate signal
            signal = strategy.generate_signal(current_data, position)

            # Track signal with performance analyzer if available
            if self.signal_performance_analyzer:
                try:
                    self.signal_performance_analyzer.track_signal(
                        timestamp=timestamp,
                        signal_data=signal,
                        market_data=row.to_dict(),
                        position=position
                    )
                except Exception as e:
                    print(f"Warning: Signal tracking failed: {e}")

            # Memory management: clear feature cache periodically to prevent memory leaks
            if hasattr(strategy, "clear_feature_cache") and i % 100 == 0 and i > 0:
                strategy.clear_feature_cache()

            # Execute trade if signal
            if signal["action"] in ["buy", "sell"]:
                price = row["close"] * (1 + self.slippage_bps / 10000)  # Apply slippage

                # Calculate position size
                if self.position_sizer:
                    # Use position sizer
                    signals = {"BTC_JPY": 1.0 if signal["action"] == "buy" else -1.0}
                    current_prices = {"BTC_JPY": price}
                    asset_vols = {"BTC_JPY": 0.5}  # Simplified volatility assumption

                    sizes = self.position_sizer.calculate_position_sizes(
                        signals, current_prices, capital, asset_vols
                    )

                    if sizes:
                        size = sizes[0]
                        shares = size.quantity
                        sizing_reason = size.sizing_reason
                    else:
                        # Fallback: use available capital, but prevent division by zero
                        if price > 0:
                            shares = capital / price
                        else:
                            shares = 0
                        sizing_reason = "Fallback: full capital"
                else:
                    # Original logic: use max_position_size limit, but prevent division by zero
                    if price > 0:
                        max_shares = (capital * self.max_position_size) / price
                        shares = min(
                            max_shares, capital / price
                        )  # Don't exceed available capital
                    else:
                        shares = 0
                    sizing_reason = f"Max position size: {self.max_position_size}"

                # Apply commission
                commission = shares * price * (self.commission_bps / 10000)
                effective_price = price + (commission / shares if shares > 0 else 0)

                if signal["action"] == "buy" and position <= 0:
                    # Buy to long
                    if shares > 0 and capital >= effective_price * shares:
                        order = {
                            "timestamp": timestamp,
                            "action": "buy",
                            "price": effective_price,
                            "shares": shares,
                            "notional": shares * effective_price,
                            "position_before": position,
                            "position_after": 1,
                            "sizing_reason": sizing_reason,
                            "pnl": 0.0,  # Will be calculated on close
                        }
                        position = 1
                        capital -= shares * effective_price  # Deduct from capital
                        orders.append(order)

                        # Record trade outcome with signal performance analyzer
                        if self.signal_performance_analyzer:
                            try:
                                self.signal_performance_analyzer.record_trade_outcome(
                                    signal_timestamp=timestamp,
                                    trade_data=order,
                                    outcome="executed"
                                )
                            except Exception as e:
                                print(f"Warning: Trade outcome recording failed: {e}")
                    else:
                        print(f"Insufficient capital for buy order at {timestamp}")

                        # Record failed trade with signal performance analyzer
                        if self.signal_performance_analyzer:
                            try:
                                self.signal_performance_analyzer.record_trade_outcome(
                                    signal_timestamp=timestamp,
                                    trade_data={
                                        "timestamp": timestamp,
                                        "action": "buy",
                                        "price": effective_price,
                                        "shares": shares,
                                        "reason": "insufficient_capital"
                                    },
                                    outcome="failed"
                                )
                            except Exception as e:
                                print(f"Warning: Failed trade recording failed: {e}")

                elif signal["action"] == "sell" and position >= 0:
                    # Sell to short or close long
                    if position == 1:
                        # Close long position
                        entry_order = next(
                            (o for o in reversed(orders) if o["action"] == "buy"), None
                        )
                        if entry_order:
                            pnl = (
                                effective_price - entry_order["price"]
                            ) * entry_order["shares"] - commission
                            capital += (
                                entry_order["notional"] + pnl
                            )  # Restore capital + pnl

                    if shares > 0:
                        order = {
                            "timestamp": timestamp,
                            "action": "sell",
                            "price": effective_price,
                            "shares": shares,
                            "notional": shares * effective_price,
                            "position_before": position,
                            "position_after": -1 if position == 0 else 0,
                            "sizing_reason": sizing_reason,
                            "pnl": pnl if "pnl" in locals() else 0.0,
                        }
                        position = -1 if position == 0 else 0
                        capital += (
                            shares * effective_price - commission
                        )  # Add proceeds minus commission
                        orders.append(order)

                        # Record trade outcome with signal performance analyzer
                        if self.signal_performance_analyzer:
                            try:
                                self.signal_performance_analyzer.record_trade_outcome(
                                    signal_timestamp=timestamp,
                                    trade_data=order,
                                    outcome="executed"
                                )
                            except Exception as e:
                                print(f"Warning: Trade outcome recording failed: {e}")

            # Record equity
            if position == 0:
                current_equity = capital
            elif position == 1:
                # Long position: capital + (current_price * shares)
                entry_order = next(
                    (o for o in reversed(orders) if o["action"] == "buy"), None
                )
                if entry_order:
                    current_equity = capital + (row["close"] * entry_order["shares"])
                else:
                    current_equity = capital  # Fallback if no entry order found
            elif position == -1:
                # Short position: capital + (entry_price * shares) - (current_price * shares)
                entry_order = next(
                    (o for o in reversed(orders) if o["action"] == "sell"), None
                )
                if entry_order:
                    current_equity = (
                        capital
                        + (entry_order["price"] * entry_order["shares"])
                        - (row["close"] * entry_order["shares"])
                    )
                else:
                    current_equity = capital  # Fallback if no entry order found
            else:
                current_equity = capital  # Fallback for unexpected position

            # Ensure equity doesn't go negative (bankruptcy protection)
            current_equity = max(current_equity, 0)

            timestamp_str = (
                timestamp.isoformat()
                if hasattr(timestamp, "isoformat")
                else str(timestamp)
            )
            equity_curve.append(
                {
                    "timestamp": timestamp_str,
                    "equity": current_equity,
                }
            )

        # Convert to pandas objects
        equity_df = pd.DataFrame(equity_curve)
        equity_series = pd.Series(
            [p["equity"] for p in equity_curve], index=data.index[: len(equity_curve)]
        )

        orders_df = pd.DataFrame(orders)

        # Prepare adaptation summary if adaptation was enabled
        adaptation_summary = None
        if adaptation_history is not None and adaptation_history:
            adaptation_summary = {
                "total_adaptations": len(adaptation_history),
                "adaptation_history": adaptation_history,
                "final_hyperparameters": self.adaptation_system.get_current_hyperparameters()
                if self.adaptation_system
                else {},
                "adaptation_statistics": self.adaptation_system.get_adaptation_statistics()
                if self.adaptation_system
                else {},
            }

        # Prepare signal performance summary if analyzer was provided
        signal_performance_summary = None
        if self.signal_performance_analyzer:
            try:
                signal_performance_summary = self.signal_performance_analyzer.get_performance_report()
            except Exception as e:
                print(f"Warning: Failed to generate signal performance summary: {e}")

        return equity_series, orders_df, adaptation_summary, signal_performance_summary


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run trading strategy backtest")
    parser.add_argument(
        "--policy",
        required=True,
        choices=["rl", "sma_fast_slow", "buy_hold"],
        help="Trading strategy to test",
    )
    parser.add_argument(
        "--model-path",
        help="Path to RL model file (required for rl policy)",
    )
    parser.add_argument(
        "--dataset", default="btc_usd_1m", help="Dataset to use (default: btc_usd_1m)"
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/backtest",
        help="Output directory (default: results/backtest)",
    )
    parser.add_argument(
        "--enable-risk",
        action="store_true",
        help="Enable risk management features (kill switches, circuit breakers)",
    )
    parser.add_argument(
        "--risk-profile",
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Risk profile (default: balanced)",
    )
    parser.add_argument(
        "--kill-file",
        default="/tmp/ztb.stop",
        help="Kill switch file path (default: /tmp/ztb.stop)",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        help="Target volatility for position sizing (enables vol targeting)",
    )
    parser.add_argument(
        "--enable-adaptation",
        action="store_true",
        help="Enable dynamic hyperparameter adaptation during backtest",
    )
    parser.add_argument(
        "--adaptation-interval",
        type=int,
        default=15,
        help="Adaptation interval in minutes (default: 15)",
    )
    parser.add_argument(
        "--adaptation-safety-margin",
        type=float,
        default=0.15,
        help="Safety margin for parameter changes (default: 0.15)",
    )
    parser.add_argument(
        "--auto-analyze",
        action="store_true",
        help="Automatically run analyze_backtest after backtest completion",
    )

    args = parser.parse_args()

    # Generate correlation ID for this run
    correlation_id = generate_correlation_id()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.policy}_{timestamp}"
    ensure_dir(output_dir)

    print(f"Running backtest for {args.policy} strategy...")
    print(f"Output directory: {output_dir}")
    print(f"Correlation ID: {correlation_id}")

    # Setup observability
    obs_client = setup_observability("backtest", output_dir, correlation_id)

    try:
        # Initialize components
        adaptation_config = None
        if args.enable_adaptation:
            adaptation_config = {
                "hyperparameter_config": {
                    "adaptation_interval_minutes": args.adaptation_interval,
                    "safety_margin": args.adaptation_safety_margin,
                }
            }

        engine = BacktestEngine(
            initial_capital=args.initial_capital,
            slippage_bps=args.slippage_bps,
            enable_risk=args.enable_risk,
            risk_profile=args.risk_profile,
            kill_file=args.kill_file,
            target_vol=args.target_vol,
            correlation_id=correlation_id,
            enable_adaptation=args.enable_adaptation,
            adaptation_config=adaptation_config,
        )

        # Create strategy adapter
        if args.policy == "rl":
            if not args.model_path:
                parser.error("--model-path is required when using rl policy")
            strategy = create_adapter(args.policy, model_path=args.model_path)
        else:
            strategy = create_adapter(args.policy)
        data = engine.load_data(args.dataset)

        # Run backtest
        equity_curve, orders, adaptation_summary = engine.run_backtest(strategy, data)

        # Log adaptation summary if available
        if adaptation_summary:
            print("Adaptation Summary:")
            print(f"  Total adaptations: {adaptation_summary['total_adaptations']}")
            print(
                f"  Final hyperparameters: {adaptation_summary['final_hyperparameters']}"
            )
            if adaptation_summary["adaptation_statistics"]:
                stats = adaptation_summary["adaptation_statistics"]
                print(
                    f"  Strategy performance: {stats.get('strategy_performance', {})}"
                )

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            equity_curve=equity_curve,
            orders=orders,
            initial_capital=args.initial_capital,
            slippage_bps=args.slippage_bps,
        )

        # Create run metadata
        run_metadata = RunMetadata()
        run_metadata.metadata.update(
            {
                "correlation_id": correlation_id,
                "run_id": f"backtest_{args.policy}_{timestamp}",
                "type": "backtest",
                "config": {
                    "policy": args.policy,
                    "dataset": args.dataset,
                    "slippage_bps": args.slippage_bps,
                    "initial_capital": args.initial_capital,
                    "enable_risk": args.enable_risk,
                    "risk_profile": args.risk_profile,
                    "target_vol": args.target_vol,
                    "enable_adaptation": args.enable_adaptation,
                    "adaptation_interval": args.adaptation_interval
                    if args.enable_adaptation
                    else None,
                    "adaptation_safety_margin": args.adaptation_safety_margin
                    if args.enable_adaptation
                    else None,
                },
                "seeds": {
                    "numpy": 42,  # From load_data
                    "random": None,
                },
                "package_hashes": {},  # TODO: Add package hashes
            }
        )
        # Add system info
        run_metadata.metadata["environment"] = run_metadata.capture_system_info()

        # Save run metadata
        obs_client.export_artifact("run_metadata", run_metadata.to_dict())

        # Generate reports
        metadata = {
            "strategy": args.policy,
            "dataset": args.dataset,
            "slippage_bps": args.slippage_bps,
            "initial_capital": args.initial_capital,
            "adaptation_enabled": args.enable_adaptation,
            "adaptation_summary": adaptation_summary,
        }

        equity_list = [
            {"timestamp": ts, "equity": eq} for ts, eq in equity_curve.items()
        ]
        orders_list = cast(
            List[Dict[str, Any]], orders.to_dict("records") if not orders.empty else []
        )

        # Generate outputs
        ReportGenerator.generate_json_report(
            metrics,
            equity_list,
            orders_list,
            metadata,
            str(output_dir / "metrics.json"),
        )

        ReportGenerator.generate_markdown_report(
            metrics, metadata, str(output_dir / "report.md")
        )

        ReportGenerator.generate_equity_csv(
            equity_list, str(output_dir / "equity_curve.csv")
        )

        ReportGenerator.generate_orders_csv(orders_list, str(output_dir / "orders.csv"))

        print("Backtest completed successfully!")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Win Rate: {metrics.win_rate:.1%}")
        print(f"Total Trades: {metrics.total_trades}")

        # Auto-run analyze_backtest if requested
        if args.auto_analyze:
            print("\nRunning automatic backtest analysis...")
            try:
                # Import and run analyze_backtest
                import sys

                from ztb.analysis.unified_analyze import UnifiedAnalysisSuite

                # Create analysis suite
                suite = UnifiedAnalysisSuite()

                # Create mock args for analyze_backtest
                class MockArgs:
                    def __init__(self, backtest_path):
                        self.backtest_path = backtest_path

                mock_args = MockArgs(str(output_dir))

                # Run analyze_backtest
                result = suite.run_analyze_backtest(mock_args)
                if result == 0:
                    print("Backtest analysis completed successfully!")
                else:
                    print("Backtest analysis failed!")

            except Exception as e:
                print(f"Automatic backtest analysis failed: {e}")

    except Exception as e:
        print(f"Backtest failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# Mock classes for backtest adaptation integration
class MockOnlineLearningPipeline:
    """Mock online learning pipeline for backtest environment."""

    def __init__(self):
        self.hyperparameters = {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "regularization_strength": 1e-5,
            "dropout_rate": 0.1,
        }

    def update_hyperparameter(self, name: str, value: float) -> None:
        """Update hyperparameter value."""
        self.hyperparameters[name] = value
        print(f"Updated hyperparameter {name} = {value}")

    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()


class MockEvaluationManager:
    """Mock evaluation manager for backtest environment."""

    def __init__(self):
        self.performance_score = 0.5

    def get_current_performance(self) -> float:
        """Get current performance score."""
        return self.performance_score

    def update_performance(self, score: float) -> None:
        """Update performance score."""
        self.performance_score = score
