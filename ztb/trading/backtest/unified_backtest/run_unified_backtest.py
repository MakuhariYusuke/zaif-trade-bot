#!/usr/bin/env python3
"""
Unified Backtest Runner

Command-line interface for running unified backtests with multiple strategies.
Supports SAC models, Action Signal Guide, and comprehensive analysis.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from .unified_backtest import (
    UnifiedBacktester,
    SACStrategy,
    ActionSignalGuideStrategy,
    BacktestAnalyzer,
    BacktestConfig,
    DataManager,
)

def create_sac_strategy(model_path: str, name: str = "SAC_v444") -> SACStrategy:
    """Create SAC strategy instance."""
    return SACStrategy(
        name=name,
        model_path=model_path,
        regime_classifier_path=None,  # Can be added later
    )

def create_action_signal_guide_strategy(
    name: str = "ActionSignalGuide",
    pattern_types: list[str] | None = None
) -> ActionSignalGuideStrategy:
    """Create Action Signal Guide strategy instance."""
    return ActionSignalGuideStrategy(
        name=name,
        pattern_types=pattern_types or ["candlestick", "fibonacci", "wave"]
    )

def load_data(data_path: str) -> pd.DataFrame:
    """Load market data for backtesting."""
    data_manager = DataManager()
    return data_manager.load_data(data_path)

def run_single_backtest(
    strategy_name: str,
    data_path: str,
    config: BacktestConfig,
    output_dir: str = "backtest_results"
) -> None:
    """Run a single strategy backtest."""
    print(f"Running backtest for strategy: {strategy_name}")

    # Initialize backtester
    backtester = UnifiedBacktester()

    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data)} data points")

    # Create and register strategy
    if strategy_name.startswith("SAC"):
        # Assume SAC model path is provided as part of strategy name or config
        model_path = f"models/{strategy_name}.zip"  # Placeholder
        strategy = create_sac_strategy(model_path, strategy_name)
    elif strategy_name == "ActionSignalGuide":
        strategy = create_action_signal_guide_strategy()
    else:
        print(f"Unknown strategy: {strategy_name}")
        return

    backtester.register_strategy(strategy_name, strategy)

    # Run backtest
    result = backtester.run_backtest(strategy_name, data, config)

    # Analyze results
    analyzer = BacktestAnalyzer()
    analysis = analyzer.analyze_single_result(result)

    # Print summary
    print("\n=== Backtest Results ===")
    print(f"Strategy: {result.strategy_name}")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"Total Trades: {result.performance_metrics.total_trades}")
    print(".2f")
    print(".2f")

    print(f"\nResults saved to: {output_dir}")

def run_comparison_backtest(
    strategy_names: list[str],
    data_path: str,
    config: BacktestConfig,
    output_dir: str = "backtest_results"
) -> None:
    """Run comparison backtest for multiple strategies."""
    print(f"Running comparison backtest for strategies: {strategy_names}")

    # Initialize backtester
    backtester = UnifiedBacktester()

    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data)} data points")

    # Create and register strategies
    for strategy_name in strategy_names:
        if strategy_name.startswith("SAC"):
            model_path = f"models/{strategy_name}.zip"  # Placeholder
            strategy = create_sac_strategy(model_path, strategy_name)
        elif strategy_name == "ActionSignalGuide":
            strategy = create_action_signal_guide_strategy()
        else:
            print(f"Unknown strategy: {strategy_name}")
            continue

        backtester.register_strategy(strategy_name, strategy)

    # Run comparison
    results = backtester.compare_strategies(strategy_names, data, config)

    # Analyze results
    analyzer = BacktestAnalyzer()
    comparison_analysis = analyzer.compare_strategies(results)

    # Print comparison summary
    print("\n=== Strategy Comparison ===")
    for strategy_name, result in results.items():
        print(f"\n{strategy_name}:")
        print(".2f")
        print(".2f")
        print(f"  Trades: {result.performance_metrics.total_trades}")

    print(f"\nComparison report saved to: {output_dir}")

def run_advanced_analysis(
    strategy_name: str,
    data_path: str,
    config: BacktestConfig,
    analysis_types: list[str],
    output_dir: str = "backtest_results"
) -> None:
    """Run advanced analysis on backtest results."""
    print(f"Running advanced analysis for strategy: {strategy_name}")
    print(f"Analysis types: {analysis_types}")

    # Initialize backtester
    backtester = UnifiedBacktester()

    # Load data
    data = load_data(data_path)
    print(f"Loaded {len(data)} data points")

    # Create and register strategy
    if strategy_name.startswith("SAC"):
        model_path = f"models/{strategy_name}.zip"
        strategy = create_sac_strategy(model_path, strategy_name)
    elif strategy_name == "ActionSignalGuide":
        strategy = create_action_signal_guide_strategy()
    else:
        print(f"Unknown strategy: {strategy_name}")
        return

    backtester.register_strategy(strategy_name, strategy)

    # Run backtest
    result = backtester.run_backtest(strategy_name, data, config, save_results=False)

    # Run advanced analysis
    advanced_results = backtester.run_advanced_analysis(result, analysis_types)

    # Print results
    print("\n=== Advanced Analysis Results ===")
    for analysis_type, analysis_result in advanced_results.items():
        print(f"\n{analysis_type.upper()}:")
        if "error" in analysis_result:
            print(f"  Error: {analysis_result['error']}")
        else:
            # Print key metrics
            for key, value in analysis_result.items():
                if isinstance(value, (int, float)):
                    print(".4f")
                elif isinstance(value, dict) and len(value) <= 5:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")

    print(f"\nAdvanced analysis completed. Results saved to: {output_dir}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified Backtest Framework")
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name for single backtest"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Strategy names for comparison backtest"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to market data CSV file"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtest"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--analysis-types",
        type=str,
        nargs="+",
        help="Analysis types for advanced analysis (risk_detailed, temporal, regime, feature_importance, walkforward)"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Run advanced analysis instead of basic backtest"
    )

    args = parser.parse_args()

    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=args.initial_capital,
    )

    # Run appropriate analysis
    if args.advanced and args.analysis_types:
        if not args.strategy:
            print("Error: --strategy is required for advanced analysis")
            sys.exit(1)
        run_advanced_analysis(args.strategy, args.data, config, args.analysis_types, args.output_dir)
    elif args.strategy:
        run_single_backtest(args.strategy, args.data, config, args.output_dir)
    elif args.strategies:
        run_comparison_backtest(args.strategies, args.data, config, args.output_dir)
    else:
        print("Error: Must specify either --strategy, --strategies, or --advanced with --analysis-types")
        sys.exit(1)

if __name__ == "__main__":
    main()
