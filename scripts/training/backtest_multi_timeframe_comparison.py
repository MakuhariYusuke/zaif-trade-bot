#!/usr/bin/env python3
"""
Multi-Timeframe Feature Backtest Comparison Script

Backtests SAC v435 models with and without multi-timeframe features to evaluate performance impact.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config_utils import load_config_from_json
from ztb.analysis.sac_backtester import SACBacktester
from ztb.trading.environment.schema_env_factory import create_env_from_schema
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_backtest_for_config(
    config_name: str, output_dir: str = "results/backtest_multi_timeframe"
) -> dict:
    """
    Run backtest for a specific configuration.

    Args:
        config_name: Name of the configuration file (without .json)
        output_dir: Output directory for results

    Returns:
        Backtest results summary
    """
    config_path = f"config/{config_name}.json"
    model_path = "models/v435_unified/sac_v435_final.zip"

    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"🚀 Starting backtest for {config_name}")
    logger.info(f"📁 Config: {config_path}")
    logger.info(f"🤖 Model: {model_path}")

    try:
        # Load configuration
        config_data = load_config_from_json(config_path)
        logger.info("✅ Configuration loaded successfully")

        # Extract configurations
        training_config = config_data.get("training", {})
        env_config = training_config.get("environment", {})
        data_config = training_config.get("data_config", {})

        # Load data
        data_path = data_config.get("csv_path", "btc_jpy_featured_dataset.csv")
        if not os.path.exists(data_path):
            # Try alternative path
            data_path = f"data/{data_path}"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

        import pandas as pd

        df = pd.read_csv(data_path)
        logger.info("✅ Data loaded successfully")

        # Create environment
        env = create_env_from_schema(
            model_name=training_config.get("model_name", "sac_v435"),
            df=df,
            config=env_config,
        )
        logger.info("✅ Environment created successfully")

        # Create backtester
        backtester = SACBacktester(model_path=model_path, config_path=config_path)
        # Override environment
        backtester.env = env
        logger.info("✅ Backtester created successfully")

        # Run backtest
        logger.info("🏃 Running backtest...")
        results = backtester.run_backtest(data_path=data_path, deterministic=True)

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/backtest_{config_name}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Results saved to: {output_file}")

        # Extract key metrics
        summary = {
            "config": config_name,
            "model_path": model_path,
            "timestamp": timestamp,
            "total_return": results.get("total_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "total_trades": results.get("total_trades", 0),
            "avg_trade_return": results.get("avg_trade_return", 0),
            "success": True,
            "output_file": output_file,
        }

        logger.info(f"✅ Backtest completed for {config_name}")
        logger.info(".2%")
        logger.info(".2f")
        logger.info(".1%")
        logger.info(".1%")

        return summary

    except Exception as e:
        logger.error(f"❌ Backtest failed for {config_name}: {e}")
        return {
            "config": config_name,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }


def compare_results(results: list) -> dict:
    """
    Compare backtest results between configurations.

    Args:
        results: List of backtest result summaries

    Returns:
        Comparison analysis
    """
    if len(results) != 2:
        return {"error": "Expected exactly 2 results for comparison"}

    # Separate results
    multi_timeframe_result = None
    no_multi_timeframe_result = None

    for result in results:
        if "no_multi_timeframe" in result["config"]:
            no_multi_timeframe_result = result
        else:
            multi_timeframe_result = result

    if not multi_timeframe_result or not no_multi_timeframe_result:
        return {
            "error": "Could not identify multi-timeframe and non-multi-timeframe results"
        }

    # Calculate differences
    comparison = {
        "multi_timeframe_enabled": multi_timeframe_result,
        "multi_timeframe_disabled": no_multi_timeframe_result,
        "differences": {
            "total_return_diff": multi_timeframe_result["total_return"]
            - no_multi_timeframe_result["total_return"],
            "sharpe_ratio_diff": multi_timeframe_result["sharpe_ratio"]
            - no_multi_timeframe_result["sharpe_ratio"],
            "max_drawdown_diff": multi_timeframe_result["max_drawdown"]
            - no_multi_timeframe_result["max_drawdown"],
            "win_rate_diff": multi_timeframe_result["win_rate"]
            - no_multi_timeframe_result["win_rate"],
            "total_trades_diff": multi_timeframe_result["total_trades"]
            - no_multi_timeframe_result["total_trades"],
        },
        "analysis": {
            "multi_timeframe_better_return": multi_timeframe_result["total_return"]
            > no_multi_timeframe_result["total_return"],
            "multi_timeframe_better_sharpe": multi_timeframe_result["sharpe_ratio"]
            > no_multi_timeframe_result["sharpe_ratio"],
            "multi_timeframe_lower_drawdown": multi_timeframe_result["max_drawdown"]
            < no_multi_timeframe_result["max_drawdown"],
            "multi_timeframe_higher_win_rate": multi_timeframe_result["win_rate"]
            > no_multi_timeframe_result["win_rate"],
        },
    }

    return comparison


def main():
    """Main backtest comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare backtests with/without multi-timeframe features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/backtest_multi_timeframe",
        help="Output directory for results",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=[
            "sac_v435_unified_config",
            "sac_v435_unified_config_no_multi_timeframe",
        ],
        help="Configuration files to test",
    )

    args = parser.parse_args()

    logger.info("🎯 Starting Multi-Timeframe Feature Backtest Comparison")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"⚙️ Configurations: {args.configs}")

    # Run backtests
    results = []
    for config in args.configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing configuration: {config}")
        logger.info(f"{'='*60}")

        result = run_backtest_for_config(config, args.output_dir)
        results.append(result)

    # Compare results
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON ANALYSIS")
    logger.info(f"{'='*80}")

    comparison = compare_results(results)

    if "error" in comparison:
        logger.error(f"❌ Comparison failed: {comparison['error']}")
        return

    # Display comparison
    mt_enabled = comparison["multi_timeframe_enabled"]
    mt_disabled = comparison["multi_timeframe_disabled"]
    diff = comparison["differences"]
    analysis = comparison["analysis"]

    print("\n📊 BACKTEST COMPARISON RESULTS")
    print("=" * 80)
    print(
        f"{'Metric':<20} {'Multi-Timeframe':<20} {'No Multi-Timeframe':<20} {'Difference':<20}"
    )
    print(f"{'':<20} {'Enabled':<20} {'Disabled':<20} {'':<20}")
    print("-" * 80)
    print(
        f"{'Total Return':<20} {mt_enabled['total_return']:<20.2%} {mt_disabled['total_return']:<20.2%} {diff['total_return_diff']:<20.2%}"
    )
    print(
        f"{'Sharpe Ratio':<20} {mt_enabled['sharpe_ratio']:<20.2f} {mt_disabled['sharpe_ratio']:<20.2f} {diff['sharpe_ratio_diff']:<20.2f}"
    )
    print(
        f"{'Max Drawdown':<20} {mt_enabled['max_drawdown']:<20.2%} {mt_disabled['max_drawdown']:<20.2%} {diff['max_drawdown_diff']:<20.2%}"
    )
    print(
        f"{'Win Rate':<20} {mt_enabled['win_rate']:<20.1%} {mt_disabled['win_rate']:<20.1%} {diff['win_rate_diff']:<20.1%}"
    )
    print(
        f"{'Total Trades':<20} {mt_enabled['total_trades']:<20} {mt_disabled['total_trades']:<20} {diff['total_trades_diff']:<20}"
    )
    print("-" * 80)
    print("<20")
    print("<20")
    print("<20")
    print("<20")
    print("<20")

    print("\n🎯 ANALYSIS:")
    print(
        f"  • Multi-timeframe {'IMPROVES' if analysis['multi_timeframe_better_return'] else 'WORSENS'} total return"
    )
    print(
        f"  • Multi-timeframe {'IMPROVES' if analysis['multi_timeframe_better_sharpe'] else 'WORSENS'} risk-adjusted returns"
    )
    print(
        f"  • Multi-timeframe {'REDUCES' if analysis['multi_timeframe_lower_drawdown'] else 'INCREASES'} maximum drawdown"
    )
    print(
        f"  • Multi-timeframe {'IMPROVES' if analysis['multi_timeframe_higher_win_rate'] else 'WORSENS'} win rate"
    )

    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"{args.output_dir}/multi_timeframe_comparison_{timestamp}.json"

    with open(comparison_file, "w") as f:
        json.dump(
            {"results": results, "comparison": comparison, "timestamp": timestamp},
            f,
            indent=2,
            default=str,
        )

    logger.info(f"💾 Comparison saved to: {comparison_file}")
    logger.info("✅ Multi-timeframe backtest comparison completed!")


if __name__ == "__main__":
    main()
