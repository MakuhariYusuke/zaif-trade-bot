#!/usr/bin/env python3
"""Transaction cost simulation and analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.evaluation.evaluate import TradingEvaluator  # noqa: E402
from ztb.utils.errors import safe_operation

LOGGER = logging.getLogger(__name__)


def simulate_transaction_costs(
    model_path: Path, cost_range: List[float], data_path: Path, output_dir: Path
) -> Dict[float, Dict[str, float]]:
    """
    Simulate trading performance with different transaction costs.

    Args:
        model_path: Path to trained model
        cost_range: List of transaction costs to test
        data_path: Path to evaluation data
        output_dir: Output directory for results

    Returns:
        Dictionary mapping cost to performance metrics
    """
    return safe_operation(
        logger=LOGGER,
        operation=lambda: _simulate_transaction_costs_impl(model_path, cost_range, data_path, output_dir),
        context="transaction_cost_simulation",
        default_result={},  # Return empty dict on error
    )


def _simulate_transaction_costs_impl(
    model_path: Path, cost_range: List[float], data_path: Path, output_dir: Path
) -> Dict[float, Dict[str, float]]:
    """Implementation of transaction cost simulation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for cost in cost_range:
        LOGGER.info(f"Simulating with transaction cost: {cost}")

        try:
            # Create evaluator with specific transaction cost
            config = {
                "transaction_cost": cost,
                "max_position_size": 1.0,
                "feature_set": "full",
                "timeframe": "1m",
            }
            evaluator = TradingEvaluator(
                model_path=str(model_path),
                data_path=str(data_path),
                config=config,
            )

            # Run evaluation
            metrics = evaluator.evaluate_model()

            results[cost] = {
                "total_return": metrics.get("total_return", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "win_rate": metrics.get("win_rate", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "total_trades": metrics.get("total_trades", 0),
                "avg_trade_return": metrics.get("avg_trade_return", 0),
            }

            LOGGER.info(
                f"Cost {cost}: Return={results[cost]['total_return']:.4f}, "
                f"Sharpe={results[cost]['sharpe_ratio']:.4f}"
            )

        except Exception as e:
            LOGGER.error(f"Failed to evaluate cost {cost}: {e}")
            results[cost] = {"error": str(e)}

    # Save results
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index.name = "transaction_cost"
    results_df.to_csv(output_dir / "transaction_cost_analysis.csv")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Transaction Cost Impact Analysis", fontsize=16)

    valid_costs = [c for c in cost_range if "error" not in results.get(c, {})]
    valid_returns = [results[c]["total_return"] for c in valid_costs]
    valid_sharpe = [results[c]["sharpe_ratio"] for c in valid_costs]
    valid_win_rate = [results[c]["win_rate"] for c in valid_costs]
    valid_trades = [results[c]["total_trades"] for c in valid_costs]

    # Total Return vs Cost
    axes[0, 0].plot(valid_costs, valid_returns, "o-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Transaction Cost")
    axes[0, 0].set_ylabel("Total Return")
    axes[0, 0].set_title("Return vs Transaction Cost")
    axes[0, 0].grid(True, alpha=0.3)

    # Sharpe Ratio vs Cost
    axes[0, 1].plot(
        valid_costs, valid_sharpe, "s-", color="orange", linewidth=2, markersize=8
    )
    axes[0, 1].set_xlabel("Transaction Cost")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].set_title("Sharpe Ratio vs Transaction Cost")
    axes[0, 1].grid(True, alpha=0.3)

    # Win Rate vs Cost
    axes[1, 0].plot(
        valid_costs, valid_win_rate, "^-", color="green", linewidth=2, markersize=8
    )
    axes[1, 0].set_xlabel("Transaction Cost")
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].set_title("Win Rate vs Transaction Cost")
    axes[1, 0].grid(True, alpha=0.3)

    # Total Trades vs Cost
    axes[1, 1].plot(
        valid_costs, valid_trades, "D-", color="red", linewidth=2, markersize=8
    )
    axes[1, 1].set_xlabel("Transaction Cost")
    axes[1, 1].set_ylabel("Total Trades")
    axes[1, 1].set_title("Trading Frequency vs Transaction Cost")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "transaction_cost_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze transaction cost impact on trading performance"
    )
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/ml-dataset-enhanced-balanced.csv"),
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--cost-range",
        nargs="+",
        type=float,
        default=[0.001, 0.002, 0.005, 0.01],
        help="Transaction costs to test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cost_analysis"),
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    LOGGER.info(f"Analyzing transaction costs: {args.cost_range}")
    results = simulate_transaction_costs(
        args.model_path, args.cost_range, args.data_path, args.output_dir
    )

    LOGGER.info("Analysis completed!")
    for cost, metrics in results.items():
        if "error" not in metrics:
            LOGGER.info(
                f"Cost {cost}: Return={metrics['total_return']:.4f}, "
                f"Trades={metrics['total_trades']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
