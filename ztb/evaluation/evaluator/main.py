#!/usr/bin/env python3
"""
Main entry point for Trading Evaluator.
"""

import argparse

from ztb.evaluation.evaluator.evaluator import TradingEvaluator
from ztb.utils.errors import safe_operation


def main() -> None:
    """メイン関数"""
    safe_operation(
        _main_impl, logger=None, context="Model evaluation execution"
    )  # Will be configured inside


def _main_impl(logger) -> None:
    """Implementation of main function."""
    parser = argparse.ArgumentParser(
        description="Trading RL Model Evaluation and Visualization"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to evaluation data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluate", "visualize", "compare"],
        default="evaluate",
        help="Operation mode",
    )
    parser.add_argument(
        "--compare-models", nargs="+", help="Paths to models for comparison"
    )
    parser.add_argument("--model-names", nargs="+", help="Names for compared models")
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results as JSON",
    )

    args = parser.parse_args()

    # 設定の更新（デフォルト設定を維持しつつ上書き）
    config = {
        "results_dir": "./results/",
        "n_eval_episodes": 20,
        "max_steps_per_episode": 10000,
        "render_mode": None,
        "deterministic": True,
        "plot_style": "seaborn",
    }
    config.update(
        {
            "n_eval_episodes": args.n_episodes,
            "results_dir": "./results/",
        }
    )

    evaluator = TradingEvaluator(args.model, args.data, config)

    try:
        if args.mode == "evaluate":
            stats = evaluator.evaluate_model()
            print("\nEvaluation Summary:")
            print(f"Total Return: {stats.get('total_return', 0):.4f}")
            print(f"Annual Return: {stats.get('annual_return', 0):.4f}")
            print(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.4f}")
            print(f"Max Drawdown: {stats.get('max_drawdown', 0):.4f}")
            print(f"Win Rate: {stats.get('win_rate', 0):.4f}")
            print(f"Total Trades: {stats.get('total_trades', 0)}")

            # Save results if requested
            if args.output:
                import json

                with open(args.output, "w") as f:
                    json.dump(stats, f, indent=2, default=str)
                print(f"\nResults saved to {args.output}")

        elif args.mode == "visualize":
            evaluator.create_visualizations()

        elif args.mode == "compare":
            if not args.compare_models:
                print("Error: --compare-models required for comparison mode")
                return

            evaluator.compare_models(
                args.compare_models,
                args.model_names,
            )

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
