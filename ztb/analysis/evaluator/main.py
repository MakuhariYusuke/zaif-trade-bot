#!/usr/bin/env python3
"""
Main entry point for Trading Evaluator.
"""

import argparse
import warnings

from ztb.evaluation.unified_evaluation import EvaluationType, UnifiedEvaluator
from ztb.evaluation.evaluator.evaluator import TradingEvaluator
from ztb.io.json_io import write_json
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

    try:
        if args.mode == "evaluate":
            evaluator = UnifiedEvaluator(config=config)
            evaluation = evaluator.evaluate_model(
                args.model,
                args.data,
                evaluation_type=EvaluationType.BACKTEST,
            )
            print("\nEvaluation Summary:")
            perf = evaluation.performance_metrics
            print(f"Total Return: {perf.get('total_return', 0):.4f}")
            print(f"Annual Return: {perf.get('annual_return', 0):.4f}")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.4f}")
            print(f"Max Drawdown: {perf.get('max_drawdown', 0):.4f}")
            print(f"Win Rate: {perf.get('win_rate', 0):.4f}")
            print(f"Total Trades: {evaluation.total_trades}")

            # Save results if requested
            if args.output:
                write_json(args.output, evaluation.to_dict(), indent=2, default=str)
                print(f"\nResults saved to {args.output}")

        elif args.mode == "visualize":
            warnings.warn(
                "Visualization is still handled by TradingEvaluator; "
                "UnifiedEvaluator does not implement visualization yet.",
                DeprecationWarning,
                stacklevel=2,
            )
            evaluator = TradingEvaluator(args.model, args.data, config)
            evaluator.create_visualizations()

        elif args.mode == "compare":
            if not args.compare_models:
                print("Error: --compare-models required for comparison mode")
                return
            warnings.warn(
                "Model comparison is still handled by TradingEvaluator; "
                "UnifiedEvaluator does not implement comparison yet.",
                DeprecationWarning,
                stacklevel=2,
            )
            evaluator = TradingEvaluator(args.model, args.data, config)

            evaluator.compare_models(
                args.compare_models,
                args.model_names,
            )

    finally:
        if "evaluator" in locals():
            try:
                evaluator.close()
            except AttributeError:
                pass


if __name__ == "__main__":
    main()
