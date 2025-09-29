"""
Common CLI utilities for ML reinforcement learning experiments.

Provides standardized argument parsing and experiment execution
for ML reinforcement learning experiment scripts.
"""

import argparse
from typing import Any, Dict, Type, Callable

from ztb.training.entrypoints.base_ml_reinforcement import MLReinforcementExperiment


class ExperimentCLI:
    """Common CLI interface for ML reinforcement learning experiments."""

    @staticmethod
    def create_parser(
        description: str,
        default_steps: int,
        experiment_suffix: str
    ) -> argparse.ArgumentParser:
        """Create standardized argument parser for experiments."""
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--strategy",
            choices=["generalization", "aggressive"],
            default="generalization",
            help="Trading strategy (default: generalization)",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=default_steps,
            help=f"Total steps (default: {default_steps})"
        )
        parser.add_argument("--name", help="Experiment name override")
        parser.add_argument(
            "--dataset",
            default="coingecko",
            help="Dataset (default: coingecko)"
        )
        return parser

    @staticmethod
    def create_config(args: argparse.Namespace, experiment_suffix: str) -> Dict[str, Any]:
        """Create experiment configuration from parsed arguments."""
        return {
            "strategy": args.strategy,
            "total_steps": args.steps,
            "name": args.name or f"run{experiment_suffix}_{args.strategy}",
            "dataset": args.dataset,
        }

    @staticmethod
    def run_experiment(
        experiment_class: Type[MLReinforcementExperiment],
        config: Dict[str, Any],
        experiment_suffix: str
    ) -> None:
        """Run experiment and send notifications."""
        experiment = experiment_class(config, total_steps=config["total_steps"])
        result = experiment.execute()

        # 実験完了通知
        logger = experiment.logger_manager
        logger.enqueue_notification(
            f"{experiment_suffix} Experiment completed. Results saved to {experiment.results_dir}"
        )
        logger.enqueue_notification(f"Status: {result.status}")
        logger.enqueue_notification(f"Strategy: {config['strategy']}")
        logger.enqueue_notification(f"Metrics: {result.metrics}")


def create_experiment_main(
    experiment_class: Type[MLReinforcementExperiment],
    description: str,
    default_steps: int,
    experiment_suffix: str
) -> Callable[[], None]:
    """Create a main function for an experiment."""
    def main() -> None:
        parser = ExperimentCLI.create_parser(description, default_steps, experiment_suffix)
        args = parser.parse_args()
        config = ExperimentCLI.create_config(args, experiment_suffix)
        ExperimentCLI.run_experiment(experiment_class, config, experiment_suffix)

    return main