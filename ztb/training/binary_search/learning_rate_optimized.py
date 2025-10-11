#!/usr/bin/env python3
"""
Binary search optimization for learning_rate parameter in PPO.
Uses the base optimizer class for common functionality.
"""

import sys
from typing import Union

from ztb.utils.path_utils import get_project_root

# Add project root to path for imports
sys.path.insert(0, str(get_project_root()))

from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)


class LearningRateOptimizer(HyperparameterOptimizer):
    """Optimizer for learning_rate parameter."""

    @property
    def parameter_name(self) -> str:
        return "learning_rate"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for learning_rate binary search."""
        return (1e-5, 1e-2)  # Reasonable range for learning rates

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with learning_rate value."""
        self.ppo_params["learning_rate"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize learning_rate parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(
        parser, "learning_rate", float, 5e-4
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = LearningRateOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.learning_rate, args.timesteps)
        print(f"\nFinal score for learning_rate {args.learning_rate}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best learning_rate: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
