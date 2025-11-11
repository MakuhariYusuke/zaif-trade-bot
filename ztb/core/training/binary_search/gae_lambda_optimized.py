#!/usr/bin/env python3
"""
Binary search optimization for gae_lambda parameter in PPO.
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


class GaeLambdaOptimizer(HyperparameterOptimizer):
    """Optimizer for gae_lambda parameter."""

    @property
    def parameter_name(self) -> str:
        return "gae_lambda"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for gae_lambda binary search."""
        return (0.8, 1.0)  # Reasonable range for GAE lambda

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with gae_lambda value."""
        self.ppo_params["gae_lambda"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize gae_lambda parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "gae_lambda", float, 0.95)

    args = parser.parse_args()

    # Create optimizer
    optimizer = GaeLambdaOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.gae_lambda, args.timesteps)
        print(f"\nFinal score for gae_lambda {args.gae_lambda}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best gae_lambda: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
