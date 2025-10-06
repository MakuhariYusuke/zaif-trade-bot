#!/usr/bin/env python3
"""
Binary search optimization for gamma parameter in PPO.
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


class GammaOptimizer(HyperparameterOptimizer):
    """Optimizer for gamma parameter."""

    @property
    def parameter_name(self) -> str:
        return "gamma"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for gamma binary search."""
        return (0.8, 0.99)  # Reasonable range for discount factor

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with gamma value."""
        self.ppo_params["gamma"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize gamma parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "gamma", float, 0.95)

    args = parser.parse_args()

    # Create optimizer
    optimizer = GammaOptimizer()

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.gamma, args.timesteps)
        print(f"\nFinal score for gamma {args.gamma}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best gamma: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
