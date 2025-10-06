#!/usr/bin/env python3
"""
Binary search optimization for n_epochs parameter in PPO.
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


class NEpochsOptimizer(HyperparameterOptimizer):
    """Optimizer for n_epochs parameter."""

    @property
    def parameter_name(self) -> str:
        return "n_epochs"

    def get_parameter_range(self) -> tuple[int, int]:
        """Get the range for n_epochs binary search."""
        return (4, 20)  # Reasonable range for number of epochs

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with n_epochs value."""
        self.ppo_params["n_epochs"] = int(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize n_epochs parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "n_epochs", int, 10)

    args = parser.parse_args()

    # Create optimizer
    optimizer = NEpochsOptimizer()

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.n_epochs, args.timesteps)
        print(f"\nFinal score for n_epochs {args.n_epochs}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best n_epochs: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
