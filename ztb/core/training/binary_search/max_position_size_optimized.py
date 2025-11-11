#!/usr/bin/env python3
"""
Binary search optimization for max_position_size parameter in environment config.
Uses the base optimizer class for common functionality.
"""

import sys
from pathlib import Path
from typing import Union

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)


class MaxPositionSizeOptimizer(HyperparameterOptimizer):
    """Optimizer for max_position_size parameter."""

    @property
    def parameter_name(self) -> str:
        return "max_position_size"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for max_position_size binary search."""
        return (0.1, 2.0)  # Reasonable range for position sizes

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update environment config with max_position_size value."""
        self.env_config.max_position_size = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize max_position_size parameter for environment"
    )
    BinarySearchArgumentParser.add_parameter_argument(
        parser, "max_position_size", float, 1.0
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = MaxPositionSizeOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.max_position_size, args.timesteps)
        print(
            f"\nFinal score for max_position_size {args.max_position_size}: {score:.6f}"
        )

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best max_position_size: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
