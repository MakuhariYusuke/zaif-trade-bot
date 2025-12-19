#!/usr/bin/env python3
"""
Binary search optimization for batch_size parameter in PPO.
Uses the base optimizer class for common functionality.
"""

import sys

from ztb.utils.path_utils import get_project_root

# Add project root to path for imports
sys.path.insert(0, str(get_project_root()))

from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)


class BatchSizeOptimizer(HyperparameterOptimizer):
    """Optimizer for batch_size parameter."""

    @property
    def parameter_name(self) -> str:
        return "batch_size"

    def get_parameter_range(self) -> tuple[int, int]:
        """Get the range for batch_size binary search."""
        return (16, 256)  # Reasonable range for batch sizes


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize batch_size parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "batch_size", int, 64)

    args = parser.parse_args()

    # Create optimizer
    optimizer = BatchSizeOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.batch_size, args.timesteps)
        print(f"\nFinal score for batch_size {args.batch_size}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best batch_size: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
