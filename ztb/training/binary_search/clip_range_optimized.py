#!/usr/bin/env python3
"""
Binary search optimization for clip_range parameter in PPO.
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


class ClipRangeOptimizer(HyperparameterOptimizer):
    """Optimizer for clip_range parameter."""

    @property
    def parameter_name(self) -> str:
        return "clip_range"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for clip_range binary search."""
        return (0.1, 0.5)  # Reasonable range for PPO clip range

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with clip_range value."""
        self.ppo_params["clip_range"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize clip_range parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "clip_range", float, 0.2)

    args = parser.parse_args()

    # Create optimizer
    optimizer = ClipRangeOptimizer()

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.clip_range, args.timesteps)
        print(f"\nFinal score for clip_range {args.clip_range}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best clip_range: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
