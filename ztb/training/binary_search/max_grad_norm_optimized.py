#!/usr/bin/env python3
"""
Binary search optimization for max_grad_norm parameter in PPO.
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


class MaxGradNormOptimizer(HyperparameterOptimizer):
    """Optimizer for max_grad_norm parameter."""

    @property
    def parameter_name(self) -> str:
        return "max_grad_norm"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for max_grad_norm binary search."""
        return (0.1, 10.0)  # Reasonable range for gradient clipping

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with max_grad_norm value."""
        self.ppo_params["max_grad_norm"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize max_grad_norm parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(
        parser, "max_grad_norm", float, 1.0
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = MaxGradNormOptimizer()

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.max_grad_norm, args.timesteps)
        print(f"\nFinal score for max_grad_norm {args.max_grad_norm}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best max_grad_norm: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
