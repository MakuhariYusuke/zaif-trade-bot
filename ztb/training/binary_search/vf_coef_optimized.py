#!/usr/bin/env python3
"""
Binary search optimization for vf_coef parameter in PPO.
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

class VfCoefOptimizer(HyperparameterOptimizer):
    """Optimizer for vf_coef parameter."""

    @property
    def parameter_name(self) -> str:
        return "vf_coef"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for vf_coef binary search."""
        return (0.1, 1.0)  # Reasonable range for value function coefficient

def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize vf_coef parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "vf_coef", float, 0.5)

    args = parser.parse_args()

    # Create optimizer
    optimizer = VfCoefOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.vf_coef, args.timesteps)
        print(f"\nFinal score for vf_coef {args.vf_coef}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best vf_coef: {best_value}, Score: {best_score:.6f}"
        )

if __name__ == "__main__":
    main()
