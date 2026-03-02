#!/usr/bin/env python3
"""
Binary search optimization for reward_scaling parameter in environment config.
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

class RewardScalingOptimizer(HyperparameterOptimizer):
    """Optimizer for reward_scaling parameter."""

    @property
    def parameter_name(self) -> str:
        return "reward_scaling"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for reward_scaling binary search."""
        return (0.1, 10.0)  # Reasonable range for reward scaling

    def update_ppo_params(self, value: int | float) -> None:
        """Update environment config with reward_scaling value."""
        self.env_config.reward_scaling = float(value)

def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize reward_scaling parameter for environment"
    )
    BinarySearchArgumentParser.add_parameter_argument(
        parser, "reward_scaling", float, 6.0
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = RewardScalingOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.reward_scaling, args.timesteps)
        print(f"\nFinal score for reward_scaling {args.reward_scaling}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best reward_scaling: {best_value}, Score: {best_score:.6f}"
        )

if __name__ == "__main__":
    main()
