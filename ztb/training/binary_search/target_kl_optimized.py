#!/usr/bin/env python3
"""
Binary search optimization for target_kl parameter in PPO.
Uses the base optimizer class for common functionality.
"""

import sys
from typing import Union

from ztb.utils.path_utils import get_project_root

# Add project root to path for imports
sys.path.insert(0, str(get_project_root()))

from ztb.training.binary_search.base_optimizer import BinarySearchArgumentParser, HyperparameterOptimizer


class TargetKLOptimizer(HyperparameterOptimizer):
    """Optimizer for target_kl parameter."""

    @property
    def parameter_name(self) -> str:
        return "target_kl"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for target_kl binary search."""
        return (0.001, 0.1)  # Reasonable range for target KL divergence

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with target_kl value."""
        self.ppo_params["target_kl"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser('Optimize target_kl parameter for PPO')
    BinarySearchArgumentParser.add_parameter_argument(parser, 'target_kl', float, 0.01)

    args = parser.parse_args()

    # Create optimizer
    optimizer = TargetKLOptimizer()

    if args.mode == 'single':
        # Run single test
        score = optimizer.run_single_test(args.target_kl, args.timesteps)
        print(f"\nFinal score for target_kl {args.target_kl}: {score:.6f}")

    elif args.mode == 'binary':
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(args.max_iterations, args.timesteps)
        print(f"\nOptimization complete. Best target_kl: {best_value}, Score: {best_score:.6f}")


if __name__ == "__main__":
    main()