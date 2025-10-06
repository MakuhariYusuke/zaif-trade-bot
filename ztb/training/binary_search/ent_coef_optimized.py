#!/usr/bin/env python3
"""
Binary search optimization for ent_coef parameter in PPO.
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


class EntCoefOptimizer(HyperparameterOptimizer):
    """Optimizer for ent_coef parameter."""

    @property
    def parameter_name(self) -> str:
        return "ent_coef"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for ent_coef binary search."""
        return (0.001, 0.1)  # Reasonable range for entropy coefficient

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with ent_coef value."""
        self.ppo_params["ent_coef"] = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize ent_coef parameter for PPO"
    )
    BinarySearchArgumentParser.add_parameter_argument(parser, "ent_coef", float, 0.01)

    args = parser.parse_args()

    # Create optimizer
    optimizer = EntCoefOptimizer()

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.ent_coef, args.timesteps)
        print(f"\nFinal score for ent_coef {args.ent_coef}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best ent_coef: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
