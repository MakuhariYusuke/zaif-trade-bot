#!/usr/bin/env python3
"""
Binary search optimization for n_steps parameter in PPO.
Uses the base optimizer class for common functionality.
"""

import argparse
import sys
from pathlib import Path
from typing import Union

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.training.binary_search.base_optimizer import BinarySearchArgumentParser, HyperparameterOptimizer


class NStepsOptimizer(HyperparameterOptimizer):
    """Optimizer for n_steps parameter."""

    @property
    def parameter_name(self) -> str:
        return "n_steps"

    def get_parameter_range(self) -> tuple[int, int]:
        """Get the range for n_steps binary search."""
        return (1024, 4096)  # Reasonable range for number of steps per environment

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with n_steps value."""
        self.ppo_params["n_steps"] = int(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser('Optimize n_steps parameter for PPO')
    BinarySearchArgumentParser.add_parameter_argument(parser, 'n_steps', int, 2048)

    args = parser.parse_args()

    # Create optimizer
    optimizer = NStepsOptimizer()

    if args.mode == 'single':
        # Run single test
        score = optimizer.run_single_test(args.n_steps, args.timesteps)
        print(f"\nFinal score for n_steps {args.n_steps}: {score:.6f}")

    elif args.mode == 'binary':
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(args.max_iterations, args.timesteps)
        print(f"\nOptimization complete. Best n_steps: {best_value}, Score: {best_score:.6f}")


if __name__ == "__main__":
    main()