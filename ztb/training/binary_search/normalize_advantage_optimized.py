#!/usr/bin/env python3
"""
Optimization for normalize_advantage parameter in PPO.
Tests both True and False values to find optimal setting.
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


class NormalizeAdvantageOptimizer(HyperparameterOptimizer):
    """Optimizer for normalize_advantage parameter."""

    @property
    def parameter_name(self) -> str:
        return "normalize_advantage"

    def get_parameter_range(self) -> tuple[bool, bool]:
        """Get the range for normalize_advantage (both True and False)."""
        return (False, True)  # Test both boolean values


def binary_search_optimize(
    self, max_iterations: int = 2, total_timesteps: int = 100000
) -> tuple[bool, float]:
    """
    Override binary search to test both True and False values.
    Returns (best_value, best_score).
    """
    print("\n=== Testing normalize_advantage parameter ===")
    print("Testing both True and False values...")

    best_value = False
    best_score = float("-inf")

    # Test False
    print("\nTesting normalize_advantage=False")
    score_false = self.run_single_test(False, total_timesteps)

    # Test True
    print("\nTesting normalize_advantage=True")
    score_true = self.run_single_test(True, total_timesteps)

    # Compare results
    if score_false > score_true:
        best_value = False
        best_score = score_false
        print(f"\nFalse performs better: {score_false:.6f} vs {score_true:.6f}")
    else:
        best_value = True
        best_score = score_true
        print(f"\nTrue performs better: {score_true:.6f} vs {score_false:.6f}")

    print(f"\nBest normalize_advantage: {best_value} (score: {best_score:.6f})")
    return best_value, best_score


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize normalize_advantage parameter for PPO"
    )
    parser.add_argument(
        "--normalize_advantage",
        action="store_true",
        default=False,
        help="normalize_advantage value for single test (default: False)",
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = NormalizeAdvantageOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.normalize_advantage, args.timesteps)
        print(
            f"\nFinal score for normalize_advantage {args.normalize_advantage}: {score:.6f}"
        )

    elif args.mode == "binary":
        # Run comparison test for both values
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best normalize_advantage: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
