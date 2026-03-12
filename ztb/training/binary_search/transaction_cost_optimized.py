#!/usr/bin/env python3
"""
Binary search optimization for transaction_cost parameter in environment config.
Uses the base optimizer class for common functionality.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)

class TransactionCostOptimizer(HyperparameterOptimizer):
    """Optimizer for transaction_cost parameter."""

    @property
    def parameter_name(self) -> str:
        return "transaction_cost"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for transaction_cost binary search."""
        return (0.0001, 0.01)  # Reasonable range for transaction costs

    def update_ppo_params(self, value: int | float) -> None:
        """Update environment config with transaction_cost value."""
        self.env_config.transaction_cost = float(value)

def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize transaction_cost parameter for environment"
    )
    BinarySearchArgumentParser.add_parameter_argument(
        parser, "transaction_cost", float, 0.001
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = TransactionCostOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.transaction_cost, args.timesteps)
        print(
            f"\nFinal score for transaction_cost {args.transaction_cost}: {score:.6f}"
        )

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best transaction_cost: {best_value}, Score: {best_score:.6f}"
        )

if __name__ == "__main__":
    main()
