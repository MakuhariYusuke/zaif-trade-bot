#!/usr/bin/env python3
"""
Binary search optimization for risk_free_rate parameter in environment config.
Uses the base optimizer class for common functionality.
"""

import sys
from pathlib import Path
from typing import Union

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)


class RiskFreeRateOptimizer(HyperparameterOptimizer):
    """Optimizer for risk_free_rate parameter."""

    @property
    def parameter_name(self) -> str:
        return "risk_free_rate"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for risk_free_rate binary search."""
        return (0.0, 0.1)  # Reasonable range for risk-free rates

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update environment config with risk_free_rate value."""
        self.env_config.risk_free_rate = float(value)


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize risk_free_rate parameter for environment"
    )
    BinarySearchArgumentParser.add_parameter_argument(
        parser, "risk_free_rate", float, 0.02
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = RiskFreeRateOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.risk_free_rate, args.timesteps)
        print(f"\nFinal score for risk_free_rate {args.risk_free_rate}: {score:.6f}")

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best risk_free_rate: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
