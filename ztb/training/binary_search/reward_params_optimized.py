#!/usr/bin/env python3
"""
Binary search optimization for reward function parameters.
Uses the base optimizer class for common functionality.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from ztb.utils.path_utils import get_project_root

# Add project root to path for imports
sys.path.insert(0, str(get_project_root()))

from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)


class RewardParamsOptimizer(HyperparameterOptimizer):
    """Optimizer for reward function parameters."""

    def __init__(self, project_root: Optional[Path] = None):
        super().__init__(project_root)
        # Reward parameters to optimize
        self.reward_multipliers = [1.0, 1.0, 1.0]  # [hold, buy, sell] multipliers

    @property
    def parameter_name(self) -> str:
        return "reward_multipliers"

    def get_parameter_range(self) -> tuple[float, float]:
        """Get the range for reward multiplier binary search."""
        return (0.1, 5.0)  # Reasonable range for reward multipliers

    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update environment config with reward multiplier value."""
        # For simplicity, we'll optimize one multiplier at a time
        # In practice, you might want to optimize combinations
        self.reward_multipliers = [float(value), float(value), float(value)]
        self.env_config.reward_profit_bonus_multipliers = list(self.reward_multipliers)
    def evaluate_result(
        self, callback: Any
    ) -> Tuple[float, Dict[str, Union[int, float]], Dict[str, Union[int, float]]]:
        """Evaluate training result with focus on action balance."""
        stats = callback.get_training_stats()
        action_dist = callback.get_action_distribution()

        hold_pct = float(action_dist.get("hold_pct", 0.0))
        buy_pct = float(action_dist.get("buy_pct", 0.0))
        sell_pct = float(action_dist.get("sell_pct", 0.0))

        ideal_pct = 33.3
        balance_score = (
            abs(hold_pct - ideal_pct)
            + abs(buy_pct - ideal_pct)
            + abs(sell_pct - ideal_pct)
        )

        reward_score = float(stats.get("avg_reward", 0.0))
        combined_score = reward_score - balance_score * 0.01

        return combined_score, stats, action_dist


def main() -> None:
    parser = BinarySearchArgumentParser.create_parser(
        "Optimize reward parameters for balanced action distribution"
    )
    parser.add_argument(
        "--reward_multiplier",
        type=float,
        default=1.0,
        help="Reward multiplier value for single test",
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = RewardParamsOptimizer()
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Run single test
        score = optimizer.run_single_test(args.reward_multiplier, args.timesteps)
        print(
            f"\nFinal score for reward_multiplier {args.reward_multiplier}: {score:.6f}"
        )

    elif args.mode == "binary":
        # Run binary search optimization
        best_value, best_score = optimizer.binary_search_optimize(
            args.max_iterations, args.timesteps
        )
        print(
            f"\nOptimization complete. Best reward_multiplier: {best_value}, Score: {best_score:.6f}"
        )


if __name__ == "__main__":
    main()
