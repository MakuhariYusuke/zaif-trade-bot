#!/usr/bin/env python3
"""
Lagrange Constraint Parameters Optimizer

Optimizes Lagrange constraint hyperparameters for SELL bias mitigation:
- r_target: Target SELL action rate (0.10-0.25)
- tolerance: Acceptable deviation from target (0.01-0.10)
- eta: Learning rate for dual variable update (0.001-0.1)
- lambda_max: Maximum dual variable value (0.5-5.0)
- warmup_steps: Steps before constraint activation (500-5000)

The optimizer uses binary search to find optimal values that balance:
1. SELL rate close to target (minimize |r_actual - r_target|)
2. Stable training (avoid excessive penalties)
3. Good episode rewards
"""

import inspect
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.training.models.custom_ppo import CustomPPO
from ztb.training.binary_search.base_optimizer import (
    BinarySearchArgumentParser,
    HyperparameterOptimizer,
)
from ztb.training.config.lagrange_defaults import LAGRANGE_DEFAULTS


class LagrangeOptimizerBase(HyperparameterOptimizer):
    """Base class for Lagrange parameter optimizers using CustomPPO."""

    def __init__(self, project_root: Any = None) -> None:
        """Initialize optimizer with full curriculum stage for realistic rewards."""
        super().__init__(project_root)
        # Override curriculum_stage to use full environment (not simple_portfolio)
        self.env_config.curriculum_stage = "full"

    def create_model(self, env: Any) -> Any:
        """Create CustomPPO model with Lagrange parameters."""
        # Wrap environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Extract Lagrange params (set by update_ppo_params)
        lagrange_params = self.ppo_params.pop("lagrange_params", {})
        enable_lagrange = self.ppo_params.pop("enable_lagrange", False)

        # Remove parameters not supported by CustomPPO
        signature = inspect.signature(CustomPPO.__init__)
        valid_params = {
            name
            for name, param in signature.parameters.items()
            if name not in {"self", "policy", "env"}
            and param.kind != inspect.Parameter.VAR_KEYWORD
        }

        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

        ppo_kwargs = {}
        for key, value in self.ppo_params.items():
            if key in {
                "use_sde",
                "sde_sample_freq",
                "reward_scaling",
                "transaction_cost",
                "position_penalty_scale",
                "inventory_penalty_scale",
                "trade_frequency_penalty",
                "total_timesteps",
                "max_position_size",
                "fee_model",
                "fee_rate",
                "features",
            }:
                continue

            if accepts_kwargs or key in valid_params:
                ppo_kwargs[key] = value

        # Create model with CustomPPO (supports Lagrange)
        return CustomPPO(
            "MlpPolicy",
            env,
            enable_lagrange=enable_lagrange,
            lagrange_target_action="SELL",
            lagrange_r_target=lagrange_params.get("r_target", LAGRANGE_DEFAULTS["r_target"]),
            lagrange_tolerance=lagrange_params.get("tolerance", LAGRANGE_DEFAULTS["tolerance"]),
            lagrange_eta=lagrange_params.get("eta", LAGRANGE_DEFAULTS["eta"]),
            lagrange_lambda_max=lagrange_params.get("lambda_max", LAGRANGE_DEFAULTS["lambda_max"]),
            lagrange_warmup_steps=int(lagrange_params.get("warmup_steps", LAGRANGE_DEFAULTS["warmup_steps"])),
            **ppo_kwargs,
        )

    def train_model(self, total_timesteps: int = 100000) -> Tuple[Any, Any, float]:
        """Train model and return model, callback, and elapsed time."""
        from ztb.training.binary_search.base_optimizer import TrainingCallback

        env = self.create_environment()
        model = self.create_model(env)
        callback = TrainingCallback(verbose=int(self.ppo_params.get("verbose", 0)))

        start = time.perf_counter()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            use_masking=False,
        )
        elapsed = time.perf_counter() - start

        return model, callback, elapsed


class LagrangeRTargetOptimizer(LagrangeOptimizerBase):
    """Optimizer for Lagrange target SELL rate (r_target)."""

    @property
    def parameter_name(self) -> str:
        return "lagrange_r_target"

    def get_parameter_range(self) -> Tuple[float, float]:
        """Target SELL rate range: 10% to 25%."""
        return (0.10, 0.25)

    def update_ppo_params(self, value: float) -> None:
        """Update r_target in Lagrange params."""
        if "lagrange_params" not in self.ppo_params:
            self.ppo_params["lagrange_params"] = {}
        self.ppo_params["lagrange_params"]["r_target"] = float(value)
        self.ppo_params["enable_lagrange"] = True

    def evaluate_result(
        self, callback: Any
    ) -> Tuple[float, Dict[str, Union[int, float]], Dict[str, Union[int, float]]]:
        """
        Evaluate training result.
        
        Score is based purely on average episode reward.
        Action distribution balance is monitored but not penalized,
        as Lagrange constraint naturally encourages balanced actions.
        """
        stats = callback.get_training_stats()
        action_dist = callback.get_action_distribution()
        
        avg_reward = float(stats.get("avg_reward", 0.0))
        
        # Log action distribution for monitoring (but don't penalize)
        hold_pct = float(action_dist.get("hold_pct", 0.0))
        buy_pct = float(action_dist.get("buy_pct", 0.0))
        sell_pct = float(action_dist.get("sell_pct", 0.0))
        
        # Note: Using print() here instead of logger because this is user-facing output
        # during optimization runs and should always be visible
        print(
            f"  Action distribution: HOLD {hold_pct:.1f}%, BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%"
        )

        # Score = reward only (no deviation penalty)
        return avg_reward, stats, action_dist


class LagrangeToleranceOptimizer(LagrangeOptimizerBase):
    """Optimizer for Lagrange tolerance (acceptable deviation)."""

    @property
    def parameter_name(self) -> str:
        return "lagrange_tolerance"

    def get_parameter_range(self) -> Tuple[float, float]:
        """Tolerance range: 1% to 10%."""
        return (0.01, 0.10)

    def update_ppo_params(self, value: float) -> None:
        """Update tolerance in Lagrange params."""
        if "lagrange_params" not in self.ppo_params:
            self.ppo_params["lagrange_params"] = {}
        self.ppo_params["lagrange_params"]["tolerance"] = float(value)
        if "r_target" not in self.ppo_params["lagrange_params"]:
            self.ppo_params["lagrange_params"]["r_target"] = 0.15
        self.ppo_params["enable_lagrange"] = True


class LagrangeEtaOptimizer(LagrangeOptimizerBase):
    """Optimizer for Lagrange dual variable learning rate (eta)."""

    @property
    def parameter_name(self) -> str:
        return "lagrange_eta"

    def get_parameter_range(self) -> Tuple[float, float]:
        """Eta range: 0.001 to 0.1."""
        return (0.001, 0.1)

    def update_ppo_params(self, value: float) -> None:
        """Update eta in Lagrange params."""
        if "lagrange_params" not in self.ppo_params:
            self.ppo_params["lagrange_params"] = {}
        self.ppo_params["lagrange_params"]["eta"] = float(value)
        if "r_target" not in self.ppo_params["lagrange_params"]:
            self.ppo_params["lagrange_params"]["r_target"] = 0.15
        self.ppo_params["enable_lagrange"] = True


class LagrangeLambdaMaxOptimizer(LagrangeOptimizerBase):
    """Optimizer for Lagrange maximum dual variable value (lambda_max)."""

    @property
    def parameter_name(self) -> str:
        return "lagrange_lambda_max"

    def get_parameter_range(self) -> Tuple[float, float]:
        """Lambda max range: 0.5 to 5.0."""
        return (0.5, 5.0)

    def update_ppo_params(self, value: float) -> None:
        """Update lambda_max in Lagrange params."""
        if "lagrange_params" not in self.ppo_params:
            self.ppo_params["lagrange_params"] = {}
        self.ppo_params["lagrange_params"]["lambda_max"] = float(value)
        if "r_target" not in self.ppo_params["lagrange_params"]:
            self.ppo_params["lagrange_params"]["r_target"] = 0.15
        self.ppo_params["enable_lagrange"] = True


class LagrangeWarmupStepsOptimizer(LagrangeOptimizerBase):
    """Optimizer for Lagrange warmup steps (constraint activation timing)."""

    @property
    def parameter_name(self) -> str:
        return "lagrange_warmup_steps"

    def get_parameter_range(self) -> Tuple[float, float]:
        """Warmup steps range: 500 to 5000."""
        return (500, 5000)

    def update_ppo_params(self, value: float) -> None:
        """Update warmup_steps in Lagrange params."""
        if "lagrange_params" not in self.ppo_params:
            self.ppo_params["lagrange_params"] = {}
        self.ppo_params["lagrange_params"]["warmup_steps"] = int(value)
        if "r_target" not in self.ppo_params["lagrange_params"]:
            self.ppo_params["lagrange_params"]["r_target"] = 0.15
        self.ppo_params["enable_lagrange"] = True


def main() -> None:
    """Main entry point."""
    parser = BinarySearchArgumentParser.create_parser(
        description="Optimize Lagrange constraint parameters using binary search"
    )
    
    # Add parameter selection
    parser.add_argument(
        "--parameter",
        type=str,
        choices=["r_target", "tolerance", "eta", "lambda_max", "warmup_steps"],
        default="r_target",
        help="Which Lagrange parameter to optimize",
    )
    
    # Add value override for single mode
    parser.add_argument(
        "--value",
        type=float,
        help="Specific value to test (overrides parameter-specific defaults)",
    )
    
    args = parser.parse_args()

    # Select optimizer based on parameter
    optimizer_map = {
        "r_target": LagrangeRTargetOptimizer,
        "tolerance": LagrangeToleranceOptimizer,
        "eta": LagrangeEtaOptimizer,
        "lambda_max": LagrangeLambdaMaxOptimizer,
        "warmup_steps": LagrangeWarmupStepsOptimizer,
    }

    optimizer_class = optimizer_map[args.parameter]
    optimizer = optimizer_class()  # type: ignore[abstract]
    optimizer.configure_from_args(args)

    if args.mode == "single":
        # Get parameter-specific argument or use --value
        param_value = args.value
        if param_value is None:
            # Use midpoint as default
            min_val, max_val = optimizer.get_parameter_range()
            param_value = (min_val + max_val) / 2.0
            print(f"Using default value: {param_value}")

        print(f"Testing {args.parameter} = {param_value}")
        score = optimizer.run_single_test(param_value, args.timesteps)
        print(f"\nFinal score: {score:.6f}")
    else:
        print(f"Optimizing {args.parameter} using binary search...")
        best_value, best_score = optimizer.binary_search_optimize(
            max_iterations=args.max_iterations, 
            total_timesteps=args.timesteps
        )
        print(f"\n{'=' * 80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Best {args.parameter}: {best_value}")
        print(f"Best score: {best_score:.6f}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
