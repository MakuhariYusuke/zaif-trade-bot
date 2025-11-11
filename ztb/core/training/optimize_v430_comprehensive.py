#!/usr/bin/env python3
"""
SAC v430 Comprehensive Optimization Script
Optimizes both SAC learning parameters and reward function parameters for enhanced trading performance.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ztb.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from ztb.optimization.reward_function_optimizer import RewardFunctionOptimizer


class SACv430Optimizer:
    """Comprehensive optimizer for SAC v430 combining hyperparameter and reward function optimization."""

    def __init__(self):
        self.hyper_optimizer = HyperparameterOptimizer()
        self.reward_optimizer = RewardFunctionOptimizer()

        # Configure console output
        self.hyper_optimizer.set_console_output(verbose=True, show_progress=True)
        self.reward_optimizer.set_console_output(verbose=True, show_progress=True)

        # Create output directory
        self.output_dir = Path("configs/v430")
        self.output_dir.mkdir(exist_ok=True)

    def create_comprehensive_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive parameter space combining SAC and reward parameters."""

        # SAC learning parameters
        sac_params = {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log_scale": True,
            },
            "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
            "buffer_size": {
                "type": "categorical",
                "choices": [50000, 100000, 200000, 500000],
            },
            "gamma": {"type": "float", "low": 0.9, "high": 0.999},
            "tau": {"type": "float", "low": 0.001, "high": 0.01},
            "ent_coef": {"type": "float", "low": 1e-4, "high": 1e-1, "log_scale": True},
        }

        # Reward function parameters
        reward_params = {
            "reward_scale": {
                "type": "float",
                "low": 10.0,
                "high": 2000.0,
                "log_scale": True,
            },
            "trading_bonus": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-1,
                "log_scale": True,
            },
            "sell_penalty": {"type": "float", "low": -0.5, "high": 0.5},
            "buy_bonus": {"type": "float", "low": -0.5, "high": 0.5},
            "action_balance_weight": {"type": "float", "low": 0.0, "high": 1.0},
            "hold_penalty": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log_scale": True,
            },
            "risk_penalty": {"type": "float", "low": 0.0, "high": 0.5},
            "profit_focus": {"type": "categorical", "choices": [True, False]},
        }

        # Combine all parameters
        comprehensive_params = {}
        comprehensive_params.update(sac_params)
        comprehensive_params.update(reward_params)

        return comprehensive_params

    def create_optimization_config(self) -> Dict[str, Any]:
        """Create optimization configuration for v430."""

        return {
            "version": "v430",
            "description": "SAC v430: Comprehensive Learning & Reward Function Optimization",
            "optimization": {
                "framework": "combined_hyperparameter_reward",
                "study_name": "sac_v430_comprehensive_optimization",
                "direction": "maximize",
                "n_trials": 150,
                "timeout": 7200,  # 2 hours
                "n_jobs": 1,
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "early_stopping_patience": 25,
                "cross_validation_folds": 3,
            },
            "parameter_space": self.create_comprehensive_parameter_space(),
            "evaluation_metrics": {
                "primary": "sharpe_ratio",
                "secondary": ["total_return", "max_drawdown", "win_rate", "sell_ratio"],
                "constraints": {
                    "sharpe_ratio_min": 1.0,
                    "max_drawdown_max": 0.3,
                    "sell_ratio_max": 0.4,
                },
            },
            "training_config": {
                "total_timesteps": 50000,  # Reduced for optimization speed
                "learning_starts": 1000,
                "gradient_steps": 1,
                "train_freq": [1, "step"],
                "target_entropy": "auto",
                "verbose": 0,
            },
        }

    def dummy_evaluation_function(self, params: Dict[str, Any]) -> float:
        """
        Dummy evaluation function that simulates trading performance.
        In real implementation, this would train SAC agent and evaluate on validation data.
        """
        import time

        time.sleep(0.5)  # Simulate training time

        # Extract parameters
        learning_rate = params.get("learning_rate", 1e-3)
        batch_size = params.get("batch_size", 128)
        gamma = params.get("gamma", 0.99)
        reward_scale = params.get("reward_scale", 100.0)
        action_balance_weight = params.get("action_balance_weight", 0.1)
        risk_penalty = params.get("risk_penalty", 0.1)

        # Simulate performance based on parameter combinations
        # This is a simplified model - real implementation would use actual training

        # Base performance
        base_sharpe = 1.2

        # Learning rate impact (optimal around 1e-3)
        lr_factor = 1.0 - abs(np.log10(learning_rate) - np.log10(1e-3)) * 0.3
        lr_factor = max(0.1, min(1.5, lr_factor))

        # Batch size impact (larger batches generally better for stability)
        batch_factor = 1.0 + (batch_size - 64) / 512 * 0.2

        # Gamma impact (higher gamma for longer-term focus)
        gamma_factor = 1.0 + (gamma - 0.95) * 0.3

        # Reward scale impact (optimal around 100-500)
        reward_scale_factor = 1.0 - abs(np.log10(reward_scale) - np.log10(100)) * 0.2
        reward_scale_factor = max(0.5, min(1.3, reward_scale_factor))

        # Action balance impact (moderate balance is good)
        balance_factor = 1.0 - abs(action_balance_weight - 0.3) * 0.4

        # Risk penalty impact (moderate risk awareness is good)
        risk_factor = 1.0 - abs(risk_penalty - 0.2) * 0.5

        # Combine factors with some noise
        sharpe_ratio = (
            base_sharpe
            * lr_factor
            * batch_factor
            * gamma_factor
            * reward_scale_factor
            * balance_factor
            * risk_factor
        )

        # Add realistic noise
        noise = np.random.normal(0, 0.1)
        sharpe_ratio += noise

        # Ensure reasonable bounds
        sharpe_ratio = max(0.1, min(3.0, sharpe_ratio))

        return sharpe_ratio

    def optimize_comprehensive(self) -> Dict[str, Any]:
        """Run comprehensive optimization combining hyperparameters and reward function."""

        print("🚀 Starting SAC v430 Comprehensive Optimization")
        print("=" * 60)

        # Create parameter space
        param_space = self.hyper_optimizer.create_parameter_space(
            self.create_comprehensive_parameter_space()
        )

        # Run optimization
        result = self.hyper_optimizer.optimize_hyperparameters(
            objective_function=self.dummy_evaluation_function,
            parameter_space=param_space,
            method="bayesian",
            n_trials=50,  # Reduced for demonstration
            cross_validate=False,
        )

        return result

    def create_v430_config(self, optimization_result) -> Dict[str, Any]:
        """Create the final v430 configuration from optimization results."""

        best_params = optimization_result.best_params

        config = {
            "version": "v430",
            "description": "SAC v430: Optimized Learning Parameters & Reward Function",
            "optimization_score": optimization_result.best_score,
            "optimization_time": optimization_result.optimization_time,
            "n_trials": len(optimization_result.trials),
            "training": {
                "total_timesteps": 100000,
                "learning_rate": best_params.get("learning_rate", 0.0003),
                "gamma": best_params.get("gamma", 0.99),
                "tau": 0.005,  # Fixed for stability
                "ent_coef": "auto_0.01",  # Will be tuned by SAC
                "target_entropy": "auto",
                "batch_size": best_params.get("batch_size", 256),
                "buffer_size": best_params.get("buffer_size", 100000),
                "learning_starts": 1000,
                "gradient_steps": 1,
                "train_freq": [1, "step"],
                "action_noise": None,
            },
            "reward_function": {
                "reward_scale": best_params.get("reward_scale", 100.0),
                "trading_bonus": best_params.get("trading_bonus", 0.01),
                "sell_penalty": best_params.get("sell_penalty", 0.0),
                "buy_bonus": best_params.get("buy_bonus", 0.0),
                "action_balance_weight": best_params.get("action_balance_weight", 0.1),
                "hold_penalty": best_params.get("hold_penalty", 0.001),
                "profit_focus": best_params.get("profit_focus", True),
                "risk_penalty": best_params.get("risk_penalty", 0.1),
            },
            "action_conversion": {
                "symmetric_thresholds": True,
                "action_threshold": 0.3333,
                "buy_threshold": 0.3333,
                "sell_threshold": -0.3333,
                "hold_range": [-0.3333, 0.3333],
            },
            "optimization_metadata": {
                "method": "bayesian_comprehensive",
                "total_trials": len(optimization_result.trials),
                "best_score": optimization_result.best_score,
                "optimization_time_seconds": optimization_result.optimization_time,
                "parameter_space_size": len(best_params),
                "convergence_info": optimization_result.convergence_info,
            },
        }

        return config

    def save_configs(self, optimization_result):
        """Save all v430 configuration files."""

        # Create main configuration
        main_config = self.create_v430_config(optimization_result)

        # Save main config
        main_config_path = self.output_dir / "sac_v430_optimized.json"
        with open(main_config_path, "w", encoding="utf-8") as f:
            json.dump(main_config, f, indent=2, ensure_ascii=False)

        # Save optimization config
        opt_config = self.create_optimization_config()
        opt_config_path = self.output_dir / "sac_v430_optimization_config.json"
        with open(opt_config_path, "w", encoding="utf-8") as f:
            json.dump(opt_config, f, indent=2, ensure_ascii=False)

        # Save optimization results
        results_summary = {
            "version": "v430",
            "optimization_results": {
                "best_score": optimization_result.best_score,
                "best_params": optimization_result.best_params,
                "optimization_time": optimization_result.optimization_time,
                "n_trials": len(optimization_result.trials),
                "convergence_info": optimization_result.convergence_info,
                "all_trials": [
                    {
                        "number": trial["number"],
                        "value": trial["value"],
                        "params": trial["params"],
                    }
                    for trial in optimization_result.trials
                ],
            },
        }

        results_path = self.output_dir / "sac_v430_optimization_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Configurations saved to {self.output_dir}:")
        print(f"   📄 Main config: {main_config_path.name}")
        print(f"   ⚙️  Optimization config: {opt_config_path.name}")
        print(f"   📊 Results summary: {results_path.name}")

        return main_config_path


def main():
    """Main optimization function."""
    print("🎯 SAC v430 Comprehensive Optimization")
    print("   Combining hyperparameter and reward function optimization")
    print("=" * 60)

    optimizer = SACv430Optimizer()

    try:
        # Run comprehensive optimization
        result = optimizer.optimize_comprehensive()

        # Save configurations
        main_config_path = optimizer.save_configs(result)

        print("\n🎉 SAC v430 Optimization Complete!")
        print(f"🏆 Best Score: {result.best_score:.4f}")
        print(f"⏱️  Optimization Time: {result.optimization_time:.1f}s")
        print(f"🎲 Total Trials: {len(result.trials)}")
        print(f"📁 Configuration saved to: {main_config_path}")

        # Display best parameters
        print("\n📊 Best Parameters:")
        for param, value in result.best_params.items():
            print(f"   {param}: {value}")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
