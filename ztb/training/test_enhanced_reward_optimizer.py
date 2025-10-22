#!/usr/bin/env python3
"""
Enhanced Reward Function Optimizer Test Script

This script demonstrates the improved RewardFunctionOptimizer with:
- Expanded parameter spaces
- Dynamic weighting
- Real backtesting integration
- Pareto optimization
- Market regime adaptation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.optimization.reward_function_optimizer import RewardFunctionOptimizer


def test_enhanced_optimizer():
    """Test the enhanced reward function optimizer."""

    print("🚀 Testing Enhanced Reward Function Optimizer")
    print("=" * 60)

    # Initialize optimizer
    optimizer = RewardFunctionOptimizer()

    # Test 1: Basic optimization with expanded parameters
    print("\n📊 Test 1: Basic optimization with profit_optimized stage")
    try:
        result = optimizer.optimize_reward_function(
            stage="profit_optimized",
            evaluation_function=lambda params: optimizer.run_backtest_evaluation(
                optimizer.create_backtest_config(params)
            ),
            n_trials=5,  # Short test
            objectives=["profit", "sharpe", "win_rate"],
        )

        print("✅ Basic optimization completed")
        print(f"Best parameters: {result.best_config.parameters}")
        print(f"Best scores: {result.best_scores}")

    except Exception as e:
        print(f"❌ Basic optimization failed: {e}")

    # Test 2: Dynamic weighting
    print("\n🎯 Test 2: Dynamic weighting system")
    try:
        # Simulate market data
        market_data = {"volatility": 0.7, "trend_strength": 0.8, "phase": "bull"}

        optimizer.update_dynamic_weights(market_data)
        weights = optimizer.get_dynamic_objective_weights(
            ["profit", "sharpe", "win_rate"]
        )

        print("✅ Dynamic weights updated")
        print(f"Market data: {market_data}")
        print(f"Dynamic weights: {weights}")

    except Exception as e:
        print(f"❌ Dynamic weighting failed: {e}")

    # Test 3: Market regime selection
    print("\n🌍 Test 3: Market regime-based stage selection")
    try:
        test_scenarios = [
            {
                "volatility": 0.9,
                "trend_strength": 0.1,
                "phase": "neutral",
            },  # High volatility
            {"volatility": 0.2, "trend_strength": 0.9, "phase": "bull"},  # Bull market
            {"volatility": 0.2, "trend_strength": -0.8, "phase": "bear"},  # Bear market
            {"volatility": 0.1, "trend_strength": 0.0, "phase": "neutral"},  # Sideways
        ]

        for scenario in test_scenarios:
            stage = optimizer.auto_select_stage(scenario)
            print(f"Scenario {scenario} -> Stage: {stage}")

        print("✅ Market regime selection working")

    except Exception as e:
        print(f"❌ Market regime selection failed: {e}")

    # Test 4: Adaptive optimization
    print("\n🔄 Test 4: Adaptive optimization")
    try:
        market_data = {
            "volatility": 0.6,
            "trend_strength": 0.5,
            "phase": "profit_focused",
        }

        result = optimizer.optimize_adaptive(
            market_data=market_data,
            n_trials=3,
            objectives=["profit", "sharpe"],  # Very short test
        )

        print("✅ Adaptive optimization completed")
        print(f"Selected stage: {result.best_config.stage}")
        print(f"Best scores: {result.best_scores}")

    except Exception as e:
        print(f"❌ Adaptive optimization failed: {e}")

    # Test 5: Pareto optimization (if Optuna available)
    print("\n⚖️ Test 5: Pareto optimization")
    try:
        pareto_solutions = optimizer.optimize_pareto_front(
            stage="balanced_transition",
            n_trials=5,
            objectives=["profit", "sharpe", "win_rate"],  # Short test
        )

        print(f"✅ Pareto optimization found {len(pareto_solutions)} solutions")
        if pareto_solutions:
            print(f"First solution scores: {pareto_solutions[0].best_scores}")

    except Exception as e:
        print(f"❌ Pareto optimization failed: {e}")

    # Test 6: Configuration-based optimization
    print("\n📁 Test 6: Configuration-based optimization")
    try:
        # Test loading from existing config file with reward parameters
        config_result = optimizer.optimize_from_config_file(
            config_file_path="configs/reward_optimization.json",
            exploration_range=0.3,  # ±30% exploration
            n_trials=5,
            objectives=["profit", "sharpe"],
        )

        print("✅ Config-based optimization completed")
        print(f"Stage: {config_result.best_config.stage}")
        print(f"Best scores: {config_result.best_scores}")

    except Exception as e:
        print(f"❌ Config-based optimization failed: {e}")

    # Test 7: SAC hyperparameter optimization from config
    print("\n⚙️ Test 7: SAC hyperparameter optimization from config")
    try:
        hyperparam_result = optimizer.optimize_hyperparameters_from_config(
            config_file_path="configs/sac_v428_extended_backtest.json",
            exploration_range=0.2,  # ±20% exploration
            n_trials=5,
        )

        print("✅ SAC hyperparameter optimization completed")
        print(
            f"Optimized learning_rate: {hyperparam_result['optimized_parameters'].get('learning_rate', 'N/A')}"
        )
        print(
            f"Optimized batch_size: {hyperparam_result['optimized_parameters'].get('batch_size', 'N/A')}"
        )

    except Exception as e:
        print(f"❌ SAC hyperparameter optimization failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 Enhanced Reward Function Optimizer test completed!")


if __name__ == "__main__":
    test_enhanced_optimizer()
