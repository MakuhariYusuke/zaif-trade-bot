#!/usr/bin/env python3
"""
Comprehensive test of enhanced RewardFunctionOptimizer console output
"""

import json

from ztb.training.reward_function_optimizer.reward_function_optimizer import RewardFunctionOptimizer


def test_optimization_with_enhanced_output():
    """Test optimization with enhanced console output."""
    print("🚀 Testing Optimization with Enhanced Console Output...")

    optimizer = RewardFunctionOptimizer()
    optimizer.set_console_output(verbose=True, show_progress=True)

    try:
        # Run a quick optimization test
        result = optimizer.optimize_reward_function(
            stage="profit_optimized",
            evaluation_function=lambda params: optimizer.run_backtest_evaluation(
                optimizer.create_backtest_config(params)
            ),
            n_trials=5,  # Small number for quick test
            objectives=["profit", "sharpe"],
        )

        print("\n✅ Optimization completed successfully!")
        print(f"🏆 Best Score: {result.best_scores.get('profit', 'N/A'):.4f}")
        print(f"📊 Best Parameters: {result.best_config.parameters}")

        return True

    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_based_optimization():
    """Test config-based optimization with enhanced output."""
    print("\n🔧 Testing Config-Based Optimization...")

    optimizer = RewardFunctionOptimizer()
    optimizer.set_console_output(verbose=True, show_progress=True)

    try:
        # Create a test config file
        test_config = {
            "stage": "profit_optimized",
            "parameters": {
                "profit_weight": 1.0,
                "risk_weight": 0.8,
                "profit_bonus_multiplier_buy": 1.2,
                "profit_bonus_multiplier_sell": 1.1,
                "trading_bonus": 0.05,
            },
        }

        config_path = "test_config.json"
        with open(config_path, "w") as f:
            json.dump(test_config, f, indent=2)

        # Run config-based optimization
        result = optimizer.optimize_from_config_file(
            config_file_path=config_path,
            n_trials=3,  # Small number for quick test
        )

        print("\n✅ Config-based optimization completed!")
        print(f"🏆 Best Score: {result.best_scores.get('profit', 'N/A'):.4f}")

        # Clean up
        import os

        os.remove(config_path)

        return True

    except Exception as e:
        print(f"❌ Config-based optimization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_optimization_with_enhanced_output()
    success2 = test_config_based_optimization()

    if success1 and success2:
        print(
            "\n🎉 All comprehensive tests passed! Enhanced console output is working perfectly."
        )
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
