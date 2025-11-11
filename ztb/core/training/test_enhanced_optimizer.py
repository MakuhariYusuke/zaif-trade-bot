#!/usr/bin/env python3
"""
Test script for enhanced RewardFunctionOptimizer console output
"""


from ztb.optimization.reward_function_optimizer import RewardFunctionOptimizer


def test_basic_functionality():
    """Test basic functionality of the enhanced optimizer."""
    print("🧪 Testing Enhanced RewardFunctionOptimizer...")

    # Create optimizer with enhanced console output
    optimizer = RewardFunctionOptimizer()
    optimizer.set_console_output(verbose=True, show_progress=True)

    try:
        # Test parameter space access (use existing parameter_spaces)
        param_space = optimizer.parameter_spaces.get("profit_optimized", {})
        print(
            f"✓ Parameter spaces available with {len(optimizer.parameter_spaces)} stages"
        )

        # Test config creation
        config = optimizer.create_backtest_config(
            {"profit_weight": 1.5, "risk_weight": 0.8}
        )
        print("✓ Backtest config created successfully")

        # Test evaluation
        scores = optimizer.run_backtest_evaluation(config)
        print(f'✓ Evaluation completed: profit={scores.get("profit", 0):.4f}')

        print("🎉 All basic tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


def test_console_output_methods():
    """Test the enhanced console output methods."""
    print("\n🖥️  Testing Console Output Methods...")

    optimizer = RewardFunctionOptimizer()
    optimizer.set_console_output(verbose=True, show_progress=True)

    try:
        # Test header printing
        optimizer._print_header("Test Header", "This is a test of the header function")
        print("✓ Header printing works")

        # Test score printing
        test_scores = {
            "profit": 1.234,
            "sharpe": 2.456,
            "win_rate": 0.789,
            "max_drawdown": 0.123,
        }
        optimizer._print_scores(test_scores, "Test Scores")
        print("✓ Score printing works")

        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            optimizer._handle_error(e, "test scenario")
        print("✓ Error handling works")

        print("🎉 All console output tests passed!")

    except Exception as e:
        print(f"✗ Console output test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic_functionality()
    test_console_output_methods()
