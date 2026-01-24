#!/usr/bin/env python3
"""
Test Reward Function Optimization Framework

This script tests the reward function optimization framework to ensure
all components work correctly together.
"""

import sys
from pathlib import Path
import pytest

from ztb.training.reward_function_evaluator import RewardFunctionEvaluator
from ztb.training.reward_function_optimizer import RewardFunctionOptimizer
from ztb.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def test_parameter_spaces():
    """Test parameter space definitions."""
    logger.info("Testing parameter space definitions...")

    optimizer = RewardFunctionOptimizer()

    # Test that all expected stages are defined
    expected_stages = [
        "balanced_transition",
        "trading_focused",
        "profit_optimized",
        "ultra_profit",
    ]

    for stage in expected_stages:
        assert (
            stage in optimizer.parameter_spaces
        ), f"Stage {stage} not found in parameter spaces"
        assert (
            len(optimizer.parameter_spaces[stage]) > 0
        ), f"No parameters defined for stage {stage}"

        # Check parameter structure
        for param_name, param_def in optimizer.parameter_spaces[stage].items():
            required_attrs = ["type", "low", "high"]
            for attr in required_attrs:
                assert hasattr(
                    param_def, attr
                ), f"Parameter {param_name} missing {attr}"

    logger.info("Parameter space definitions test passed")


def test_evaluation_function():
    """Test evaluation function creation."""
    logger.info("Testing evaluation function creation...")

    evaluator = RewardFunctionEvaluator()

    # Test evaluation function creation for different stages
    stages = ["balanced_transition", "trading_focused", "profit_optimized"]

    for stage in stages:
        eval_func = evaluator.create_evaluation_function(stage)
        assert callable(eval_func), f"Evaluation function for {stage} is not callable"

        # Test with sample parameters
        sample_params = {
            "balance_penalty_tolerance": 0.05,
            "balance_penalty": 5.0,
            "hold_penalty_rate": 0.01,
            "trading_bonus_multiplier": 2.0,
            "trading_bonus": 0.01,
            "profit_weight": 1.0,
            "risk_weight": 1.0,
            "consistency_weight": 1.0,
        }

        scores = eval_func(sample_params)
        assert isinstance(
            scores, dict
        ), "Evaluation function should return a dictionary"
        assert len(scores) > 0, "Evaluation function should return non-empty scores"

        # Check that expected metrics are present
        expected_metrics = ["profit", "sharpe", "win_rate", "consistency"]
        for metric in expected_metrics:
            assert metric in scores, f"Metric {metric} not found in evaluation results"

    logger.info("Evaluation function creation test passed")


def test_optimization_workflow():
    """Test the complete optimization workflow."""
    logger.info("Testing optimization workflow...")

    optimizer = RewardFunctionOptimizer()
    evaluator = RewardFunctionEvaluator()

    # Test optimization with small number of trials for speed
    stage = "balanced_transition"
    n_trials = 5  # Small number for testing

    eval_func = evaluator.create_evaluation_function(stage)

    # Run optimization
    result = optimizer.optimize_reward_function(
        stage=stage,
        evaluation_function=eval_func,
        n_trials=n_trials,
        objectives=["profit", "sharpe", "win_rate"],
    )

    # Check result structure
    assert result.best_config.stage == stage
    assert isinstance(result.best_config.parameters, dict)
    assert len(result.best_config.parameters) > 0
    assert isinstance(result.best_scores, dict)
    assert len(result.best_scores) > 0
    assert result.optimization_time > 0
    assert len(result.optimization_history) == n_trials

    logger.info("Optimization workflow test passed")


def test_file_operations():
    """Test saving and loading optimization results."""
    logger.info("Testing file operations...")

    optimizer = RewardFunctionOptimizer()

    # Create a mock result for testing
    from ztb.training.reward_function_optimizer import (
        RewardFunctionConfig,
        RewardOptimizationResult,
    )

    mock_result = RewardOptimizationResult(
        best_config=RewardFunctionConfig(
            stage="test_stage",
            parameters={"test_param": 1.0},
            objectives=["profit"],
        ),
        best_scores={"profit": 0.1, "sharpe": 1.5},
        optimization_history=[
            {
                "trial_number": 0,
                "parameters": {"test_param": 1.0},
                "scores": {"profit": 0.1},
                "timestamp": "2023-01-01T00:00:00",
            }
        ],
        optimization_time=10.0,
    )

    # Test saving
    test_output_file = "test_optimization_result.json"
    optimizer.save_optimization_result(mock_result, test_output_file)

    assert Path(test_output_file).exists(), "Result file was not created"

    # Test loading
    loaded_result = optimizer.load_optimization_result(test_output_file)

    assert loaded_result.best_config.stage == mock_result.best_config.stage
    assert loaded_result.best_scores == mock_result.best_scores

    # Clean up
    Path(test_output_file).unlink()

    logger.info("File operations test passed")


def test_config_loading():
    """Test configuration file loading."""
    logger.info("Testing configuration loading...")

    config_file = "configs/reward_optimization.json"
    if not Path(config_file).exists():
        pytest.skip(f"Configuration file {config_file} not found - skipping test")

    # Test loading config in optimizer
    optimizer = RewardFunctionOptimizer(config_path=config_file)
    assert optimizer.config_path == config_file

    # Test loading config in evaluator
    evaluator = RewardFunctionEvaluator(config_path=config_file)
    assert evaluator.config_path == config_file

    logger.info("Configuration loading test passed")


def run_all_tests():
    """Run all tests."""
    logger.info("Starting reward function optimization framework tests...")

    try:
        test_parameter_spaces()
        test_evaluation_function()
        test_optimization_workflow()
        test_file_operations()
        test_config_loading()

        logger.info("All tests passed successfully!")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    setup_logging(level="INFO")

    success = run_all_tests()

    if success:
        print("\n✅ All reward function optimization framework tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
