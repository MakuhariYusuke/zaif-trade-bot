#!/usr/bin/env python3
"""
Test script for SAC v446 parameter optimization framework
"""

import os
import sys

sys.path.insert(0, os.getcwd())


# Temporarily patch the training function to use mock
def mock_training_with_realistic_scores(config, max_steps=5000):
    import random
    import time

    # Simulate training time (shorter for testing)
    time.sleep(0.05)

    # Generate realistic scores based on parameters
    sac_params = config.get("training", {}).get("sac_hyperparameters", {})
    learning_rate = sac_params.get("learning_rate", 0.0003)
    batch_size = sac_params.get("batch_size", 256)
    gamma = sac_params.get("gamma", 0.99)
    ent_coef = sac_params.get("ent_coef_init", 1.0)

    # Calculate realistic score
    base_score = 75.0

    # Learning rate score (optimal ~0.0003)
    lr_score = max(0, 25 - abs(learning_rate - 0.0003) * 100000)

    # Batch size score (optimal ~256)
    batch_score = max(0, 20 - abs(batch_size - 256) / 15)

    # Gamma score (higher better, but not too high)
    gamma_score = min(gamma * 15, 15)

    # Entropy score (optimal ~1.0)
    ent_score = max(0, 15 - abs(ent_coef - 1.0) * 3)

    total_score = base_score + lr_score + batch_score + gamma_score + ent_score
    total_score += random.uniform(-8, 8)  # Add some noise

    return {
        "success": True,
        "total_timesteps": max_steps,
        "algorithm": "sac",
        "critic_loss": max(0.1, 12.0 - total_score / 8 + random.uniform(-1, 1)),
        "actor_loss": max(0.1, 10.0 - total_score / 10 + random.uniform(-1, 1)),
        "ent_coef": max(0.001, ent_coef * 0.08 + random.uniform(-0.01, 0.01)),
    }


# Monkey patch the function
import sac_v446_learning_params_optimization

sac_v446_learning_params_optimization.run_sac_training_with_config = (
    mock_training_with_realistic_scores
)

print("Running small-scale SAC parameter optimization with realistic mock training...")
print("This will test 5 trials to verify the optimization framework works...")

# Run with very small number of trials for testing
try:
    import logging

    from ztb.training.unified_optimizer import OptimizationConfig, UnifiedOptimizer
    from ztb.utils.logging_utils import setup_logging

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting SAC v446 Learning Parameter Optimization using Unified Optimizer"
    )

    # Create optimization configuration with fewer trials
    opt_config = OptimizationConfig(
        enable_hyperparameter_optimization=True,
        optimization_method="bayesian",
        max_trials=5,  # Just 5 trials for testing
        timeout_hours=1.0,
        max_parallel_trials=1,
    )

    # Create unified optimizer
    optimizer = UnifiedOptimizer(opt_config)

    # Define search space
    search_space = {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 64, "high": 512, "step": 64},
        "gamma": {"type": "float", "low": 0.95, "high": 0.999},
        "ent_coef_init": {"type": "float", "low": 0.5, "high": 2.0},
    }

    # Run optimization
    logger.info("Running 5 optimization trials...")
    results = optimizer.optimize_hyperparameters(
        objective_function=sac_v446_learning_params_optimization.sac_parameter_optimization_objective,
        search_space=search_space,
        method="bayesian",
    )

    # Log results
    best_params = results.best_params
    best_score = results.best_score

    logger.info("Optimization completed!")
    logger.info(f"Best score: {best_score:.2f}")
    logger.info(f"Best parameters: {best_params}")

    print("\nOptimization Results:")
    print(f"Best Score: {best_score:.2f}")
    print(f"Best Parameters: {best_params}")
    print(f"Total Trials: {len(results.optimization_history)}")
    print("\n✅ Optimization framework test completed successfully!")

except Exception as e:
    print(f"❌ Error during optimization: {e}")
    import traceback

    traceback.print_exc()
