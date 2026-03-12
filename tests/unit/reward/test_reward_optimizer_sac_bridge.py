#!/usr/bin/env python3
"""Regression tests for SAC/reward bridge behavior in RewardFunctionOptimizer."""

from ztb.training.reward_function_optimizer.reward_function_optimizer import (
    RewardFunctionOptimizer,
)


def test_create_backtest_config_applies_sac_params_and_preserves_base_reward_settings():
    optimizer = RewardFunctionOptimizer()

    base_config = optimizer.create_backtest_config(
        {
            "profit_weight": 1.25,
            "risk_weight": 0.8,
            "profit_bonus_multiplier_buy": 1.2,
            "profit_bonus_multiplier_sell": 1.1,
        }
    )

    updated_config = optimizer.create_backtest_config(
        {"learning_rate": 1e-4, "batch_size": 512},
        base_config=base_config,
    )

    reward_settings = updated_config["reward_settings"]
    assert reward_settings["profit_weight"] == 1.25
    assert reward_settings["risk_weight"] == 0.8
    assert reward_settings["profit_bonus_multipliers"][0] == 1.2
    assert reward_settings["profit_bonus_multipliers"][1] == 1.1

    sac_hyperparameters = updated_config["sac_hyperparameters"]
    assert sac_hyperparameters["learning_rate"] == 1e-4
    assert sac_hyperparameters["batch_size"] == 512


def test_run_backtest_evaluation_reflects_sac_hyperparameter_quality():
    optimizer = RewardFunctionOptimizer()

    strong_config = optimizer.create_backtest_config(
        {
            "profit_weight": 1.1,
            "risk_weight": 0.9,
            "consistency_weight": 1.0,
            "profit_bonus_multiplier_buy": 1.3,
            "profit_bonus_multiplier_sell": 1.3,
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 50_000,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": 0.01,
            "reward_scale": 1.0,
        }
    )
    weak_config = optimizer.create_backtest_config(
        {
            "profit_weight": 1.1,
            "risk_weight": 0.9,
            "consistency_weight": 1.0,
            "profit_bonus_multiplier_buy": 1.3,
            "profit_bonus_multiplier_sell": 1.3,
            "learning_rate": 1e-2,
            "batch_size": 32,
            "buffer_size": 1_000,
            "gamma": 0.85,
            "tau": 0.02,
            "ent_coef": 0.1,
            "reward_scale": 1.8,
        }
    )

    strong_scores = optimizer.run_backtest_evaluation(strong_config)
    weak_scores = optimizer.run_backtest_evaluation(weak_config)

    assert strong_scores["profit"] > weak_scores["profit"]
    assert strong_scores["max_drawdown"] < weak_scores["max_drawdown"]
