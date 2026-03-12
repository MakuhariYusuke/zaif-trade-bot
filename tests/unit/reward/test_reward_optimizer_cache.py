#!/usr/bin/env python3
"""Tests for RewardFunctionOptimizer evaluation-cache behavior."""

from ztb.training.reward_function_optimizer.reward_function_optimizer import (
    RewardFunctionOptimizer,
)


def test_build_evaluation_cache_key_is_stable_hash() -> None:
    optimizer = RewardFunctionOptimizer()

    params = {"profit_weight": 1.0, "risk_weight": 0.8, "batch_size": 256}
    objectives = ["profit", "sharpe"]

    key1 = optimizer._build_evaluation_cache_key(params, objectives)
    key2 = optimizer._build_evaluation_cache_key(
        {"batch_size": 256, "risk_weight": 0.8, "profit_weight": 1.0},
        ["sharpe", "profit"],
    )
    key3 = optimizer._build_evaluation_cache_key(
        {"batch_size": 128, "risk_weight": 0.8, "profit_weight": 1.0},
        objectives,
    )

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 40


def test_evaluation_cache_evicts_oldest_entries() -> None:
    optimizer = RewardFunctionOptimizer()
    optimizer.max_evaluation_cache_size = 2
    optimizer.evaluation_cache.clear()
    optimizer._evaluation_cache_order.clear()

    optimizer._store_evaluation_cache("k1", {"profit": 1.0})
    optimizer._store_evaluation_cache("k2", {"profit": 2.0})
    optimizer._store_evaluation_cache("k3", {"profit": 3.0})

    assert "k1" not in optimizer.evaluation_cache
    assert "k2" in optimizer.evaluation_cache
    assert "k3" in optimizer.evaluation_cache
    assert len(optimizer._evaluation_cache_order) == 2
