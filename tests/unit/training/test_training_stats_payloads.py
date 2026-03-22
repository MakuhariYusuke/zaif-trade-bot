from __future__ import annotations

from ztb.training.utils.training_stats_payloads import (
    average_reward_component_history,
    build_optimization_training_stats,
    record_average_reward_components,
    record_training_stat,
)


def test_record_training_stat_updates_mutable_mapping() -> None:
    training_stats: dict[str, object] = {}

    record_training_stat(training_stats, "reward_components", {"pnl": 1.0})

    assert training_stats == {"reward_components": {"pnl": 1.0}}


def test_build_optimization_training_stats_shapes_payload() -> None:
    payload = build_optimization_training_stats(
        memory_stats={"rss_mb": 128.0},
        performance_profile={"speed": 1.5},
        parallel_processing_enabled=True,
        cache_size=7,
    )

    assert payload == {
        "memory_stats": {"rss_mb": 128.0},
        "performance_profile": {"speed": 1.5},
        "parallel_processing_enabled": True,
        "cache_size": 7,
        "data_optimization_applied": True,
    }


def test_average_reward_component_history_ignores_non_numeric_values() -> None:
    payload = average_reward_component_history(
        [
            {"pnl": 1.0, "entropy": 0.5, "note": "ignored"},
            {"pnl": 3, "entropy": 1.5},
            {"pnl": "5.0", "entropy": None},
        ]
    )

    assert payload == {
        "pnl": 3.0,
        "entropy": 1.0,
    }


def test_record_average_reward_components_records_payload() -> None:
    training_stats: dict[str, object] = {}

    averaged = record_average_reward_components(
        training_stats,
        [
            {"pnl": 1.0, "entropy": 0.5},
            {"pnl": 3.0, "entropy": 1.5},
        ],
    )

    assert averaged == {"pnl": 2.0, "entropy": 1.0}
    assert training_stats["reward_components"] == averaged
