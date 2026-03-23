from __future__ import annotations

from ztb.training.utils.training_stats_payloads import (
    average_reward_component_history,
    build_optimization_training_stats,
    extract_reward_component_metrics,
    get_reward_components_payload,
    record_average_reward_components,
    record_optimization_training_stats,
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


def test_get_reward_components_payload_returns_shallow_copy() -> None:
    source = {"reward_components": {"pnl": 1.0, "stage": "train"}}

    payload = get_reward_components_payload(source)

    assert payload == {"pnl": 1.0, "stage": "train"}
    assert payload is not source["reward_components"]


def test_get_reward_components_payload_ignores_invalid_payload() -> None:
    assert get_reward_components_payload({"reward_components": 1.0}) is None


def test_extract_reward_component_metrics_prefers_canonical_payload() -> None:
    payload = extract_reward_component_metrics(
        {
            "reward_components": {"balance_penalty": -0.2},
            "balance_penalty": -1.0,
        }
    )

    assert payload == {"balance_penalty": -0.2}


def test_extract_reward_component_metrics_falls_back_to_flat_info() -> None:
    payload = extract_reward_component_metrics(
        {
            "balance_penalty": -0.2,
            "entropy_shaping": 0.1,
            "action_bonus": 0.05,
            "portfolio_value": 10_000.0,
        }
    )

    assert payload == {
        "balance_penalty": -0.2,
        "entropy_shaping": 0.1,
        "action_bonus": 0.05,
    }


def test_record_optimization_training_stats_builds_and_persists_payload() -> None:
    training_stats: dict[str, object] = {}

    payload = record_optimization_training_stats(
        training_stats,
        memory_stats={"rss_mb": 100.0},
        performance_profile={"step_s": 0.5},
        parallel_processing_enabled=True,
        cache_size=12,
    )

    assert payload["memory_stats"] == {"rss_mb": 100.0}
    assert payload["performance_profile"] == {"step_s": 0.5}
    assert payload["parallel_processing_enabled"] is True
    assert payload["cache_size"] == 12
    assert training_stats["optimization"] == payload
