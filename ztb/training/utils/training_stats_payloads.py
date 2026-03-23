from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence


def record_training_stat(
    training_stats: MutableMapping[str, object],
    key: str,
    value: object,
) -> None:
    """Persist training stats through a single shared path."""
    training_stats[key] = value


def build_optimization_training_stats(
    *,
    memory_stats: object,
    performance_profile: object,
    parallel_processing_enabled: bool,
    cache_size: int,
    data_optimization_applied: bool = True,
) -> dict[str, object]:
    """Build optimization stats payload for training summaries."""
    return {
        "memory_stats": memory_stats,
        "performance_profile": performance_profile,
        "parallel_processing_enabled": parallel_processing_enabled,
        "cache_size": cache_size,
        "data_optimization_applied": data_optimization_applied,
    }


def record_optimization_training_stats(
    training_stats: MutableMapping[str, object],
    *,
    memory_stats: object,
    performance_profile: object,
    parallel_processing_enabled: bool,
    cache_size: int,
    data_optimization_applied: bool = True,
) -> dict[str, object]:
    """Build and persist optimization stats through the canonical training-stats path."""
    payload = build_optimization_training_stats(
        memory_stats=memory_stats,
        performance_profile=performance_profile,
        parallel_processing_enabled=parallel_processing_enabled,
        cache_size=cache_size,
        data_optimization_applied=data_optimization_applied,
    )
    record_training_stat(training_stats, "optimization", payload)
    return payload


def average_reward_component_history(
    history: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Average numeric reward components without retaining per-key lists."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for component_map in history:
        for key, value in component_map.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            totals[key] = totals.get(key, 0.0) + numeric_value
            counts[key] = counts.get(key, 0) + 1

    return {
        key: totals[key] / counts[key]
        for key in totals
        if counts.get(key, 0) > 0
    }


def record_average_reward_components(
    training_stats: MutableMapping[str, object],
    history: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Average reward component history and persist it through the canonical stats path."""
    averaged = average_reward_component_history(history)
    record_training_stat(training_stats, "reward_components", averaged)
    return averaged


def get_reward_components_payload(
    container: Mapping[str, object],
) -> dict[str, object] | None:
    """Return a shallow copy of reward_components when present and well-typed."""
    payload = container.get("reward_components")
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def extract_reward_component_metrics(
    container: Mapping[str, object],
) -> dict[str, object] | None:
    """Extract reward-component metrics from canonical payload or legacy flat info."""
    payload = get_reward_components_payload(container)
    if payload is not None:
        return payload

    fallback: dict[str, object] = {}
    for key, value in container.items():
        if (
            key.endswith("_penalty")
            or key.endswith("_shaping")
            or key == "action_bonus"
        ):
            fallback[key] = value
    return fallback or None


__all__ = [
    "average_reward_component_history",
    "build_optimization_training_stats",
    "extract_reward_component_metrics",
    "get_reward_components_payload",
    "record_optimization_training_stats",
    "record_average_reward_components",
    "record_training_stat",
]
