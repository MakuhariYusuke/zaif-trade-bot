"""Helpers for RewardCalculator stage bookkeeping."""

from __future__ import annotations

from collections.abc import Mapping


def build_reward_components(
    stage: str,
    **components: str | float | int | None,
) -> dict[str, str | float]:
    """Build a compact reward-component payload, filtering unset values."""
    payload: dict[str, str | float] = {"stage": stage}
    for key, value in components.items():
        if value is None:
            continue
        if isinstance(value, bool):
            payload[key] = float(value)
        elif isinstance(value, int | float):
            payload[key] = float(value)
        elif isinstance(value, str):
            payload[key] = value
    return payload


def extend_reward_components(
    payload: dict[str, str | float],
    **components: str | float | int | None,
) -> None:
    """Update an existing reward-component payload in place."""
    payload.update(build_reward_components(str(payload.get("stage", "unknown")), **components))
    payload["stage"] = str(payload.get("stage", "unknown"))


def set_reward_telemetry(
    payload: dict[str, object],
    key: str,
    value: object,
) -> None:
    """Attach arbitrary telemetry while preserving the stage contract."""
    payload[key] = value
    payload["stage"] = str(payload.get("stage", "unknown"))


def merge_reward_components(
    payload: dict[str, object],
    components: Mapping[str, object],
) -> None:
    """Merge component details while preserving the stage contract."""
    for key, value in components.items():
        if key == "stage" or value is None:
            continue
        if isinstance(value, bool):
            payload[key] = float(value)
        elif isinstance(value, int | float):
            payload[key] = float(value)
        elif isinstance(value, str):
            payload[key] = value
        else:
            payload[key] = value
    payload["stage"] = str(payload.get("stage", "unknown"))


__all__ = [
    "build_reward_components",
    "extend_reward_components",
    "merge_reward_components",
    "set_reward_telemetry",
]
