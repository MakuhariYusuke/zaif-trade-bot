"""Helpers for RewardCalculator stage bookkeeping."""

from __future__ import annotations


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


__all__ = ["build_reward_components"]
