from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ztb.metrics.fill_guard_pipeline import GuardPipelineResult


def serialize_guard_pipeline_result(
    result: "GuardPipelineResult | None",
) -> dict[str, object] | None:
    """Serialize an optional guard pipeline result for FillRecord payloads."""

    if result is None:
        return None
    return result.to_dict()


def strip_guard_pipeline_result(
    payload: MutableMapping[str, object],
) -> None:
    """Drop derived guard pipeline data before dataclass reconstruction."""

    payload.pop("guard_pipeline_result", None)


__all__ = [
    "serialize_guard_pipeline_result",
    "strip_guard_pipeline_result",
]
