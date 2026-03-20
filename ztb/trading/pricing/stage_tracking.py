from __future__ import annotations

import json


def make_offset_stage_store(enabled: bool) -> dict[str, float] | None:
    return {} if enabled else None


def record_offset_stage(
    stages: dict[str, float] | None,
    stage_name: str,
    value: float,
) -> None:
    if stages is None:
        return
    stages[stage_name] = value


def serialize_offset_stages(
    stages: dict[str, float] | None,
) -> str | None:
    if stages is None:
        return None
    return json.dumps(stages)
