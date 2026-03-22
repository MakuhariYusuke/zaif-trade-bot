from __future__ import annotations

import json
from typing import TypeAlias

OFFSET_STAGES_SCHEMA_VERSION = "549"
OffsetStageValue: TypeAlias = float | str
OffsetStageStore: TypeAlias = dict[str, OffsetStageValue]


def make_offset_stage_store(enabled: bool) -> OffsetStageStore | None:
    if not enabled:
        return None
    return {"schema_version": OFFSET_STAGES_SCHEMA_VERSION}


def record_offset_stage(
    stages: OffsetStageStore | None,
    stage_name: str,
    value: float,
) -> None:
    if stages is None:
        return
    stages[stage_name] = value


def serialize_offset_stages(
    stages: OffsetStageStore | None,
) -> str | None:
    if stages is None:
        return None
    return json.dumps(stages)
