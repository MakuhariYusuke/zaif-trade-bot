"""695# regime AS deep-dive analysis protocol."""

from __future__ import annotations

from pathlib import Path

from scripts.v460.analysis.sections.section_regime_as_deep_dive import (
    build_regime_as_deep_dive_section,
)
from . import AnalysisProtocol, ProtocolResult, RecordList, register_protocol


@register_protocol
class Protocol695RegimeAS(AnalysisProtocol):
    protocol_name = "695_regime_as"
    description = "regime × spread adverse-selection attribution"

    def execute(
        self,
        records: RecordList,
        *,
        output_dir: Path | None = None,
    ) -> ProtocolResult:
        del output_dir
        payload = build_regime_as_deep_dive_section(records)
        lines = [
            "# 695 regime AS deep dive",
            f"ranging_buckets={len(payload['ranging_spread_attribution'])}",
            f"trend_5s_veto_overlap_count={payload['trend_5s_veto_overlap_count']}",
            f"spread_p50_bps={payload['spread_distribution']['quantiles_bps']['p50']}",
        ]
        return ProtocolResult(
            text_report="\n".join(lines),
            json_payload={
                "protocol": "695_regime_as",
                **payload,
            },
            warnings=[],
        )
