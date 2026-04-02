"""695# trend_5s counterfactual analysis protocol."""

from __future__ import annotations

from pathlib import Path

from scripts.v460.analysis.sections.section_trend_5s_counterfactual import (
    build_trend_5s_counterfactual_section,
)
from . import AnalysisProtocol, ProtocolResult, RecordList, register_protocol


@register_protocol
class Protocol695Trend5s(AnalysisProtocol):
    protocol_name = "695_trend5s"
    description = "trend_5s veto counterfactual analysis"

    def execute(
        self,
        records: RecordList,
        *,
        output_dir: Path | None = None,
    ) -> ProtocolResult:
        del output_dir
        payload = build_trend_5s_counterfactual_section(records)
        lines = [
            "# 695 trend_5s counterfactual",
            f"veto_count={payload['veto_group']['count']}",
            f"control_count={payload['control_group']['count']}",
            f"value_of_veto_bps={payload['value_of_veto_bps']}",
            f"net_impact_bps={payload['net_impact_bps']}",
        ]
        warnings = [
            str(warning) for warning in payload.get("warnings", []) if isinstance(warning, str)
        ]
        return ProtocolResult(
            text_report="\n".join(lines),
            json_payload={
                "protocol": "695_trend5s",
                **payload,
            },
            warnings=warnings,
        )

