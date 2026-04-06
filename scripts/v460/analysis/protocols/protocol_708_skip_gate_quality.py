from __future__ import annotations

from pathlib import Path

from scripts.v460.analysis.skip_gate_quality_analysis import build_skip_gate_quality_report
from . import AnalysisProtocol, ProtocolResult, RecordList, register_protocol


@register_protocol
class Protocol708SkipGateQuality(AnalysisProtocol):
    protocol_name = "708_skip_gate_quality"
    description = "skip_gate score quality, bimodality, and threshold counterfactual analysis"

    def execute(
        self,
        records: RecordList,
        *,
        output_dir: Path | None = None,
    ) -> ProtocolResult:
        del output_dir
        report = build_skip_gate_quality_report(records)
        return ProtocolResult(
            text_report=report.text_report,
            json_payload=report.json_payload,
            warnings=report.warnings,
        )
