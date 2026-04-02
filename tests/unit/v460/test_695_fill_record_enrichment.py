from __future__ import annotations

from scripts.v460.analysis.analyze_694_multi_perspective import analyze_period
from ztb.metrics.fill_quality import FillRecord, build_fill_record


class TestFillRecordEnrichment:
    def test_enriched_record_has_guard_pipeline(self) -> None:
        record = build_fill_record(
            cycle_id="g1",
            timestamp=1.0,
            side="sell",
            order_price=100.0,
            order_quantity=0.01,
            entry_gate_ev=0.4,
            entry_gate_blocked=False,
            spread_bps=1200.0,
            regime_at_order="ranging",
            trend_5s_at_order=0.8,
            trend_5s_guard_action="boost",
            skip_gate_score=-0.2,
            skip_gate_bypassed=True,
        )

        payload = record.to_dict()

        assert payload["schema_version"] == 2
        assert payload["guard_pipeline_result"]["entry_gate_action"] == "allow"
        assert payload["guard_pipeline_result"]["skip_gate_action"] == "bypass"

    def test_backward_compatible_deserialization(self) -> None:
        record = FillRecord.from_dict(
            {
                "cycle_id": "legacy",
                "timestamp": 1.0,
                "side": "buy",
                "order_price": 100.0,
                "order_quantity": 0.01,
            }
        )

        assert record.schema_version == 1

    def test_optional_fields_none_safe(self) -> None:
        payload = FillRecord(
            cycle_id="minimal",
            timestamp=1.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.01,
        ).to_dict()

        assert payload["guard_pipeline_result"] is None

    def test_guard_pipeline_type_safety(self) -> None:
        payload = build_fill_record(
            cycle_id="typed",
            timestamp=1.0,
            side="sell",
            order_price=100.0,
            order_quantity=0.01,
            entry_gate_ev=-0.2,
            entry_gate_blocked=True,
            trend_5s_at_order=0.4,
            skip_gate_score=0.3,
            skip_gate_skipped=True,
        ).to_dict()["guard_pipeline_result"]

        assert isinstance(payload["entry_gate_ev_bps"], float)
        assert isinstance(payload["entry_gate_action"], str)
        assert isinstance(payload["skip_gate_score"], float)

    def test_existing_analysis_scripts_unaffected(self) -> None:
        record = build_fill_record(
            cycle_id="analysis",
            timestamp=1_710_000_000.0,
            side="buy",
            order_price=100.0,
            order_quantity=0.01,
            filled=True,
            post_fill_30s_pnl=1.2,
            spread_at_order=1200.0,
            adverse_selected_raw=False,
            adverse_selected=False,
            regime="ranging",
        )

        payload = analyze_period([record.to_dict()], "sample")

        assert payload["total"] == 1
        assert payload["filled_count"] == 1
