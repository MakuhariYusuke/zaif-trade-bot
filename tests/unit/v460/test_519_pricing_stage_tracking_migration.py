from __future__ import annotations

from ztb.trading.pricing.stage_tracking import (
    OFFSET_STAGES_SCHEMA_VERSION,
    make_offset_stage_store,
    record_offset_stage,
    serialize_offset_stages,
)


class TestPricingStageTrackingMigration:
    def test_disabled_stage_store_is_none(self) -> None:
        store = make_offset_stage_store(False)
        assert store is None
        assert serialize_offset_stages(store) is None

    def test_stage_store_records_and_serializes(self) -> None:
        store = make_offset_stage_store(True)
        assert store == {"schema_version": OFFSET_STAGES_SCHEMA_VERSION}
        record_offset_stage(store, "base", 0.1)
        record_offset_stage(store, "final", 0.2)
        assert store == {
            "schema_version": OFFSET_STAGES_SCHEMA_VERSION,
            "base": 0.1,
            "final": 0.2,
        }
        assert (
            serialize_offset_stages(store)
            == '{"schema_version": "549", "base": 0.1, "final": 0.2}'
        )
