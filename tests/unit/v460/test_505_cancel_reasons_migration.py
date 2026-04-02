from __future__ import annotations

from pathlib import Path
from typing import get_args

from scripts.v460.lib import cancel_reasons as shim_cr
from tests.unit.v460._fill_test_source import read_source_text
from ztb.metrics import fill_quality
from ztb.trading.common import cancel_reasons as canonical_cr


class TestCancelReasonsCanonicalMigration:
    def test_shim_and_canonical_audit_sets_match(self) -> None:
        assert shim_cr.AUDIT_CANCEL_REASONS == canonical_cr.AUDIT_CANCEL_REASONS

    def test_shim_and_canonical_constants_match(self) -> None:
        assert shim_cr.CROSS_VENUE_LEAD_LAG_VETO == canonical_cr.CROSS_VENUE_LEAD_LAG_VETO
        assert shim_cr.SKIP_GATE_RULE_VELOCITY_SELL == canonical_cr.SKIP_GATE_RULE_VELOCITY_SELL
        assert shim_cr.ROUTE_TO_KILL_DEADLOCK == canonical_cr.ROUTE_TO_KILL_DEADLOCK

    def test_cancel_reason_literal_contains_recent_constants(self) -> None:
        literal_values = set(get_args(canonical_cr.CancelReason))
        assert canonical_cr.CROSS_VENUE_LEAD_LAG_VETO in literal_values
        assert canonical_cr.FINAL_CLAMP_HARD_SKIP in literal_values
        assert canonical_cr.ROUTE_TO_KILL_DEADLOCK in literal_values

    def test_fill_record_integrity_imports_canonical_cancel_reasons(self) -> None:
        from ztb.metrics import fill_record_integrity

        source = read_source_text(Path(fill_record_integrity.__file__))
        assert "from ztb.trading.common.cancel_reasons import AUDIT_CANCEL_REASONS" in source
