from __future__ import annotations

from types import SimpleNamespace

from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from ztb.ml.skip_gate_runtime import get_trade_field, normalize_recent_trades


class TestSkipGateRuntimeMigration:
    def test_trade_field_shim_points_to_canonical(self) -> None:
        trade = {"ts": 1.0, "amount": 0.25}
        assert SkipGateEvaluator._get_trade_field is not None
        assert SkipGateEvaluator._get_trade_field(
            trade,
            key="timestamp",
            fallback_key="ts",
            default=0.0,
        ) == get_trade_field(
            trade,
            key="timestamp",
            fallback_key="ts",
            default=0.0,
        )

    def test_normalize_recent_trades_shim_matches_canonical(self) -> None:
        trades = [
            {"ts": 1.0, "price": 100.0, "amount": 0.1, "side": "buy"},
            SimpleNamespace(timestamp=2.0, price=101.0, quantity=0.2, side="sell"),
        ]
        legacy = SkipGateEvaluator._normalize_recent_trades(
            trades,
            fallback_timestamp=999.0,
        )
        canonical = normalize_recent_trades(
            trades,
            fallback_timestamp=999.0,
        )
        assert legacy == canonical
