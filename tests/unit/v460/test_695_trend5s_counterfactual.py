from __future__ import annotations

import pytest

from scripts.v460.analysis.sections.section_trend_5s_counterfactual import (
    build_trend_5s_counterfactual_section,
)


def _record(
    *,
    cycle_id: str,
    ts: float,
    side: str = "sell",
    filled: bool = True,
    order_price: float = 100.0,
    mid_at_order: float | None = None,
    mid_at_fill: float | None = None,
    pnl: float | None = None,
    cancel_reason: str | None = None,
    trend_action: str | None = None,
    adverse_selected: bool = False,
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "timestamp": ts,
        "side": side,
        "filled": filled,
        "order_price": order_price,
        "mid_at_order": mid_at_order,
        "mid_at_fill": mid_at_fill,
        "post_fill_30s_pnl": pnl,
        "cancel_reason": cancel_reason,
        "trend_5s_guard_action": trend_action,
        "adverse_selected": adverse_selected,
    }


class TestTrend5sCounterfactual:
    def test_counterfactual_pnl_computation(self) -> None:
        payload = build_trend_5s_counterfactual_section(
            [
                _record(
                    cycle_id="veto",
                    ts=100.0,
                    filled=False,
                    cancel_reason="trend_5s_sell_guard_veto",
                    order_price=100.0,
                ),
                _record(
                    cycle_id="future",
                    ts=131.0,
                    mid_at_order=98.0,
                ),
            ]
        )

        assert payload["veto_group"]["counterfactual_pnl_30s_bps"]["mean"] == pytest.approx(200.0)

    def test_veto_group_filtering(self) -> None:
        payload = build_trend_5s_counterfactual_section(
            [
                _record(cycle_id="a", ts=1.0, filled=False, cancel_reason="trend_5s_sell_guard_veto"),
                _record(cycle_id="b", ts=2.0, filled=False, cancel_reason="timeout"),
            ]
        )

        assert payload["veto_group"]["count"] == 1

    def test_control_group_filtering(self) -> None:
        payload = build_trend_5s_counterfactual_section(
            [
                _record(cycle_id="a", ts=1.0, trend_action="boost", pnl=1.2),
                _record(cycle_id="b", ts=2.0, trend_action="pass", pnl=0.1),
            ]
        )

        assert payload["control_group"]["count"] == 1

    def test_empty_veto_group(self) -> None:
        payload = build_trend_5s_counterfactual_section([])
        assert "no trend_5s veto records" in payload["warnings"]

    def test_net_impact_calculation(self) -> None:
        payload = build_trend_5s_counterfactual_section(
            [
                _record(
                    cycle_id="veto",
                    ts=100.0,
                    filled=False,
                    cancel_reason="trend_5s_sell_guard_veto",
                    order_price=100.0,
                ),
                _record(cycle_id="future", ts=131.0, mid_at_order=102.0),
                _record(cycle_id="ctrl", ts=200.0, trend_action="boost", pnl=1.0),
            ]
        )

        assert payload["net_impact_bps"] is not None

