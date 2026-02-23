from __future__ import annotations

import math

import pytest

from scripts.v460.lib.results_analyzer import compute_event_contribution
from ztb.metrics.fill_quality import FillRecord


def _make_record(
    cycle_id: str,
    *,
    timestamp: float,
    filled: bool,
    pnl_30s: float | None,
    ffd: bool | None = None,
    vg: bool | None = None,
    as_prob: float | None = None,
) -> FillRecord:
    return FillRecord(
        cycle_id=cycle_id,
        timestamp=timestamp,
        side="buy",
        order_price=14_500_000.0,
        order_quantity=0.001,
        fill_price=14_500_000.0 if filled else None,
        filled=filled,
        post_fill_30s_pnl=pnl_30s,
        ffd_boost_active=ffd,
        vg_triggered=vg,
        skip_gate_as_prob=as_prob,
    )


def test_compute_event_contribution_basic() -> None:
    records = [
        _make_record("a", timestamp=1.0, filled=True, pnl_30s=1.0, ffd=True, vg=True, as_prob=0.80),
        _make_record("b", timestamp=2.0, filled=True, pnl_30s=-1.0, ffd=False, vg=False, as_prob=0.20),
        _make_record("c", timestamp=3.0, filled=True, pnl_30s=2.0, ffd=True, vg=False, as_prob=0.90),
        _make_record("d", timestamp=4.0, filled=False, pnl_30s=None, ffd=True, vg=True, as_prob=0.99),
    ]

    result = compute_event_contribution(records)

    assert result["ffd"]["active"]["n"] == 2
    assert result["ffd"]["inactive"]["n"] == 1
    assert result["ffd"]["active"]["pnl_mean"] == pytest.approx(1.5)
    assert result["ffd"]["inactive"]["pnl_mean"] == pytest.approx(-1.0)
    assert result["ffd"]["delta"] == pytest.approx(2.5)

    assert result["vg"]["triggered"]["n"] == 1
    assert result["vg"]["not_triggered"]["n"] == 2
    assert result["vg"]["triggered"]["pnl_mean"] == pytest.approx(1.0)
    assert result["vg"]["not_triggered"]["pnl_mean"] == pytest.approx(0.5)
    assert result["vg"]["delta"] == pytest.approx(0.5)

    assert result["sg"]["high_prob"]["n"] == 2
    assert result["sg"]["low_prob"]["n"] == 1
    assert result["sg"]["high_prob"]["median_threshold"] == pytest.approx(0.8)
    assert result["sg"]["high_prob"]["pnl_mean"] == pytest.approx(1.5)
    assert result["sg"]["low_prob"]["pnl_mean"] == pytest.approx(-1.0)
    assert result["sg"]["delta"] == pytest.approx(2.5)


def test_compute_event_contribution_ignores_non_finite_values() -> None:
    records = [
        _make_record("a", timestamp=1.0, filled=True, pnl_30s=math.nan, ffd=True, vg=True, as_prob=0.9),
        _make_record("b", timestamp=2.0, filled=True, pnl_30s=1.0, ffd=True, vg=False, as_prob=math.inf),
        _make_record("c", timestamp=3.0, filled=True, pnl_30s=-1.0, ffd=False, vg=True, as_prob=0.2),
    ]

    result = compute_event_contribution(records)

    assert result["ffd"]["active"]["n"] == 1
    assert result["ffd"]["inactive"]["n"] == 1
    assert result["ffd"]["delta"] == pytest.approx(2.0)

    # as_prob=inf は除外されるため、SG 集計は 1 件のみになる
    assert result["sg"]["high_prob"]["n"] == 1
    assert result["sg"]["low_prob"]["n"] == 0
    assert result["sg"]["high_prob"]["median_threshold"] == pytest.approx(0.2)
    assert result["sg"]["delta"] == pytest.approx(-1.0)
