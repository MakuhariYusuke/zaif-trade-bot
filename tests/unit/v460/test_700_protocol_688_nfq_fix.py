from __future__ import annotations

from typing import cast

from scripts.v460.analysis.analysis_common import Record
from scripts.v460.analysis.protocols.protocol_688 import (
    Protocol688,
    _cancel_payload,
    _nfq_payload,
)


def _record(
    *,
    cycle_id: str,
    filled: bool,
    cancel_reason: str | None = None,
    regime: str = "ranging",
    side: str = "buy",
) -> Record:
    return cast(
        Record,
        {
            "cycle_id": cycle_id,
            "timestamp": 1_710_000_000.0,
            "filled": filled,
            "cancel_reason": cancel_reason,
            "regime": regime,
            "side": side,
            "post_fill_30s_pnl": 1.0 if filled else None,
            "git_sha": "sha",
        },
    )


def test_nfq_payload_filters_only_nfq() -> None:
    payload = _nfq_payload(
        [
            _record(cycle_id="a", filled=False, cancel_reason="no_feasible_quote"),
            _record(cycle_id="b", filled=False, cancel_reason="timeout"),
            _record(cycle_id="c", filled=True),
        ]
    )

    assert payload["nfq_total"] == 1
    assert payload["cancel_total"] == 2
    assert payload["nfq_ratio"] == 0.5


def test_cancel_payload_unchanged() -> None:
    payload = _cancel_payload(
        [
            _record(cycle_id="a", filled=False, cancel_reason="no_feasible_quote"),
            _record(cycle_id="b", filled=False, cancel_reason="timeout"),
        ]
    )

    assert payload["total"] == 2
    assert payload["reasons"]["no_feasible_quote"] == 1
    assert payload["reasons"]["timeout"] == 1


def test_nfq_payload_collects_regime_and_side_distribution() -> None:
    payload = _nfq_payload(
        [
            _record(cycle_id="a", filled=False, cancel_reason="no_feasible_quote", regime="ranging", side="buy"),
            _record(cycle_id="b", filled=False, cancel_reason="no_feasible_quote", regime="trending_down", side="sell"),
            _record(cycle_id="c", filled=False, cancel_reason="timeout", regime="trending_down", side="sell"),
        ]
    )

    assert payload["regime"] == {"ranging": 1, "trending_down": 1}
    assert payload["side"] == {"buy": 1, "sell": 1}


def test_protocol_output_has_nfq_and_cancels() -> None:
    result = Protocol688().execute(
        [
            _record(cycle_id="a", filled=False, cancel_reason="no_feasible_quote"),
            _record(cycle_id="b", filled=False, cancel_reason="timeout"),
            _record(cycle_id="c", filled=True),
        ]
    )

    assert "nfq" in result.json_payload
    assert "cancels" in result.json_payload
    assert result.json_payload["nfq"]["nfq_total"] == 1
    assert result.json_payload["cancels"]["total"] == 2
