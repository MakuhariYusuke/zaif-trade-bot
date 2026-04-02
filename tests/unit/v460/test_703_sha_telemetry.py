from __future__ import annotations

from typing import cast

import pytest

from scripts.v460.analysis.analysis_common import Record
from scripts.v460.analysis.protocols.protocol_688 import Protocol688, _sha_payload


def _record(
    *,
    cycle_id: str,
    git_sha: str,
    filled: bool,
    pnl: float | None = None,
    adverse_selected: bool | None = None,
    is_adverse: bool | None = None,
) -> Record:
    payload: dict[str, object] = {
        "cycle_id": cycle_id,
        "timestamp": 1_710_000_000.0,
        "git_sha": git_sha,
        "filled": filled,
        "post_fill_30s_pnl": pnl,
    }
    if adverse_selected is not None:
        payload["adverse_selected"] = adverse_selected
    if is_adverse is not None:
        payload["is_adverse"] = is_adverse
    return cast(Record, payload)


def test_sha_as_rate_calculation_uses_current_adverse_field() -> None:
    payload = _sha_payload(
        [
            _record(cycle_id="a", git_sha="sha_a", filled=True, pnl=-1.0, adverse_selected=True),
            _record(cycle_id="b", git_sha="sha_a", filled=True, pnl=0.5, adverse_selected=False),
            _record(cycle_id="c", git_sha="sha_a", filled=False, pnl=None),
        ]
    )

    sha = cast(dict[str, object], payload["sha_a"])
    assert sha["adverse_selection_count"] == 1
    assert sha["adverse_selection_rate_pct"] == pytest.approx(50.0)
    assert sha["total_pnl_contribution_bps"] == pytest.approx(-0.5)


def test_sha_empty_fills_zeroes_metrics() -> None:
    payload = _sha_payload([_record(cycle_id="a", git_sha="sha_empty", filled=False)])

    sha = cast(dict[str, object], payload["sha_empty"])
    assert sha["filled"] == 0
    assert sha["adverse_selection_count"] == 0
    assert sha["adverse_selection_rate_pct"] == pytest.approx(0.0)
    assert sha["total_pnl_contribution_bps"] == pytest.approx(0.0)


def test_sha_sorting_worst_contribution_first() -> None:
    payload = _sha_payload(
        [
            _record(cycle_id="a", git_sha="sha_good", filled=True, pnl=1.0, adverse_selected=False),
            _record(cycle_id="b", git_sha="sha_bad", filled=True, pnl=-2.0, adverse_selected=True),
        ]
    )

    assert list(payload.keys())[0] == "sha_bad"


def test_sha_payload_supports_legacy_is_adverse_flag() -> None:
    payload = _sha_payload(
        [
            _record(cycle_id="a", git_sha="legacy", filled=True, pnl=-1.0, is_adverse=True),
            _record(cycle_id="b", git_sha="legacy", filled=True, pnl=0.5, is_adverse=False),
        ]
    )

    sha = cast(dict[str, object], payload["legacy"])
    assert sha["adverse_selection_count"] == 1
    assert sha["adverse_selection_rate_pct"] == pytest.approx(50.0)


def test_protocol_output_keeps_sha_section_shape() -> None:
    result = Protocol688().execute(
        [
            _record(cycle_id="a", git_sha="sha_bad", filled=True, pnl=-1.0, adverse_selected=True),
            _record(cycle_id="b", git_sha="sha_good", filled=True, pnl=0.5, adverse_selected=False),
        ]
    )

    sha_payload = cast(dict[str, dict[str, object]], result.json_payload["sha"])
    assert "sha_bad" in sha_payload
    assert "adverse_selection_rate_pct" in sha_payload["sha_bad"]
    assert "total_pnl_contribution_bps" in sha_payload["sha_bad"]
