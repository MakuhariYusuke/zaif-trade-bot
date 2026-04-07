from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from scripts.v460.analysis.entry_gate_ev_distribution import (
    build_entry_gate_ev_distribution_report,
    main,
)


class _FakeCalibration:
    def __init__(self, p_win: float) -> None:
        self._p_win = p_win

    def get_stats(self, regime: str, action: float) -> dict[str, object]:
        del regime, action
        return {
            "l1": {},
            "fallback": {
                "p_win_lcb": self._p_win,
                "p_win_mean": self._p_win,
                "avg_win": 1.0,
                "avg_loss": 1.0,
                "n_eff": 100.0,
            },
            "n_min": 30.0,
        }


def test_distribution_report_compares_baseline() -> None:
    records = [
        {"side": "buy", "regime": "ranging", "post_fill_30s_pnl": 0.1},
        {"side": "sell", "regime": "trending_up", "post_fill_30s_pnl": -0.2},
    ]
    with patch(
        "scripts.v460.analysis.entry_gate_ev_distribution.load_calibration_state",
        side_effect=[_FakeCalibration(0.6), _FakeCalibration(0.4)],
    ):
        report = build_entry_gate_ev_distribution_report(
            records,
            calibration_path=Path("new.json"),
            baseline_calibration_path=Path("old.json"),
        )
    assert report["analysis"] == "710_entry_gate_ev_distribution"
    assert report["counts"]["evaluated"] == 2
    assert report["current_distribution"]["median"] is not None


def test_main_handles_missing_results_dir(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    with patch(
        "scripts.v460.analysis.entry_gate_ev_distribution.load_calibration_state",
        return_value=_FakeCalibration(0.5),
    ):
        assert main(
            [
                "--results-dir",
                str(tmp_path / "missing"),
                "--json",
                "--output",
                str(output),
            ]
        ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["analysis"] == "710_entry_gate_ev_distribution"
