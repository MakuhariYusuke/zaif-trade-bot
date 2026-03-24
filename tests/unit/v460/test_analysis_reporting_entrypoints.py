"""analysis reporting entrypoint contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.v460.analysis.compare_regime_ab import GateResult, SimRecord, _save_summary
from scripts.v460.analysis.diagnose_deadlock import main as diagnose_deadlock_main
from scripts.v460.analysis.side_regime_dashboard import main as side_regime_dashboard_main


def test_diagnose_deadlock_main_uses_shared_output(tmp_path: Path) -> None:
    log_path = tmp_path / "fill_test.log"
    log_path.write_text("2026-03-25 00:00:00 SAFE_STOP: 連続 preflight スキップ 5\n", encoding="utf-8")

    with patch(
        "scripts.v460.analysis.diagnose_deadlock.write_output",
    ) as mocked_output:
        diagnose_deadlock_main(["--log", str(log_path)])

    mocked_output.assert_called_once()
    written = str(mocked_output.call_args.args[0])
    assert "サマリー: 膠着=" in written


def test_side_regime_dashboard_main_uses_shared_json_output(tmp_path: Path) -> None:
    with patch(
        "scripts.v460.analysis.side_regime_dashboard.run_dashboard",
        return_value={
            "timestamp": "2026-03-25T00:00:00+00:00",
            "results_dir": str(tmp_path),
            "total_records": 1,
            "total_filled": 1,
            "overall_fill_rate": 1.0,
            "side_summary": {},
            "regime_side_detail": [],
            "trending_daily": [],
        },
    ), patch(
        "scripts.v460.analysis.side_regime_dashboard.write_json_output",
    ) as mocked_output:
        side_regime_dashboard_main(["--results-dir", str(tmp_path), "--json"])

    mocked_output.assert_called_once()
    payload = mocked_output.call_args.args[0]
    assert payload["total_records"] == 1


def test_compare_regime_ab_save_summary_uses_shared_json_output(tmp_path: Path) -> None:
    gates = [
        GateResult(
            gate_id="G1",
            passed=True,
            threshold="<= 3%",
            actual="1.0%",
            detail="ok",
        )
    ]
    sim_results = [
        SimRecord(
            timestamp=1.0,
            order_price=10_000_000.0,
            recorded_regime="ranging",
            old_regime="unknown",
            new_regime="ranging",
            old_confidence=0.1,
            new_confidence=0.9,
            filled=True,
            pnl_30s=1.2,
        )
    ]

    with patch(
        "scripts.v460.analysis.compare_regime_ab.write_json_output",
    ) as mocked_output:
        _save_summary(gates, sim_results, tmp_path)

    mocked_output.assert_called_once()
    payload, output_path = mocked_output.call_args.args
    assert payload["total_records"] == 1
    assert output_path == tmp_path / "regime_ab_summary.json"
