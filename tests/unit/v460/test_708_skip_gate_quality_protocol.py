from __future__ import annotations

import json
from pathlib import Path

from scripts.v460.analysis.analyze_708_skip_gate_quality import main as analyze_main
from scripts.v460.analysis.protocols.protocol_708_skip_gate_quality import Protocol708SkipGateQuality


def _record(score: float, pnl: float, *, date_str: str, bypassed: bool = False, forced: bool = False) -> dict[str, object]:
    return {
        "filled": True,
        "side": "buy",
        "regime": "ranging",
        "skip_gate_score": score,
        "skip_gate_reason": "pass" if not bypassed else "skip",
        "skip_gate_bypassed": bypassed,
        "skip_gate_forced_pass": forced,
        "post_fill_30s_pnl": pnl,
        "date_str": date_str,
        "timestamp": f"{date_str}T00:00:00+00:00",
        "spread_bps": 2.0,
        "orderbook_imbalance": 0.1,
        "price_velocity_bps": 0.2,
    }


def test_protocol_708_skip_gate_quality_executes() -> None:
    protocol = Protocol708SkipGateQuality()
    result = protocol.execute(
        [
            _record(0.4, 1.0, date_str="2026-04-01"),
            _record(0.2, -0.5, date_str="2026-04-02"),
            _record(-0.1, -1.0, date_str="2026-04-04", bypassed=True),
            _record(0.8, 2.0, date_str="2026-04-05", forced=True),
        ]
    )

    assert result.json_payload["analysis"] == "708_skip_gate_quality"
    assert result.json_payload["pre"]["count"] == 2
    assert result.json_payload["post"]["count"] == 2
    assert "counterfactual_thresholds" in result.json_payload["post"]


def test_analyze_708_skip_gate_quality_writes_default_output(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    fill_path = results_dir / "fill_records_20260404.jsonl"
    fill_path.write_text(
        "\n".join(
            [
                json.dumps(_record(-0.2, -1.0, date_str="2026-04-04", bypassed=True)),
                json.dumps(_record(0.7, 1.5, date_str="2026-04-05", forced=True)),
                json.dumps(_record(0.3, 0.2, date_str="2026-04-02")),
                json.dumps(_record(0.1, -0.1, date_str="2026-04-03")),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rc = analyze_main(["--results-dir", str(results_dir), "--json"])

    assert rc == 0
    output_path = tmp_path / "analysis_results" / "708_skip_gate_quality.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["analysis"] == "708_skip_gate_quality"
