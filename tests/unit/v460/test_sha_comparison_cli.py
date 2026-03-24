from __future__ import annotations

from pathlib import Path

import scripts.v460.analysis.sha_comparison as sha_comparison
from scripts.v460.analysis.sha_comparison import AnalysisResult, SideMetrics


def _make_record(
    *,
    ts: float,
    side: str,
    filled: bool,
    pnl: float | None,
    regime: str,
    sha: str = "abc1234",
) -> dict[str, object]:
    record: dict[str, object] = {
        "timestamp": ts,
        "git_sha": sha,
        "side": side,
        "filled": filled,
        "regime": regime,
        "order_price": 10_000_000.0,
    }
    if pnl is not None:
        record["post_fill_30s_pnl"] = pnl
    return record


def test_run_analysis_uses_typed_hourly_and_daily_buckets(monkeypatch) -> None:
    records = [
        _make_record(ts=1_710_000_000.0, side="buy", filled=True, pnl=0.8, regime="ranging"),
        _make_record(ts=1_710_000_600.0, side="sell", filled=True, pnl=-0.4, regime="trending"),
        _make_record(ts=1_710_008_000.0, side="buy", filled=False, pnl=None, regime="ranging"),
    ]

    monkeypatch.setattr(sha_comparison, "load_fill_record_objects_glob", lambda _: records)

    result = sha_comparison.run_analysis(["abc1234"])

    assert result.n_records == 3
    assert result.n_filled == 2
    assert result.overall is not None
    assert result.overall.avg_pnl_bps == 0.2
    assert len(result.daily) == 1
    assert result.daily[0].n_total == 3
    assert len(result.by_regime) == 2
    hourly_sell = next(side for side in result.by_side if side.side == "sell")
    assert hourly_sell.avg_pnl_bps == -0.4


def test_main_writes_json_via_shared_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    result = AnalysisResult(
        shas=["abc1234"],
        n_records=1,
        n_filled=1,
        n_skipped=0,
        fill_rate=1.0,
        overall=SideMetrics(
            side="all",
            n_total=1,
            n_filled=1,
            fill_rate=1.0,
            avg_pnl_bps=0.5,
            sum_pnl_bps=0.5,
            win_rate=1.0,
            p10_bps=0.5,
            p25_bps=0.5,
            p50_bps=0.5,
            p75_bps=0.5,
            p90_bps=0.5,
            as_rate=0.0,
            severe_as_rate=0.0,
        ),
    )

    def fake_run_analysis(shas: list[str]) -> AnalysisResult:
        captured["shas"] = shas
        return result

    def fake_write_json_output(data: object, output_path: object) -> None:
        captured["data"] = data
        captured["output_path"] = output_path

    monkeypatch.setattr(sha_comparison, "run_analysis", fake_run_analysis)
    monkeypatch.setattr(sha_comparison, "write_json_output", fake_write_json_output)
    monkeypatch.setattr(sha_comparison, "OUTPUT_DIR", tmp_path)

    sha_comparison.main(["--json"])

    assert captured["shas"] == sha_comparison.DEFAULT_SHAS
    assert isinstance(captured["data"], dict)
    assert captured["data"]["shas"] == ["abc1234"]
    assert captured["output_path"] == tmp_path / "333_sha_isolated_dcc3064_4e67014.json"
