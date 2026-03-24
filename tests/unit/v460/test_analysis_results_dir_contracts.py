from __future__ import annotations

import argparse
from pathlib import Path

import scripts.v460.analysis.ab_offset_comparison as ab_offset_comparison
import scripts.v460.analysis.oracle_baseline as oracle_baseline
import scripts.v460.analysis.oracle_test as oracle_test
import scripts.v460.analysis.vg_and_trend as vg_and_trend
from scripts.v460.analysis.analysis_common import add_results_dir_arg


def test_add_results_dir_arg_uses_shared_default() -> None:
    parser = argparse.ArgumentParser()
    add_results_dir_arg(parser)
    args = parser.parse_args([])
    assert args.results_dir == "results/v460/fill_test"


def test_oracle_baseline_main_accepts_shared_results_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_oracle_baseline(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(oracle_baseline, "run_oracle_baseline", fake_run_oracle_baseline)
    oracle_baseline.main(["--results-dir", str(tmp_path)])
    assert captured["results_dir"] == str(tmp_path)


def test_oracle_test_main_accepts_shared_results_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(oracle_test, "run_oracle_test", lambda results_dir="results/v460/fill_test": {
        "status": "error",
        "reason": "dummy",
        "oracle": {},
    } | {"results_dir": results_dir})

    oracle_test.main(["--results-dir", str(tmp_path)])


def test_vg_and_trend_main_accepts_shared_results_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    fake_records: list[object] = []

    def fake_load_all_records(results_dir: Path) -> list[object]:
        captured["results_dir"] = results_dir
        return fake_records

    monkeypatch.setattr(vg_and_trend, "_load_all_records", fake_load_all_records)
    monkeypatch.setattr(vg_and_trend, "filter_clean_records", lambda records, require_git_sha=True: (fake_records, []))
    monkeypatch.setattr(vg_and_trend, "_load_vg_activations_jsonl", lambda path: [])
    monkeypatch.setattr(vg_and_trend, "_parse_vg_activations", lambda path: [])
    monkeypatch.setattr(vg_and_trend, "_match_vg_to_records", lambda activations, records: set())
    monkeypatch.setattr(vg_and_trend, "analyze_vg_effectiveness", lambda records, cycle_ids: {"kind": "vg"})
    monkeypatch.setattr(vg_and_trend, "analyze_daily_trend", lambda records: [{"kind": "daily"}])
    monkeypatch.setattr(vg_and_trend, "analyze_8h_trend", lambda records: [{"kind": "8h"}])
    monkeypatch.setattr(vg_and_trend, "print_vg_report", lambda result: None)
    monkeypatch.setattr(vg_and_trend, "print_daily_report", lambda result: None)
    monkeypatch.setattr(vg_and_trend, "print_8h_report", lambda result: None)
    monkeypatch.setattr(vg_and_trend, "write_output", lambda content, output_path=None: None)

    vg_and_trend.main(["--results-dir", str(tmp_path), "--output", str(tmp_path / "vg.txt")])
    assert captured["results_dir"] == Path(tmp_path)


def test_ab_offset_comparison_main_accepts_shared_results_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        ab_offset_comparison,
        "_show_baseline",
        lambda results_dir, git_sha=None, run_id=None: captured.update(
            {"results_dir": results_dir, "git_sha": git_sha, "run_id": run_id}
        ),
    )

    ab_offset_comparison.main(["--results-dir", str(tmp_path), "--show-baseline"])
    assert captured["results_dir"] == Path(tmp_path)
