from __future__ import annotations

from pathlib import Path

import scripts.v460.analysis.stopgap_daily_report as stopgap_daily_report
from scripts.v460.lib.stopgap_health import DailyHealthReport, serialize_health_report


def _make_report() -> DailyHealthReport:
    return DailyHealthReport(
        generated_at="2026-03-24T00:00:00Z",
        window_hours=48,
        total_records=12,
        total_filled=7,
        filters_applied={"run_id": "run-1"},
        daily_metrics=[],
        model_used_breakdown=[],
        stopgap_checks=[],
        alerts=[],
    )


def test_main_uses_shared_filter_args_and_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    report = _make_report()

    def fake_load_fill_records(results_dir: Path) -> list[dict[str, object]]:
        captured["results_dir"] = results_dir
        return [{"filled": True}]

    def fake_apply_filters(
        records: list[dict[str, object]],
        *,
        run_id: str | None = None,
        git_sha: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, str | None]]:
        captured["run_id"] = run_id
        captured["git_sha"] = git_sha
        captured["date_from"] = date_from
        captured["date_to"] = date_to
        return records, {"run_id": run_id, "git_sha": git_sha}

    def fake_generate_health_report(
        records: list[dict[str, object]],
        *,
        window_hours: int,
        daily_limit: int,
        filters_applied: dict[str, str | None],
    ) -> DailyHealthReport:
        captured["window_hours"] = window_hours
        captured["daily_limit"] = daily_limit
        captured["filters_applied"] = filters_applied
        return report

    def fake_write_json_output(data: object, output_path: object) -> None:
        captured["output_data"] = data
        captured["output_path"] = output_path

    monkeypatch.setattr(stopgap_daily_report, "load_fill_records", fake_load_fill_records)
    monkeypatch.setattr(stopgap_daily_report, "apply_filters", fake_apply_filters)
    monkeypatch.setattr(stopgap_daily_report, "generate_health_report", fake_generate_health_report)
    monkeypatch.setattr(stopgap_daily_report, "write_json_output", fake_write_json_output)

    stopgap_daily_report.main([
        "--results-dir", str(tmp_path),
        "--run-id", "run-1",
        "--git-sha", "abc1234",
        "--date-from", "2026-03-01",
        "--date-to", "2026-03-02",
        "--window", "48",
        "--daily-limit", "5",
        "--output", str(tmp_path / "health.json"),
    ])

    assert captured["results_dir"] == tmp_path
    assert captured["run_id"] == "run-1"
    assert captured["git_sha"] == "abc1234"
    assert captured["date_from"] == "2026-03-01"
    assert captured["date_to"] == "2026-03-02"
    assert captured["window_hours"] == 48
    assert captured["daily_limit"] == 5
    assert captured["filters_applied"] == {"run_id": "run-1", "git_sha": "abc1234"}
    assert captured["output_data"] == serialize_health_report(report)
    assert captured["output_path"] == str(tmp_path / "health.json")
