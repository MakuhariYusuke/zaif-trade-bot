from __future__ import annotations

from pathlib import Path

import scripts.v460.analysis.hour_matched_comparison as hour_matched_comparison
import scripts.v460.analysis.tail_loss_analysis as tail_loss_analysis
import scripts.v460.analysis.vg_and_trend as vg_and_trend


def test_hour_matched_main_uses_shared_json_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_hour_matched_comparison(
        variant_a: str,
        variant_b: str,
        *,
        key_field: str = "git_sha",
        results_dir: Path | None = None,
        side_filter: str | None = None,
    ) -> hour_matched_comparison.HourComparisonResult:
        captured["variants"] = (variant_a, variant_b)
        captured["key_field"] = key_field
        captured["side_filter"] = side_filter
        return {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "key_field": key_field,
            "side_filter": side_filter,
            "n_hours_compared": 1,
            "n_a_total": 2,
            "n_b_total": 2,
            "n_unmatched": 0,
            "overall_pnl_a": 1.0,
            "overall_pnl_b": 2.0,
            "overall_pnl_diff": 1.0,
            "overall_t_stat": None,
            "overall_p_value": None,
            "by_hour": [{
                "utc_hour": 10,
                "jst_hour": 19,
                "a_n": 2,
                "b_n": 2,
                "a_fill_rate": 1.0,
                "b_fill_rate": 1.0,
                "fill_rate_diff": 0.0,
                "a_pnl_bps": 1.0,
                "b_pnl_bps": 2.0,
                "pnl_diff_bps": 1.0,
                "a_as_rate": 0.1,
                "b_as_rate": 0.0,
                "as_rate_diff": -0.1,
                "t_stat": None,
                "p_value": None,
            }],
        }

    def fake_print_report(result: object) -> None:
        captured["printed"] = result

    def fake_write_json_output(data: object, output_path: object) -> None:
        captured["json_data"] = data
        captured["output_path"] = output_path

    monkeypatch.setattr(hour_matched_comparison, "run_hour_matched_comparison", fake_run_hour_matched_comparison)
    monkeypatch.setattr(hour_matched_comparison, "_print_report", fake_print_report)
    monkeypatch.setattr(hour_matched_comparison, "write_json_output", fake_write_json_output)
    monkeypatch.setattr(hour_matched_comparison, "_OUTPUT_DIR", tmp_path)

    hour_matched_comparison.main(["--sha", "aaa1111", "bbb2222", "--json"])

    assert captured["variants"] == ("aaa1111", "bbb2222")
    assert captured["key_field"] == "git_sha"
    assert captured["output_path"] == tmp_path / "hour_matched_aaa1111_bbb2222.json"
    assert isinstance(captured["json_data"], dict)


def test_tail_loss_main_uses_shared_json_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    records = [{"filled": True, "post_fill_30s_pnl": -1.0}]
    analysis = {"buy": {"n": 1, "tail_n": 1}}

    monkeypatch.setattr(tail_loss_analysis, "load_and_filter_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(tail_loss_analysis, "extract_filled", lambda recs: recs)
    monkeypatch.setattr(tail_loss_analysis, "analyze_tail_loss", lambda recs, percentile: analysis)
    monkeypatch.setattr(tail_loss_analysis, "print_analysis", lambda analysis, percentile: None)

    def fake_write_json_output(data: object, output_path: object) -> None:
        captured["data"] = data
        captured["output_path"] = output_path

    monkeypatch.setattr(tail_loss_analysis, "write_json_output", fake_write_json_output)

    tail_loss_analysis.main(["--results-dir", str(tmp_path), "--output", str(tmp_path / "tail.json")])

    assert captured["output_path"] == tmp_path / "tail.json"
    assert isinstance(captured["data"], dict)
    assert captured["data"]["analysis"] == analysis


def test_vg_and_trend_main_uses_shared_output_helpers(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    fake_records: list[object] = []

    monkeypatch.setattr(vg_and_trend, "_load_all_records", lambda results_dir: fake_records)
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

    def fake_write_json_output(data: object, output_path: object | None = None) -> None:
        captured["json_data"] = data
        captured["json_path"] = output_path

    def fake_write_output(content: str, output_path: object | None = None) -> None:
        captured["text_content"] = content
        captured["text_path"] = output_path

    monkeypatch.setattr(vg_and_trend, "write_json_output", fake_write_json_output)
    monkeypatch.setattr(vg_and_trend, "write_output", fake_write_output)

    vg_and_trend.main(["--results-dir", str(tmp_path), "--json", "--output", str(tmp_path / "vg.json")])
    assert captured["json_path"] == tmp_path / "vg.json"
    assert isinstance(captured["json_data"], dict)
    assert captured["json_data"]["vg_effectiveness"] == {"kind": "vg"}

    captured.clear()
    vg_and_trend.main(["--results-dir", str(tmp_path), "--output", str(tmp_path / "vg.txt")])
    assert captured["text_path"] == str(tmp_path / "vg.txt")
    assert isinstance(captured["text_content"], str)

