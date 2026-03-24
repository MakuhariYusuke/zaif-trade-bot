from __future__ import annotations

import json
from pathlib import Path

import scripts.v460.analysis.ab_offset_comparison as ab_offset_comparison
import scripts.v460.analysis.oracle_baseline as oracle_baseline
import scripts.v460.analysis.oracle_test as oracle_test
import scripts.v460.analysis.print_ab_summary as print_ab_summary
import scripts.v460.analysis.reproduce_152_metrics as reproduce_152_metrics
from ztb.metrics.fill_quality import FillRecord


def test_print_ab_summary_uses_shared_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "score": 1.25,
                    "params": {"alpha": 0.1},
                    "avg_distribution": {"HOLD": 0.2, "BUY": 0.4, "SELL": 0.4},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(print_ab_summary, "write_output", lambda content, output_path=None: captured.update({"content": content, "output_path": output_path}))

    print_ab_summary.main(["--file", str(summary_path), "--top", "1"])

    assert captured["output_path"] is None
    assert "Top candidates by score:" in str(captured["content"])
    assert "alpha: 0.1" in str(captured["content"])


def test_reproduce_152_metrics_uses_shared_json_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    records = [
        {
            "timestamp": 1739400000,
            "order_price": 10_000_000,
            "order_quantity": 0.001,
            "filled": True,
            "regime": "ranging",
            "post_fill_30s_pnl": 0.5,
            "side": "buy",
            "run_id": "test_run_1",
            "skip_gate_as_prob": 0.3,
        },
    ]

    monkeypatch.setattr(reproduce_152_metrics, "load_fill_record_objects_glob", lambda *args, **kwargs: records)
    monkeypatch.setattr(reproduce_152_metrics, "write_json_output", lambda data, output_path=None: captured.update({"data": data, "output_path": output_path}))

    metrics = reproduce_152_metrics.main(["--data-dir", str(tmp_path), "--output", str(tmp_path / "out.json"), "--quiet"])

    assert metrics["total_records"] == 1
    assert captured["output_path"] == tmp_path / "out.json"
    assert isinstance(captured["data"], dict)
    assert captured["data"]["metrics"]["total_records"] == 1


def test_run_oracle_baseline_uses_shared_json_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    records = [
        FillRecord(
            cycle_id="o1",
            timestamp=1_700_000_000.0,
            side="buy",
            order_price=10_000_000,
            order_quantity=0.001,
            filled=True,
            post_fill_30s_pnl=0.8,
            post_fill_60s_pnl=0.6,
            post_fill_120s_pnl=0.4,
            regime="ranging",
        ),
        FillRecord(
            cycle_id="o2",
            timestamp=1_700_000_120.0,
            side="sell",
            order_price=10_000_000,
            order_quantity=0.001,
            filled=True,
            post_fill_30s_pnl=-0.2,
            post_fill_60s_pnl=-0.1,
            post_fill_120s_pnl=0.2,
            regime="trending",
        ),
    ]

    monkeypatch.setattr(oracle_baseline, "iter_fill_records_glob", lambda *args, **kwargs: records)
    monkeypatch.setattr(oracle_baseline, "partition_clean_records", lambda iterable: (list(iterable), []))
    monkeypatch.setattr(oracle_baseline, "write_json_output", lambda data, output_path=None: captured.update({"data": data, "output_path": output_path}))

    report = oracle_baseline.run_oracle_baseline(results_dir=str(tmp_path), output_path=str(tmp_path / "oracle.json"))

    assert captured["output_path"] == tmp_path / "oracle.json"
    assert isinstance(captured["data"], dict)
    assert report["all"]["n_total"] == 2


def test_oracle_test_main_uses_shared_json_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    result: oracle_test.OracleRunResult = {
        "status": "completed",
        "total_records": 2,
        "filled_records": 2,
        "oracle": {
            "pnl30": {
                "status": "ok",
                "n": 2,
                "baseline_mean_bps": 0.5,
                "oracle_skip_mean_bps": 0.8,
                "oracle_flip_mean_bps": 1.0,
                "oracle_skip_improvement_bps": 0.3,
                "profitable_rate": 0.5,
            },
        },
        "as_cost": {
            "n_as": 1,
            "n_non_as": 1,
            "as_ratio": 0.5,
            "as_avg_pnl30_bps": -1.0,
            "non_as_avg_pnl30_bps": 1.0,
            "as_cost_bps": 0.5,
            "oracle_net_of_as_bps": 0.5,
        },
        "kill_switch": {
            "pnl30": "PASS",
            "oracle_pnl30_bps": 1.2,
            "pnl120": "PASS",
            "oracle_pnl120_bps": 1.1,
        },
    }

    monkeypatch.setattr(oracle_test, "run_oracle_test", lambda results_dir="results/v460/fill_test": result)
    monkeypatch.setattr(oracle_test, "append_jsonl", lambda *args, **kwargs: None)
    monkeypatch.setattr(oracle_test, "write_json_output", lambda data, output_path=None: captured.update({"data": data, "output_path": output_path}))

    oracle_test.main(["--results-dir", str(tmp_path)])

    assert captured["output_path"] is None
    assert captured["data"] == result


def test_ab_offset_helpers_use_shared_json_output(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    records = [
        {"cycle_id": "c1", "filled": True, "post_fill_30s_pnl": 0.5, "regime": "ranging", "side": "buy", "timestamp": 1_700_000_000.0},
        {"cycle_id": "c2", "filled": True, "post_fill_30s_pnl": -0.2, "regime": "ranging", "side": "sell", "timestamp": 1_700_000_120.0},
        {"cycle_id": "c3", "filled": True, "post_fill_30s_pnl": 0.8, "regime": "ranging", "side": "buy", "timestamp": 1_700_086_400.0},
    ]

    monkeypatch.setattr(ab_offset_comparison, "_load_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(
        ab_offset_comparison,
        "write_json_output",
        lambda data, output_path=None: captured.setdefault("writes", []).append((data, output_path)),
    )

    ab_offset_comparison._save_baseline(tmp_path, tmp_path / "baseline.json")
    assert len(captured["writes"]) == 1
    assert captured["writes"][0][1] == tmp_path / "baseline.json"

    monkeypatch.setattr(
        ab_offset_comparison,
        "compare_buckets",
        lambda before, after, buckets: [{
            "regime": "ranging",
            "side": "buy",
            "before_n": 1,
            "after_n": 1,
            "before_pnl30": 0.5,
            "after_pnl30": 0.8,
            "pnl_diff": 0.3,
            "before_fill_rate": 1.0,
            "after_fill_rate": 1.0,
            "fill_rate_diff": 0.0,
            "before_as_rate": 0.0,
            "after_as_rate": 0.0,
            "t_statistic": None,
            "p_value": None,
            "significant": False,
        }],
    )
    monkeypatch.setattr(ab_offset_comparison, "_print_comparison", lambda rows: None)

    ab_offset_comparison._run_comparison(tmp_path, "2023-11-14", tmp_path / "compare.json")
    assert len(captured["writes"]) == 2
    assert captured["writes"][1][1] == tmp_path / "compare.json"

