"""analysis loader/filter contract tests."""

from __future__ import annotations

import argparse
from unittest.mock import patch

from scripts.v460.analysis.analysis_common import add_common_filter_args
from scripts.v460.analysis.compare_regime_ab import main as compare_regime_ab_main
from scripts.v460.analysis.reproduce_152_metrics import main as reproduce_152_main


def test_add_common_filter_args_accepts_legacy_aliases() -> None:
    parser = argparse.ArgumentParser()
    add_common_filter_args(parser, include_legacy_aliases=True)

    args = parser.parse_args(
        ["--data-dir", "results/v460/fill_test", "--start", "2026-02-13", "--end", "2026-02-22"]
    )

    assert args.results_dir == "results/v460/fill_test"
    assert args.date_from == "2026-02-13"
    assert args.date_to == "2026-02-22"


def test_reproduce_main_uses_shared_loader_with_legacy_aliases() -> None:
    records = [
        {
            "timestamp": 1739400000,
            "order_price": 10_000_000,
            "order_quantity": 0.001,
            "filled": True,
            "regime": "ranging",
            "post_fill_30s_pnl": 0.5,
            "side": "buy",
            "run_id": "r1",
            "skip_gate_as_prob": 0.1,
        }
    ]

    with patch(
        "scripts.v460.analysis.reproduce_152_metrics.load_records_from_args",
        return_value=records,
    ) as mocked_loader:
        metrics = reproduce_152_main(
            ["--data-dir", "results/v460/fill_test", "--start", "2026-02-13", "--end", "2026-02-22", "--quiet"]
        )

    mocked_loader.assert_called_once()
    args = mocked_loader.call_args.args[0]
    assert args.results_dir == "results/v460/fill_test"
    assert args.date_from == "2026-02-13"
    assert args.date_to == "2026-02-22"
    assert metrics["total_records"] == 1


def test_compare_main_uses_shared_loader_with_legacy_aliases() -> None:
    records = [
        {
            "timestamp": 1739400000.0,
            "order_price": 10_000_000.0,
            "filled": True,
            "regime": "ranging",
            "post_fill_30s_pnl": 0.2,
        }
    ]

    with patch(
        "scripts.v460.analysis.compare_regime_ab.load_records_from_args",
        return_value=records,
    ) as mocked_loader, patch(
        "scripts.v460.analysis.compare_regime_ab._simulate",
        return_value=([], {"valid_records": 1, "total_input": 1}),
    ), patch(
        "scripts.v460.analysis.compare_regime_ab._evaluate_gates",
        return_value=[],
    ), patch(
        "scripts.v460.analysis.compare_regime_ab._print_report",
    ):
        result = compare_regime_ab_main(
            ["--data-dir", "results/v460/fill_test", "--start", "2026-02-13", "--end", "2026-02-22"]
        )

    mocked_loader.assert_called_once()
    args = mocked_loader.call_args.args[0]
    assert args.results_dir == "results/v460/fill_test"
    assert args.date_from == "2026-02-13"
    assert args.date_to == "2026-02-22"
    assert result == []
