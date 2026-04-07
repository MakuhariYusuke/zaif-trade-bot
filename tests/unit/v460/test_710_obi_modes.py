from __future__ import annotations

import json
from pathlib import Path

from scripts.v460.analysis.obi_mode_comparison import build_obi_mode_report, main
from scripts.v460.lib.obi_mode import compute_ranging_obi_multiplier


def test_linear_mode_preserves_existing_directionality() -> None:
    buy_mult = compute_ranging_obi_multiplier(
        0.8,
        side="buy",
        imbalance=0.4,
        threshold=0.1,
        factor=0.3,
        mode="linear",
    )
    sell_mult = compute_ranging_obi_multiplier(
        0.8,
        side="sell",
        imbalance=0.4,
        threshold=0.1,
        factor=0.3,
        mode="linear",
    )
    assert buy_mult < 0.8
    assert sell_mult > 0.8


def test_absolute_mode_boosts_both_extremes() -> None:
    buy_heavy = compute_ranging_obi_multiplier(
        1.0,
        side="buy",
        imbalance=0.4,
        threshold=0.1,
        factor=0.3,
        mode="absolute",
    )
    sell_heavy = compute_ranging_obi_multiplier(
        1.0,
        side="buy",
        imbalance=-0.4,
        threshold=0.1,
        factor=0.3,
        mode="absolute",
    )
    assert buy_heavy > 1.0
    assert sell_heavy > 1.0


def test_obi_mode_report() -> None:
    report = build_obi_mode_report(
        [
            {"side": "buy", "orderbook_imbalance": 0.4, "post_fill_30s_pnl": -0.5},
            {"side": "buy", "orderbook_imbalance": -0.5, "post_fill_30s_pnl": -0.4},
        ],
        modes=["linear", "absolute"],
        factor=0.3,
        threshold=0.1,
    )
    assert report["analysis"] == "710_obi_mode_comparison"
    assert "absolute" in report["results"]


def test_main_handles_missing_results_dir(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
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
    assert payload["analysis"] == "710_obi_mode_comparison"
