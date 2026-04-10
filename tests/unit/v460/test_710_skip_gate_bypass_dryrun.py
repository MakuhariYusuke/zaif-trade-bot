from __future__ import annotations

import json
from pathlib import Path

from scripts.v460.analysis.skip_gate_bypass_dryrun import (
    _parse_thresholds,
    build_bypass_dryrun_report,
    main,
)
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator


def test_parse_thresholds() -> None:
    assert _parse_thresholds("0.1, 0.2,0.4") == [0.1, 0.2, 0.4]


def test_dryrun_report_side_breakdown() -> None:
    records = [
        {
            "side": "buy",
            "regime": "ranging",
            "skip_gate_as_prob": 0.7,
            "post_fill_30s_pnl": -0.3,
        },
        {
            "side": "sell",
            "regime": "trending_up",
            "skip_gate_as_prob": 0.2,
            "post_fill_30s_pnl": 0.1,
        },
    ]
    report = build_bypass_dryrun_report(
        records,
        thresholds=[0.6],
        runtime_cfg={"adaptive_threshold": True, "max_skip_rate": 0.3},
    )
    payload = report["threshold_report"]["0.600"]
    assert payload["overall"]["block_count"] == 1
    assert payload["by_side"]["buy"]["block_count"] == 1
    assert payload["by_side"]["sell"]["block_count"] == 0


def test_from_yaml_reads_side_aware_bypass() -> None:
    cfg = FillTestConfig.from_yaml(
        {
            "skip_gate": {
                "enabled": True,
                "bypass_mode": True,
                "bypass_mode_buy": True,
                "bypass_mode_sell": False,
            }
        }
    )
    assert cfg.skip_gate_bypass_mode is True
    assert cfg.skip_gate_bypass_mode_buy is True
    assert cfg.skip_gate_bypass_mode_sell is False


def test_side_aware_bypass_resolution() -> None:
    cfg = FillTestConfig(
        skip_gate_enabled=True,
        skip_gate_bypass_mode=False,
        skip_gate_bypass_mode_buy=True,
        skip_gate_bypass_mode_sell=False,
    )
    evaluator = object.__new__(SkipGateEvaluator)
    evaluator._config = cfg  # type: ignore[attr-defined]
    assert evaluator._is_bypass_mode_active("buy") is True  # type: ignore[attr-defined]
    assert evaluator._is_bypass_mode_active("sell") is False  # type: ignore[attr-defined]


def test_bypass_regime_exclude() -> None:
    """724# regime-conditional bypass exclusion."""
    cfg = FillTestConfig(
        skip_gate_enabled=True,
        skip_gate_bypass_mode=True,
        skip_gate_bypass_mode_buy=True,
        skip_gate_bypass_mode_sell=True,
        skip_gate_bypass_regime_exclude=["sell/trending_down"],
    )
    evaluator = object.__new__(SkipGateEvaluator)
    evaluator._config = cfg  # type: ignore[attr-defined]
    # sell/trending_down is excluded → bypass inactive
    assert evaluator._is_bypass_mode_active("sell", regime="trending_down") is False  # type: ignore[attr-defined]
    # sell/ranging is NOT excluded → bypass active
    assert evaluator._is_bypass_mode_active("sell", regime="ranging") is True  # type: ignore[attr-defined]
    # buy/trending_down is NOT excluded → bypass active
    assert evaluator._is_bypass_mode_active("buy", regime="trending_down") is True  # type: ignore[attr-defined]
    # None regime → bypass active (fallback)
    assert evaluator._is_bypass_mode_active("sell") is True  # type: ignore[attr-defined]


def test_main_handles_missing_dir(tmp_path: Path) -> None:
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
    assert payload["analysis"] == "710_skip_gate_bypass_dryrun"
