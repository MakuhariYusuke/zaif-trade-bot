from __future__ import annotations

import importlib
import json
from pathlib import Path

from scripts.v460.analysis.analyze_704_sell_offset_pipeline import (
    _parse_offset_stages,
    _pearson_correlation,
    main,
)


def test_analysis_script_imports() -> None:
    module = importlib.import_module("scripts.v460.analysis.analyze_704_sell_offset_pipeline")
    assert module is not None


def test_offset_stage_parsing() -> None:
    parsed = _parse_offset_stages('{"base": 0.02, "tox_buffer": 0.03, "noop": null}')
    assert parsed == {"base": 0.02, "tox_buffer": 0.03}


def test_spread_capture_correlation() -> None:
    corr = _pearson_correlation([0.01, 0.02, 0.03, 0.04], [-0.4, -0.1, 0.2, 0.6])
    assert corr is not None
    assert corr > 0.0


def test_main_handles_missing_results_dir(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    assert (
        main(
            [
                "--results-dir",
                str(tmp_path / "missing"),
                "--json",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["analysis"] == "704_sell_offset_pipeline"
    assert payload["counts"]["records"] == 0
