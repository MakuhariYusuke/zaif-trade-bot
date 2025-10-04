import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from ztb.ops.rollup.rollup_artifacts import (
    aggregate_metrics,
    generate_executive_summary,
    redact_summary,
    validate_summary,
)


@pytest.fixture
def temp_artifacts() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_aggregate_metrics(temp_artifacts: Path) -> None:
    corr_id = "test123"
    artifacts_dir = temp_artifacts / "artifacts" / corr_id
    artifacts_dir.mkdir(parents=True)

    # Create metrics.json
    metrics = {
        "global_step": 5000,
        "wall_clock_duration": 7200,  # 2 hours
        "rss_peaks": [1024, 2048],
        "vram_peaks": [2048, 4096],
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Create eval file
    eval_data = {
        "sharpe": 1.5,
        "dsr": 0.8,
        "bootstrap_p_value": 0.05,
        "acceptance": {"pass": True, "reason": "good performance"},
    }
    with open(artifacts_dir / "eval_results.json", "w") as f:
        json.dump(eval_data, f)

    # Change to temp directory so aggregate_metrics finds the files
    import os

    old_cwd = os.getcwd()
    os.chdir(temp_artifacts)
    try:
        summary = aggregate_metrics(corr_id)

        assert summary["summary"]["global_step"] == 5000
        assert summary["summary"]["wall_clock_duration_hours"] == 2.0
    finally:
        os.chdir(old_cwd)
    assert summary["summary"]["rss_peak_mb"] == 2048
    assert summary["summary"]["best_sharpe"] == 1.5
    assert summary["summary"]["acceptance_gate"]["pass"] is True


def test_generate_executive_summary() -> None:
    summary = {
        "correlation_id": "test123",
        "timestamp": "2025-09-29T12:00:00",
        "summary": {
            "global_step": 5000,
            "wall_clock_duration_hours": 2.0,
            "rss_peak_mb": 2048,
            "vram_peak_mb": 4096,
            "best_sharpe": 1.5,
            "best_dsr": 0.8,
            "best_bootstrap_p_value": 0.05,
            "acceptance_gate": {"pass": True},
            "artifact_paths": {"metrics_json": "path/to/metrics.json"},
        },
    }

    md = generate_executive_summary(summary)
    assert "# Training Summary for test123" in md
    assert "Global Step: 5000" in md
    assert "Best Sharpe Ratio: 1.5" in md


def test_redact_summary() -> None:
    summary = {
        "metadata": {"config": {"api_key": "secret123", "normal_field": "value"}}
    }

    redacted = redact_summary(summary)
    assert "api_key" not in redacted["metadata"]["config"]
    assert redacted["metadata"]["config"]["normal_field"] == "value"


def test_validate_summary_valid() -> None:
    summary = {
        "correlation_id": "test",
        "timestamp": "2025-09-29T12:00:00",
        "metadata": {
            "version": "1.0.0",
            "timestamp": "2025-09-29T12:00:00",
            "run_id": "123",
            "type": "training",
        },
        "performance": {},
        "metrics": {},
        "evaluations": {},
    }

    # Mock schema validation
    with patch(
        "ztb.ops.rollup.rollup_artifacts.jsonschema.validate", return_value=None
    ):
        valid = validate_summary(summary)
        assert valid


def test_validate_summary_invalid() -> None:
    summary = {}

    with patch(
        "ztb.ops.rollup.rollup_artifacts.jsonschema.validate",
        side_effect=Exception("invalid"),
    ):
        valid = validate_summary(summary)
        assert not valid
