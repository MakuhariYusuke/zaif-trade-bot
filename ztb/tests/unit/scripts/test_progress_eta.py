import json
import tempfile
from pathlib import Path

from ztb.scripts.progress_eta import (
    compute_eta,
    estimate_steps_per_sec_from_logs,
    estimate_steps_per_sec_from_metrics,
    update_summary,
)


def test_estimate_steps_per_sec_from_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        metrics_path = Path(tmp) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"steps": 1000, "elapsed_time": 60}, f)

        rate = estimate_steps_per_sec_from_metrics(metrics_path)
        assert rate == 1000 / 60


def test_estimate_steps_per_sec_from_metrics_missing():
    with tempfile.TemporaryDirectory() as tmp:
        metrics_path = Path(tmp) / "metrics.json"
        rate = estimate_steps_per_sec_from_metrics(metrics_path)
        assert rate is None


def test_estimate_steps_per_sec_from_logs():
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        log_file = logs_dir / "train.log"
        with open(log_file, "w") as f:
            f.write("2023-09-29T12:00:00 Step 100/1000000\n")
            f.write("2023-09-29T12:01:00 Step 200/1000000\n")

        rate = estimate_steps_per_sec_from_logs(logs_dir)
        assert rate == 100 / 60  # 100 steps in 60 seconds


def test_estimate_steps_per_sec_from_logs_no_logs():
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = Path(tmp)
        rate = estimate_steps_per_sec_from_logs(logs_dir)
        assert rate is None


def test_compute_eta():
    eta, completion, pct = compute_eta(1000, 1000000, 10.0)
    assert eta == "99900s"
    assert pct == 0.1
    assert "2025" in completion  # rough check for future date


def test_compute_eta_zero_rate():
    eta, completion, pct = compute_eta(1000, 1000000, 0.0)
    assert eta == "unknown"
    assert pct == 0.0


def test_update_summary():
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({"existing": "data"}, f)

        update_summary(summary_path, 10.0, "100s", "2023-09-29T13:00:00", 0.1)

        with open(summary_path, "r") as f:
            data = json.load(f)

        assert data["progress"]["steps_per_sec"] == 10.0
        assert data["progress"]["eta"] == "100s"
        assert data["progress"]["pct"] == 0.1


def test_update_summary_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary.json"
        # No error should occur
        update_summary(summary_path, 10.0, "100s", "2023-09-29T13:00:00", 0.1)
