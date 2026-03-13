import json
import subprocess
import sys
from pathlib import Path


def test_run_mtf_scheduler_dry_run(tmp_path: Path):
    cfg = {
        "training": {"model_name": "mtf_test"},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "base_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    cmd = [
        sys.executable,
        "tools/training/run_mtf_scheduler.py",
        "--config",
        str(cfg_path),
        "--dry-run",
        "--out",
        str(tmp_path / "out"),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    assert completed.returncode == 0, f"{completed.stderr}"
    assert (
        "Applied candidate" in completed.stdout
        or "No candidate applied" in completed.stdout
    )
