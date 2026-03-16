from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ztb.experiments.job_manager import JobConfig, JobManager


def _make_manager(tmp_path: Path) -> JobManager:
    manager = JobManager(base_dir=str(tmp_path / "jobs"), timeout_hours=1)
    manager.total_steps = 1
    manager.job_size = 1
    manager.num_repeats = 1
    return manager


def _read_status(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("status", "unknown"))


def test_run_all_jobs_default_backend_supports_local_callable(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    calls = {"count": 0}

    def train_fn(_job: JobConfig) -> dict[str, object]:
        calls["count"] += 1
        return {
            "total_pnl": 1.2,
            "win_rate": 0.55,
            "max_drawdown": 0.12,
            "sharpe_ratio": 1.8,
        }

    summary = manager.run_all_jobs(train_fn, max_workers=2)

    assert summary["completed_jobs"] == 1
    assert summary["failed_jobs"] == 0
    assert calls["count"] == 1
    assert _read_status(manager.results_dir / "result_00_00.json") == "completed"


def test_timeout_status_not_overwritten_after_late_completion(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    manager.timeout_seconds = 0.05
    finished = threading.Event()

    def slow_train(_job: JobConfig) -> dict[str, object]:
        try:
            time.sleep(0.12)
            return {
                "total_pnl": 0.5,
                "win_rate": 0.4,
                "max_drawdown": 0.2,
                "sharpe_ratio": 0.8,
            }
        finally:
            finished.set()

    started_at = time.time()
    summary = manager.run_all_jobs(slow_train, max_workers=2, parallel_backend="thread")
    elapsed = time.time() - started_at

    assert summary["completed_jobs"] == 0
    assert summary["failed_jobs"] == 1
    assert elapsed < 0.2

    result_file = manager.results_dir / "result_00_00.json"
    assert _read_status(result_file) == "timeout"

    # Worker may complete after timeout; output must not be overwritten.
    assert finished.wait(0.2)
    assert _read_status(result_file) == "timeout"
