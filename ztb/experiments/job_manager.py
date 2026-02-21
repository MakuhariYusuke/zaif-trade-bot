"""
Job manager for parallel ML training execution.

Manages 100k × 10 job splitting and execution with timeout and aggregation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypedDict, cast

import numpy as np

from ztb.ops.monitoring.monitoring import get_exporter
from ztb.types.common import ObjectMap, ObjectRecords
from ztb.utils.file_utils import safe_json_load
from ztb.utils.safety import ensure_dict, safe_to_float

logger = logging.getLogger(__name__)


class JobConfig(TypedDict):
    """Configuration payload for a single job."""

    job_id: str
    repeat: int
    start_step: int
    end_step: int
    steps: int
    output_file: Path


class JobResult(TypedDict, total=False):
    """Normalized result payload for a single job."""

    job_id: str
    status: str
    execution_time: float
    result: ObjectMap
    error: str
    timestamp: str


class JobStateRecord(TypedDict, total=False):
    """Persisted job-state record from sqlite."""

    id: str
    status: str
    start_time: Optional[float]
    end_time: Optional[float]
    checkpoint_path: Optional[str]
    metrics: Optional[ObjectMap]
    created_at: float
    updated_at: float


def _json_default(value: object) -> object:
    """JSON serializer for numpy and path-like objects."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _as_object_map(value: object) -> ObjectMap:
    return ensure_dict(value)


def _as_float(value: object, default: float = 0.0) -> float:
    return safe_to_float(value, default)


class JobStateDB:
    """SQLite database for job state persistence."""

    def __init__(self, db_path: str = "experiments/jobs/job_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    checkpoint_path TEXT,
                    metrics_json TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """
            )
            conn.commit()

    def save_job_state(
        self,
        job_id: str,
        status: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[ObjectMap] = None,
    ) -> None:
        """Save or update job state."""
        metrics_json = (
            json.dumps(metrics, default=_json_default) if isinstance(metrics, dict) else None
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, start_time, end_time, checkpoint_path, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    start_time=COALESCE(excluded.start_time, jobs.start_time),
                    end_time=COALESCE(excluded.end_time, jobs.end_time),
                    checkpoint_path=COALESCE(excluded.checkpoint_path, jobs.checkpoint_path),
                    metrics_json=COALESCE(excluded.metrics_json, jobs.metrics_json),
                    updated_at=strftime('%s', 'now')
            """,
                (job_id, status, start_time, end_time, checkpoint_path, metrics_json),
            )
            conn.commit()

    def _row_to_state(self, row: tuple[object, ...]) -> JobStateRecord:
        metrics_payload: Optional[ObjectMap] = None
        if row[5]:
            try:
                metrics_payload = _as_object_map(json.loads(cast(str, row[5])))
            except Exception:
                metrics_payload = None

        return {
            "id": cast(str, row[0]),
            "status": cast(str, row[1]),
            "start_time": cast(Optional[float], row[2]),
            "end_time": cast(Optional[float], row[3]),
            "checkpoint_path": cast(Optional[str], row[4]),
            "metrics": metrics_payload,
            "created_at": _as_float(row[6]),
            "updated_at": _as_float(row[7]),
        }

    def get_job_state(self, job_id: str) -> Optional[JobStateRecord]:
        """Get job state by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_state(cast(tuple[object, ...], row))
        return None

    def get_all_jobs(self) -> list[JobStateRecord]:
        """Get all jobs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at")
            rows = cursor.fetchall()
        return [self._row_to_state(cast(tuple[object, ...], row)) for row in rows]

    def get_incomplete_jobs(self) -> list[JobStateRecord]:
        """Get jobs that are not completed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE status != 'completed' ORDER BY created_at"
            )
            rows = cursor.fetchall()
        return [self._row_to_state(cast(tuple[object, ...], row)) for row in rows]


class JobManager:
    """
    Manages parallel execution of ML training jobs.

    Splits 1M steps into 100k × 10 jobs, executes with timeout, and aggregates
    results.
    """

    def __init__(self, base_dir: str = "experiments/jobs", timeout_hours: int = 4):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_hours = timeout_hours
        self.timeout_seconds = timeout_hours * 3600

        self.total_steps = 1_000_000
        self.job_size = 100_000
        self.num_repeats = 10

        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.manifest_dir = self.base_dir / "manifests"
        self.manifest_dir.mkdir(exist_ok=True)

        self.state_db = JobStateDB(str(self.base_dir / "job_state.db"))
        self.monitor = get_exporter()

    def _record_monitor(self, method_name: str, *args: object) -> None:
        method = getattr(self.monitor, method_name, None)
        if callable(method):
            try:
                method(*args)
            except Exception:
                logger.debug("Monitor call failed: %s", method_name, exc_info=True)

    def _get_code_hash(self) -> str:
        """Get current code hash using git rev-parse HEAD."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return self._hash_directory(Path(__file__).parent.parent)

    def _hash_directory(self, path: Path) -> str:
        """Hash directory contents for code versioning."""
        hash_md5 = hashlib.md5()
        for file_path in sorted(path.rglob("*.py")):
            if not file_path.is_file():
                continue
            with open(file_path, "rb") as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _create_job_manifest(self, job_config: JobConfig) -> ObjectMap:
        """Create manifest for job atomic execution."""
        return {
            "job_id": job_config["job_id"],
            "step_from": job_config["start_step"],
            "step_to": job_config["end_step"],
            "input_hash": self._get_input_hash(job_config),
            "code_hash": self._get_code_hash(),
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }

    def _get_input_hash(self, job_config: JobConfig) -> str:
        """Get hash of job input parameters."""
        input_data: ObjectMap = {
            "repeat": job_config["repeat"],
            "start_step": job_config["start_step"],
            "end_step": job_config["end_step"],
            "steps": job_config["steps"],
        }
        return hashlib.md5(
            json.dumps(input_data, sort_keys=True, default=_json_default).encode()
        ).hexdigest()

    def _save_manifest(self, manifest: ObjectMap) -> None:
        """Save job manifest to file."""
        manifest_file = self.manifest_dir / f"{manifest['job_id']}.json"
        with open(manifest_file, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, default=_json_default)

    def _load_manifest(self, job_id: str) -> Optional[ObjectMap]:
        """Load job manifest from file."""
        manifest_file = self.manifest_dir / f"{job_id}.json"
        if not manifest_file.exists():
            return None
        return _as_object_map(safe_json_load(manifest_file))

    def _can_skip_job(self, job_config: JobConfig) -> bool:
        """Check if job can be skipped based on manifest and output consistency."""
        manifest = self._load_manifest(job_config["job_id"])
        if not manifest:
            return False

        current_manifest = self._create_job_manifest(job_config)
        if (
            manifest.get("input_hash") != current_manifest.get("input_hash")
            or manifest.get("code_hash") != current_manifest.get("code_hash")
        ):
            return False

        result_file = job_config["output_file"]
        if not result_file.exists():
            return False

        try:
            result = _as_object_map(safe_json_load(result_file))
            return str(result.get("status")) == "completed"
        except Exception:
            return False

    def split_jobs(self) -> list[JobConfig]:
        """Split total training into individual jobs."""
        jobs: list[JobConfig] = []
        for repeat in range(self.num_repeats):
            for start_step in range(0, self.total_steps, self.job_size):
                end_step = min(start_step + self.job_size, self.total_steps)
                jobs.append(
                    {
                        "job_id": f"job_{repeat:02d}_{start_step // self.job_size:02d}",
                        "repeat": repeat,
                        "start_step": start_step,
                        "end_step": end_step,
                        "steps": end_step - start_step,
                        "output_file": self.results_dir
                        / f"result_{repeat:02d}_{start_step // self.job_size:02d}.json",
                    }
                )
        logger.info("Split into %d jobs", len(jobs))
        return jobs

    def _write_job_result(self, output_file: Path, job_result: JobResult) -> None:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(job_result, file, indent=2, default=_json_default)

    def execute_job(
        self,
        job_config: JobConfig,
        train_function: Callable[[JobConfig], ObjectMap],
    ) -> JobResult:
        """Execute a single training job."""
        job_id = job_config["job_id"]
        start_time = time.time()

        logger.info("Starting job %s", job_id)
        self._record_monitor("record_job_start")
        self.state_db.save_job_state(job_id, "running", start_time=start_time)

        manifest = self._create_job_manifest(job_config)
        manifest["status"] = "running"
        manifest["started_at"] = datetime.now().isoformat()
        self._save_manifest(manifest)

        try:
            result = _as_object_map(train_function(job_config))
            execution_time = time.time() - start_time
            job_result: JobResult = {
                "job_id": job_id,
                "status": "completed",
                "execution_time": execution_time,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
            self._record_monitor("record_job_completion", "success", execution_time)
        except TimeoutError:
            execution_time = time.time() - start_time
            logger.warning("Job %s timed out after %.1fs", job_id, execution_time)
            job_result = {
                "job_id": job_id,
                "status": "timeout",
                "execution_time": execution_time,
                "error": "Timeout",
                "timestamp": datetime.now().isoformat(),
            }
            self._record_monitor("record_job_completion", "timeout", execution_time)
        except Exception as exc:
            execution_time = time.time() - start_time
            logger.error("Job %s failed: %s", job_id, exc)
            job_result = {
                "job_id": job_id,
                "status": "failed",
                "execution_time": execution_time,
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
            self._record_monitor("record_job_completion", "failed", execution_time)
            self._record_monitor("record_error", "job_execution")

        self._write_job_result(job_config["output_file"], job_result)
        manifest["status"] = job_result["status"]
        manifest["completed_at"] = datetime.now().isoformat()
        if "error" in job_result:
            manifest["error"] = job_result["error"]
        self._save_manifest(manifest)

        end_time = time.time()
        metrics = job_result.get("result") if isinstance(job_result.get("result"), dict) else None
        self.state_db.save_job_state(
            job_id,
            str(job_result["status"]),
            start_time=start_time,
            end_time=end_time,
            checkpoint_path=str(job_config.get("output_file", "")),
            metrics=cast(Optional[ObjectMap], metrics),
        )

        logger.info("Job %s finished with status: %s", job_id, job_result["status"])
        return job_result

    def _register_async_failure(
        self, job_config: JobConfig, status: str, error: str
    ) -> JobResult:
        now_iso = datetime.now().isoformat()
        result: JobResult = {
            "job_id": job_config["job_id"],
            "status": status,
            "error": error,
            "timestamp": now_iso,
        }
        self._write_job_result(job_config["output_file"], result)

        manifest = self._load_manifest(job_config["job_id"]) or self._create_job_manifest(
            job_config
        )
        manifest["status"] = status
        manifest["completed_at"] = now_iso
        manifest["error"] = error
        self._save_manifest(manifest)

        self.state_db.save_job_state(
            job_config["job_id"],
            status,
            end_time=time.time(),
            checkpoint_path=str(job_config["output_file"]),
        )
        self._record_monitor("record_job_completion", status, 0.0)
        self._record_monitor("record_error", "job_execution")
        return result

    def _try_load_completed_result(self, job_config: JobConfig) -> Optional[JobResult]:
        try:
            payload = _as_object_map(safe_json_load(job_config["output_file"]))
            if str(payload.get("status")) == "completed":
                return cast(JobResult, payload)
        except Exception:
            return None
        return None

    def run_all_jobs(
        self,
        train_function: Callable[[JobConfig], ObjectMap],
        max_workers: int = 4,
    ) -> ObjectMap:
        """Run all jobs in parallel and aggregate results."""
        jobs = self.split_jobs()
        incomplete_jobs = self.state_db.get_incomplete_jobs()
        if incomplete_jobs:
            logger.info(
                "Found %d incomplete jobs from previous run", len(incomplete_jobs)
            )

        completed_jobs: list[JobResult] = []
        failed_jobs: list[JobResult] = []
        executable_jobs: list[JobConfig] = []
        processed_job_ids: set[str] = set()

        for job in jobs:
            if self._can_skip_job(job):
                cached = self._try_load_completed_result(job)
                if cached is not None:
                    completed_jobs.append(cached)
                    processed_job_ids.add(job["job_id"])
                    continue
            executable_jobs.append(job)

        logger.info(
            "Starting execution of %d jobs with %d workers (%d skipped)",
            len(executable_jobs),
            max_workers,
            len(completed_jobs),
        )

        def collect_result_for_job(job: JobConfig, result: JobResult) -> None:
            processed_job_ids.add(job["job_id"])
            if result.get("status") == "completed":
                completed_jobs.append(result)
            else:
                failed_jobs.append(result)

        def execute_sequential(jobs_to_run: list[JobConfig]) -> None:
            for job in jobs_to_run:
                result = self._execute_job_with_timeout(job, train_function)
                collect_result_for_job(job, result)

        if max_workers <= 1:
            execute_sequential(executable_jobs)
        else:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_job = {
                        executor.submit(
                            self._execute_job_with_timeout, job, train_function
                        ): job
                        for job in executable_jobs
                    }

                    # Apply per-job timeout from collection point to avoid
                    # blocking indefinitely on hung workers.
                    for future, job in future_to_job.items():
                        try:
                            result = cast(
                                JobResult, future.result(timeout=self.timeout_seconds)
                            )
                            collect_result_for_job(job, result)
                        except TimeoutError:
                            logger.error("Job %s future timed out", job["job_id"])
                            collect_result_for_job(
                                job,
                                self._register_async_failure(
                                    job, "timeout", "Future timeout"
                                ),
                            )
                        except Exception as exc:
                            logger.error("Job %s future failed: %s", job["job_id"], exc)
                            collect_result_for_job(
                                job, self._register_async_failure(job, "failed", str(exc))
                            )
            except Exception as exc:
                logger.error(
                    "Parallel execution setup failed; fallback to sequential mode: %s",
                    exc,
                )
                remaining = [
                    job for job in executable_jobs if job["job_id"] not in processed_job_ids
                ]
                execute_sequential(remaining)

        summary = self._aggregate_results(completed_jobs, failed_jobs)
        summary_file = self.base_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, default=_json_default)

        logger.info("Job execution completed. Summary saved to %s", summary_file)
        return summary

    def _execute_job_with_timeout(
        self,
        job_config: JobConfig,
        train_function: Callable[[JobConfig], ObjectMap],
    ) -> JobResult:
        """Execute job with timeout handling (Windows-compatible wrapper)."""
        try:
            return self.execute_job(job_config, train_function)
        except Exception as exc:
            logging.error("Job execution failed: %s", exc)
            return {
                "job_id": job_config.get("job_id", "unknown"),
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }

    def _aggregate_results(
        self, completed_jobs: list[JobResult], failed_jobs: list[JobResult]
    ) -> ObjectMap:
        """Aggregate results from all jobs."""
        total_jobs = len(completed_jobs) + len(failed_jobs)
        if not completed_jobs:
            return {
                "total_jobs": total_jobs,
                "completed_jobs": 0,
                "failed_jobs": len(failed_jobs),
                "success_rate": 0.0,
                "error": "No jobs completed successfully",
            }

        pnl_values: list[float] = []
        win_rates: list[float] = []
        max_drawdowns: list[float] = []
        sharpe_ratios: list[float] = []
        execution_times: list[float] = []

        for job in completed_jobs:
            result = _as_object_map(job.get("result"))
            pnl_values.append(_as_float(result.get("total_pnl")))
            win_rates.append(_as_float(result.get("win_rate")))
            max_drawdowns.append(_as_float(result.get("max_drawdown")))
            sharpe_ratios.append(_as_float(result.get("sharpe_ratio")))
            execution_times.append(_as_float(job.get("execution_time")))

        return {
            "total_jobs": total_jobs,
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "success_rate": (len(completed_jobs) / total_jobs) if total_jobs else 0.0,
            "pnl": {
                "mean": float(np.mean(pnl_values)) if pnl_values else 0.0,
                "std": float(np.std(pnl_values)) if pnl_values else 0.0,
                "min": float(np.min(pnl_values)) if pnl_values else 0.0,
                "max": float(np.max(pnl_values)) if pnl_values else 0.0,
            },
            "win_rate": {
                "mean": float(np.mean(win_rates)) if win_rates else 0.0,
                "std": float(np.std(win_rates)) if win_rates else 0.0,
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)) if max_drawdowns else 0.0,
                "max": float(np.max(max_drawdowns)) if max_drawdowns else 0.0,
            },
            "sharpe_ratio": {
                "mean": float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0,
                "std": float(np.std(sharpe_ratios)) if sharpe_ratios else 0.0,
            },
            "execution_time": {
                "mean": float(np.mean(execution_times)) if execution_times else 0.0,
                "total": float(np.sum(execution_times)) if execution_times else 0.0,
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_job_status(self) -> ObjectMap:
        """Get current job execution status."""
        jobs = self.split_jobs()
        completed = 0
        running = 0
        pending = 0
        failed = 0

        for job in jobs:
            output_file = job["output_file"]
            if output_file.exists():
                try:
                    result = _as_object_map(safe_json_load(output_file))
                    status = str(result.get("status", "unknown"))
                    if status == "completed":
                        completed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                continue

            manifest = self._load_manifest(job["job_id"])
            if manifest and str(manifest.get("status")) == "running":
                running += 1
            else:
                pending += 1

        total_jobs = len(jobs)
        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "running": running,
            "failed": failed,
            "pending": pending,
            "progress": (completed / total_jobs) if total_jobs else 0.0,
        }
