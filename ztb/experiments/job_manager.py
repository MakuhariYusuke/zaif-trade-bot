"""
Job manager for parallel ML training execution.

Manages 100k × 10 job splitting and execution with timeout and aggregation.
"""

import os
import json
import time
import logging
import sqlite3
import numpy as np
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from ztb.monitoring import get_exporter

logger = logging.getLogger(__name__)

class JobStateDB:
    """
    SQLite database for job state persistence.
    """

    def __init__(self, db_path: str = "experiments/jobs/job_state.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
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
            ''')
            conn.commit()

    def save_job_state(self, job_id: str, status: str, start_time: Optional[float] = None,
                       end_time: Optional[float] = None, checkpoint_path: Optional[str] = None,
                       metrics: Optional[Dict[str, Any]] = None):
        """Save or update job state"""
        with sqlite3.connect(self.db_path) as conn:
            metrics_json = json.dumps(metrics) if metrics else None
            conn.execute('''
                INSERT OR REPLACE INTO jobs
                (id, status, start_time, end_time, checkpoint_path, metrics_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
            ''', (job_id, status, start_time, end_time, checkpoint_path, metrics_json))
            conn.commit()

    def get_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job state by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'status': row[1],
                    'start_time': row[2],
                    'end_time': row[3],
                    'checkpoint_path': row[4],
                    'metrics': json.loads(row[5]) if row[5] else None,
                    'created_at': row[6],
                    'updated_at': row[7]
                }
        return None

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM jobs ORDER BY created_at')
            return [{
                'id': row[0],
                'status': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'checkpoint_path': row[4],
                'metrics': json.loads(row[5]) if row[5] else None,
                'created_at': row[6],
                'updated_at': row[7]
            } for row in cursor.fetchall()]

    def get_incomplete_jobs(self) -> List[Dict[str, Any]]:
        """Get jobs that are not completed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE status != 'completed' ORDER BY created_at")
            return [{
                'id': row[0],
                'status': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'checkpoint_path': row[4],
                'metrics': json.loads(row[5]) if row[5] else None,
                'created_at': row[6],
                'updated_at': row[7]
            } for row in cursor.fetchall()]


class JobManager:
    """
    Manages parallel execution of ML training jobs.

    Splits 1M steps into 100k × 10 jobs, executes with timeout,
    and aggregates results.
    """

    def __init__(self, base_dir: str = "experiments/jobs", timeout_hours: int = 4):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_hours = timeout_hours
        self.timeout_seconds = timeout_hours * 3600

        # Job configuration
        self.total_steps = 1000000  # 1M steps
        self.job_size = 100000     # 100k steps per job
        self.num_repeats = 10       # 10 repeats per job size

        # Results storage
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Manifest storage for atomic resume
        self.manifest_dir = self.base_dir / "manifests"
        self.manifest_dir.mkdir(exist_ok=True)

        # Job state database for persistence
        self.state_db = JobStateDB(str(self.base_dir / "job_state.db"))

        # Monitoring
        self.monitor = get_exporter()

    def _get_code_hash(self) -> str:
        """Get current code hash using git rev-parse HEAD"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                # Fallback: hash of src directory
                return self._hash_directory(Path(__file__).parent.parent)
        except Exception:
            # Fallback: hash of src directory
            return self._hash_directory(Path(__file__).parent.parent)

    def _hash_directory(self, path: Path) -> str:
        """Hash directory contents for code versioning"""
        hash_md5 = hashlib.md5()
        for file_path in sorted(path.rglob("*.py")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _create_job_manifest(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create manifest for job atomic execution"""
        return {
            "job_id": job_config["job_id"],
            "step_from": job_config["start_step"],
            "step_to": job_config["end_step"],
            "input_hash": self._get_input_hash(job_config),
            "code_hash": self._get_code_hash(),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

    def _get_input_hash(self, job_config: Dict[str, Any]) -> str:
        """Get hash of job input parameters"""
        input_data = {
            "repeat": job_config["repeat"],
            "start_step": job_config["start_step"],
            "end_step": job_config["end_step"],
            "steps": job_config["steps"]
        }
        return hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()

    def _save_manifest(self, manifest: Dict[str, Any]):
        """Save job manifest to file"""
        manifest_file = self.manifest_dir / f"{manifest['job_id']}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job manifest from file"""
        manifest_file = self.manifest_dir / f"{job_id}.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                return json.load(f)
        return None

    def _can_skip_job(self, job_config: Dict[str, Any]) -> bool:
        """Check if job can be skipped based on manifest and checkpoint"""
        manifest = self._load_manifest(job_config["job_id"])
        if not manifest:
            return False

        # Check if manifest matches current job
        current_manifest = self._create_job_manifest(job_config)
        if (manifest["input_hash"] != current_manifest["input_hash"] or
            manifest["code_hash"] != current_manifest["code_hash"]):
            return False

        # Check if result file exists and is valid
        result_file = job_config["output_file"]
        if not result_file.exists():
            return False

        # Verify result file is not corrupted
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                return result.get("status") == "completed"
        except Exception:
            return False

    def split_jobs(self) -> List[Dict[str, Any]]:
        """
        Split total training into individual jobs.

        Returns:
            List of job configurations
        """
        jobs = []

        for repeat in range(self.num_repeats):
            for start_step in range(0, self.total_steps, self.job_size):
                end_step = min(start_step + self.job_size, self.total_steps)

                job_config = {
                    "job_id": f"job_{repeat:02d}_{start_step//self.job_size:02d}",
                    "repeat": repeat,
                    "start_step": start_step,
                    "end_step": end_step,
                    "steps": end_step - start_step,
                    "output_file": self.results_dir / f"result_{repeat:02d}_{start_step//self.job_size:02d}.json"
                }
                jobs.append(job_config)

        logger.info(f"Split into {len(jobs)} jobs")
        return jobs

    def execute_job(self, job_config: Dict[str, Any], train_function: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a single training job.

        Args:
            job_config: Job configuration dict
            train_function: Training function to call

        Returns:
            Job result dict
        """
        job_id = job_config["job_id"]
        start_time = time.time()

        logger.info(f"Starting job {job_id}")
        self.monitor.record_job_start()

        # Save initial job state to database
        self.state_db.save_job_state(job_id, "running", start_time=start_time)

        # Create and save manifest
        manifest = self._create_job_manifest(job_config)
        self._save_manifest(manifest)

        try:
            # Execute training with timeout
            result = train_function(job_config)

            execution_time = time.time() - start_time

            job_result = {
                "job_id": job_id,
                "status": "completed",
                "execution_time": execution_time,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(f"Job {job_id} timed out after {execution_time:.1f}s")
            job_result = {
                "job_id": job_id,
                "status": "timeout",
                "execution_time": execution_time,
                "error": "Timeout",
                "timestamp": datetime.now().isoformat()
            }
            self.monitor.record_job_completion("timeout", execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Job {job_id} failed: {e}")
            job_result = {
                "job_id": job_id,
                "status": "failed",
                "execution_time": execution_time,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.monitor.record_job_completion("failed", execution_time)
            self.monitor.record_error("job_execution")

        # Save job result
        output_file = job_config["output_file"]
        with open(output_file, 'w') as f:
            json.dump(job_result, f, indent=2)

        # Update manifest
        manifest["status"] = job_result["status"]
        manifest["completed_at"] = datetime.now().isoformat()
        self._save_manifest(manifest)

        # Update job state in database
        end_time = time.time()
        result_data = job_result.get("result")
        metrics = result_data if isinstance(result_data, dict) else None
        self.state_db.save_job_state(
            job_id,
            str(job_result["status"]),
            start_time=start_time,
            end_time=end_time,
            checkpoint_path=str(job_config.get("output_file", "")),
            metrics=metrics
        )

        logger.info(f"Job {job_id} finished with status: {job_result['status']}")

        # Record successful completion if not already recorded
        if job_result['status'] == 'completed':
            self.monitor.record_job_completion("success", execution_time)

        return job_result

    def run_all_jobs(self, train_function: Callable[[Dict[str, Any]], Dict[str, Any]], max_workers: int = 4) -> Dict[str, Any]:
        """
        Run all jobs in parallel.

        Args:
            train_function: Training function for each job
            max_workers: Maximum number of parallel workers

        Returns:
            Aggregated results
        """
        jobs = self.split_jobs()
        
        # Check for incomplete jobs from previous runs
        incomplete_jobs = self.state_db.get_incomplete_jobs()
        if incomplete_jobs:
            logger.info(f"Found {len(incomplete_jobs)} incomplete jobs from previous run")
            # For now, we'll run all jobs again. In production, you might want to resume specific jobs
        
        completed_jobs = []
        failed_jobs = []

        logger.info(f"Starting execution of {len(jobs)} jobs with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._execute_job_with_timeout, job, train_function): job
                for job in jobs
            }

            # Collect results
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    if result["status"] == "completed":
                        completed_jobs.append(result)
                    else:
                        failed_jobs.append(result)
                except TimeoutError:
                    logger.error(f"Job {job['job_id']} future timed out")
                    failed_jobs.append({
                        "job_id": job["job_id"],
                        "status": "timeout",
                        "error": "Future timeout"
                    })
                except Exception as e:
                    logger.error(f"Job {job['job_id']} future failed: {e}")
                    failed_jobs.append({
                        "job_id": job["job_id"],
                        "status": "failed",
                        "error": str(e)
                    })

        # Aggregate results
        summary = self._aggregate_results(completed_jobs, failed_jobs)

        # Save summary
        summary_file = self.base_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Job execution completed. Summary saved to {summary_file}")
        return summary

    def _execute_job_with_timeout(self, job_config: Dict[str, Any], train_function: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Execute job with timeout handling (Windows-compatible)"""
        try:
            # For Windows compatibility, timeout is handled in run_all_jobs
            result = self.execute_job(job_config, train_function)
            return result
        except Exception as e:
            logging.error(f"Job execution failed: {e}")
            return {
                "job_id": job_config.get("job_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _aggregate_results(self, completed_jobs: List[Dict[str, Any]],
                          failed_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from all jobs.

        Args:
            completed_jobs: List of successful job results
            failed_jobs: List of failed job results

        Returns:
            Aggregated metrics
        """
        if not completed_jobs:
            return {
                "total_jobs": len(completed_jobs) + len(failed_jobs),
                "completed_jobs": 0,
                "failed_jobs": len(failed_jobs),
                "success_rate": 0.0,
                "error": "No jobs completed successfully"
            }

        # Extract metrics from completed jobs
        pnl_values = []
        win_rates = []
        max_drawdowns = []
        sharpe_ratios = []
        execution_times = []

        for job in completed_jobs:
            result = job.get("result", {})
            if isinstance(result, dict):
                pnl_values.append(result.get("total_pnl", 0))
                win_rates.append(result.get("win_rate", 0))
                max_drawdowns.append(result.get("max_drawdown", 0))
                sharpe_ratios.append(result.get("sharpe_ratio", 0))
                execution_times.append(job.get("execution_time", 0))

        # Calculate aggregates
        summary = {
            "total_jobs": len(completed_jobs) + len(failed_jobs),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "success_rate": len(completed_jobs) / (len(completed_jobs) + len(failed_jobs)),
            "pnl": {
                "mean": float(np.mean(pnl_values)) if pnl_values else 0,
                "std": float(np.std(pnl_values)) if pnl_values else 0,
                "min": float(np.min(pnl_values)) if pnl_values else 0,
                "max": float(np.max(pnl_values)) if pnl_values else 0
            },
            "win_rate": {
                "mean": float(np.mean(win_rates)) if win_rates else 0,
                "std": float(np.std(win_rates)) if win_rates else 0
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)) if max_drawdowns else 0,
                "max": float(np.max(max_drawdowns)) if max_drawdowns else 0
            },
            "sharpe_ratio": {
                "mean": float(np.mean(sharpe_ratios)) if sharpe_ratios else 0,
                "std": float(np.std(sharpe_ratios)) if sharpe_ratios else 0
            },
            "execution_time": {
                "mean": float(np.mean(execution_times)) if execution_times else 0,
                "total": float(np.sum(execution_times)) if execution_times else 0
            },
            "timestamp": datetime.now().isoformat()
        }

        return summary

    def get_job_status(self) -> Dict[str, Any]:
        """Get current job execution status"""
        jobs = self.split_jobs()
        completed = 0
        running = 0
        pending = 0

        for job in jobs:
            output_file = job["output_file"]
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        result = json.load(f)
                        if result.get("status") == "completed":
                            completed += 1
                        else:
                            # Failed or timeout
                            pass
                except Exception:
                    # Corrupted file
                    pass
            else:
                # Check manifest for running jobs
                manifest = self._load_manifest(job["job_id"])
                if manifest and manifest.get("status") == "running":
                    running += 1
                else:
                    pending += 1

        return {
            "total_jobs": len(jobs),
            "completed": completed,
            "running": running,
            "pending": pending,
            "progress": completed / len(jobs) if jobs else 0
        }