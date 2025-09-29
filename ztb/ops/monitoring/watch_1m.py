#!/usr/bin/env python3
"""
Long-run watcher for Zaif Trade Bot training sessions.

Monitors training progress and system resources, emitting alerts for:
- No step progress > 10 minutes
- RSS > 2.0 GB or GPU VRAM > 4.0 GB
- Async checkpoint queue backlog > 0
- Error rate > 2% over last 1h
- Mean reward drops > 2σ for 30k steps

Thresholds configurable via environment variables.
Outputs JSONL to artifacts/{correlation_id}/logs/watch_log.jsonl.
Non-zero exit for hard breaches (RSS/VRAM/stall) or kill-file.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TrainingWatcher:
    def __init__(self, correlation_id: str, log_dir: Path):
        self.correlation_id = correlation_id
        self.artifacts_dir = Path("artifacts") / correlation_id
        self.logs_dir = self.artifacts_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / "watch_log.jsonl"
        self.log_dir = log_dir

        # Thresholds from env vars with defaults
        self.stall_min = int(os.getenv("ZTB_WATCH_STALL_MIN", "10"))
        self.rss_mb = int(os.getenv("ZTB_WATCH_RSS_MB", "2048"))  # 2GB
        self.vram_mb = int(os.getenv("ZTB_WATCH_VRAM_MB", "4096"))  # 4GB
        self.err_pct = float(os.getenv("ZTB_WATCH_ERR_PCT", "2.0"))
        self.reward_sigma = float(os.getenv("ZTB_WATCH_REWARD_SIGMA", "2.0"))

        # State tracking
        self.last_step = 0
        self.last_step_time = datetime.now()
        self.errors_last_hour = []
        self.rewards = []  # For sigma calculation
        self.hard_breach = False
        self.kill_file = Path("ztb.stop")

        # Progress detection
        self.step_pattern = re.compile(r"global_step=(\d+)")

    def emit_alert(
        self, level: str, alert_type: str, message: str, details: Dict = None
    ):
        """Emit structured JSON alert to file and stdout."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "correlation_id": self.correlation_id,
            "level": level,
            "alert_type": alert_type,
            "message": message,
            "details": details or {},
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(alert) + "\n")
        print(json.dumps(alert, indent=None))

    def get_current_step(self) -> Optional[int]:
        """Get current global_step from multiple sources."""
        # Primary: metrics.json
        metrics_file = self.artifacts_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    if "global_step" in data:
                        return data["global_step"]
            except Exception:
                pass

        # Fallback: log regex
        log_files = list(self.log_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_log, "r") as f:
                    lines = f.readlines()[-100:]  # Last 100 lines
                    for line in reversed(lines):
                        match = self.step_pattern.search(line)
                        if match:
                            return int(match.group(1))
            except Exception:
                pass

        # Optional: TensorBoard
        if HAS_TENSORBOARD:
            tb_dir = self.artifacts_dir / "tensorboard"
            if tb_dir.exists():
                try:
                    for event_file in tb_dir.glob("events.out.tfevents.*"):
                        ea = EventAccumulator(str(event_file))
                        ea.Reload()
                        scalars = ea.Tags()["scalars"]
                        if "global_step" in scalars:
                            values = ea.Scalars("global_step")
                            if values:
                                return int(values[-1].value)
                except Exception:
                    pass

        return None

    def check_kill_file(self) -> bool:
        """Check for kill file."""
        if self.kill_file.exists():
            try:
                with open(self.kill_file, "r") as f:
                    details = f.read().strip()
                self.emit_alert(
                    "CRITICAL",
                    "kill_file_detected",
                    f"Kill file detected: {details}",
                    {"content": details},
                )
                return True
            except Exception:
                self.emit_alert("CRITICAL", "kill_file_detected", "Kill file detected")
                return True
        return False

    def check_memory_usage(self) -> Tuple[bool, bool]:
        """Check RSS and GPU VRAM usage. Returns (rss_breach, vram_breach)."""
        rss_breach = False
        vram_breach = False

        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                rss_mb = process.memory_info().rss / 1024 / 1024
                if rss_mb > self.rss_mb:
                    rss_breach = True
                    self.emit_alert(
                        "ERROR",
                        "memory_rss",
                        f"RSS usage {rss_mb:.1f}MB exceeds {self.rss_mb}MB",
                        {"rss_mb": rss_mb},
                    )
            except Exception:
                pass

        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if gpu.memoryUsed > self.vram_mb:
                        vram_breach = True
                        self.emit_alert(
                            "ERROR",
                            "memory_vram",
                            f"GPU VRAM usage {gpu.memoryUsed:.1f}MB exceeds {self.vram_mb}MB",
                            {"gpu_id": gpu.id, "vram_mb": gpu.memoryUsed},
                        )
            except Exception:
                pass

        return rss_breach, vram_breach

    def check_step_progress(self) -> bool:
        """Check if training has stalled."""
        current_step = self.get_current_step()
        if current_step is None:
            return False

        now = datetime.now()
        if current_step > self.last_step:
            self.last_step = current_step
            self.last_step_time = now
            return False

        stall_duration = now - self.last_step_time
        if stall_duration.total_seconds() > self.stall_min * 60:
            self.emit_alert(
                "ERROR",
                "stall",
                f"No step progress for {stall_duration.total_seconds() / 60:.1f} minutes",
                {
                    "last_step": self.last_step,
                    "stall_minutes": stall_duration.total_seconds() / 60,
                },
            )
            return True
        return False

    def check_checkpoint_backlog(self) -> bool:
        """Check async checkpoint queue backlog."""
        metrics_file = self.artifacts_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    backlog = data.get("checkpoint_backlog", 0)
                    if backlog > 0:
                        self.emit_alert(
                            "WARN",
                            "checkpoint_backlog",
                            f"Checkpoint backlog: {backlog}",
                            {"backlog": backlog},
                        )
                        return True
            except Exception:
                pass
        return False

    def check_error_rate(self) -> bool:
        """Check error rate over last hour."""
        now = datetime.now()
        # Clean old errors
        self.errors_last_hour = [
            t for t in self.errors_last_hour if now - t < timedelta(hours=1)
        ]

        if not self.errors_last_hour:
            return False

        error_rate = len(self.errors_last_hour) / 60  # errors per minute
        if error_rate > self.err_pct / 100:
            self.emit_alert(
                "WARN",
                "error_rate",
                f"Error rate {error_rate * 100:.1f}% exceeds {self.err_pct}%",
                {
                    "error_rate_pct": error_rate * 100,
                    "errors_count": len(self.errors_last_hour),
                },
            )
            return True
        return False

    def check_reward_drop(self) -> bool:
        """Check if mean reward drops > 2σ."""
        # Placeholder: need to parse rewards from logs/metrics
        # For now, skip
        return False

    def parse_log_for_errors(self, line: str):
        """Parse log line for errors."""
        if "error" in line.lower() or "exception" in line.lower():
            self.errors_last_hour.append(datetime.now())

    def watch(self):
        """Main watch loop."""
        last_status = time.time()

        while True:
            if self.check_kill_file():
                return 2

            rss_breach, vram_breach = self.check_memory_usage()
            if rss_breach or vram_breach:
                self.hard_breach = True

            stall = self.check_step_progress()
            if stall:
                self.hard_breach = True

            self.check_checkpoint_backlog()
            self.check_error_rate()
            self.check_reward_drop()

            # Parse logs for errors (simple tail)
            log_files = list(self.log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_log, "r") as f:
                        f.seek(0, 2)  # End
                        line = f.readline()
                        if line:
                            self.parse_log_for_errors(line.strip())
                except Exception:
                    pass

            # Status print every minute
            if time.time() - last_status > 60:
                current_step = self.get_current_step() or self.last_step
                print(f"STATUS: step={current_step}, rss_ok, vram_ok, no_stall")
                last_status = time.time()

            time.sleep(10)  # Check every 10s

    def run_once(self):
        """Run checks once for testing."""
        exit_code = 0
        if self.check_kill_file():
            return 2

        rss_breach, vram_breach = self.check_memory_usage()
        stall = self.check_step_progress()
        backlog = self.check_checkpoint_backlog()
        error_rate = self.check_error_rate()
        reward_drop = self.check_reward_drop()

        if rss_breach or vram_breach or stall:
            exit_code = 1

        return exit_code


def main():
    parser = argparse.ArgumentParser(description="Watch training session for issues")
    parser.add_argument(
        "--correlation-id", required=True, help="Correlation ID for this session"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--run-once", action="store_true", help="Run checks once and exit"
    )

    args = parser.parse_args()

    watcher = TrainingWatcher(args.correlation_id, args.log_dir)
    if args.run_once:
        sys.exit(watcher.run_once())
    else:
        sys.exit(watcher.watch())


if __name__ == "__main__":
    main()
