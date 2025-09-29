#!/usr/bin/env python3
"""
Comprehensive health check runner (ops doctor).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name, args, cwd=None):
    """Run a script and return exit code."""
    cmd = [sys.executable, f"ztb/ztb/ztb/scripts/{script_name}"] + args
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Timeout"
    except Exception as e:
        return 1, "", str(e)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive health check runner")
    parser.add_argument(
        "--correlation-id", required=True, help="Correlation ID for the session"
    )

    args = parser.parse_args()

    correlation_id = args.correlation_id

    # Scripts to run
    checks = [
        ("validate_artifacts.py", ["--correlation-id", correlation_id]),
        ("collect_last_errors.py", ["--correlation-id", correlation_id]),
        ("progress_eta.py", ["--correlation-id", correlation_id]),
        ("disk_health.py", []),
    ]

    results = []
    details = []

    for script, script_args in checks:
        print(f"Running {script}...", file=sys.stderr)
        exit_code, stdout, stderr = run_script(script, script_args)

        if exit_code == 0:
            status = "OK"
        elif exit_code == 1:
            status = "WARN"
        else:
            status = "FAIL"

        results.append(status)
        details.append(f"=== {script} ===\nStatus: {status}\nExit Code: {exit_code}\n")
        if stdout:
            details.append(f"STDOUT:\n{stdout}\n")
        if stderr:
            details.append(f"STDERR:\n{stderr}\n")
        details.append("\n")

    # Count statuses
    ok_count = results.count("OK")
    warn_count = results.count("WARN")
    fail_count = results.count("FAIL")

    summary = f"Doctor Summary: OK={ok_count}, WARN={warn_count}, FAIL={fail_count}"
    print(summary)

    # Save details
    reports_dir = Path("artifacts") / correlation_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "doctor.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Correlation ID: {correlation_id}\n\n")
        f.write(f"{summary}\n\n")
        f.writelines(details)

    print(f"Details saved to {report_path}")

    # Exit with fail count
    sys.exit(fail_count)


if __name__ == "__main__":
    main()
