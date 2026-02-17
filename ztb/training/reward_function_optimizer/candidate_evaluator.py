"""
CandidateEvaluator: wrapper that runs `tools/ab_test_runner.py` for a candidate config and returns aggregated metrics.

This is a minimal wrapper for MVP: it supports dry_run and basic metric parsing.

Return contract:
 - evaluate_candidate(...) -> CandidateEvaluationResult
 - Common keys returned in the dict:
        - mean_sharpe: float - arithmetic mean of reported sharpe_ratio values across reported runs (0.0 if none)
        - mean_total_return: float - arithmetic mean of reported total_return values across reported runs (0.0 if none)
        - report_count: int - the number of successful reports (training_report_*.json) that matched the
            candidate's `training.model_name`. This is useful to detect partial failures or timeouts.
        - run_artifacts: list[str] - matched report file paths for diagnostics.

Notes:
 - If `report_count` is 0 the returned mean metrics will be zeros to indicate the run did not produce
     matching reports.
 - The evaluator performs partial-report cleanup on failures/timeouts and retries (exponential backoff).
 - The `report_count` is diagnostic and can be used to decide whether to re-run or reject the candidate.

Example:
 >>> evaluate_candidate('config/v448/mtf_candidate_0.json', seeds=3)
 {'mean_sharpe': 0.67, 'mean_total_return': 0.12, 'report_count': 3, 'run_artifacts': ['reports/training_report_x.json']}
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict

from ztb.io.json_io import read_json_object


class CandidateEvaluationResult(TypedDict):
    mean_sharpe: float
    mean_total_return: float
    report_count: int
    run_artifacts: list[str]


def _empty_result() -> CandidateEvaluationResult:
    return {
        "mean_sharpe": 0.0,
        "mean_total_return": 0.0,
        "report_count": 0,
        "run_artifacts": [],
    }


def _load_json_object(path: Path) -> dict[str, object] | None:
    try:
        return read_json_object(path)
    except Exception:
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_nested_str(payload: dict[str, object], keys: tuple[str, ...]) -> str | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, str) and current:
        return current
    return None


def _extract_candidate_model_name(payload: dict[str, object]) -> str | None:
    return _get_nested_str(payload, ("training", "model_name"))


def _extract_report_model_name(payload: dict[str, object]) -> str | None:
    return _get_nested_str(payload, ("configuration", "training", "model_name"))


def evaluate_candidate(
    cfg_path: str,
    seeds: int = 3,
    timesteps: int = 2000,
    dry_run: bool = False,
    retries: int = 2,
    timeout: Optional[int] = None,
    report_dir: Optional[str] = "reports",
) -> CandidateEvaluationResult:
    if dry_run:
        return _empty_result()

    log = logging.getLogger("CandidateEvaluator")
    cfg_payload = _load_json_object(Path(cfg_path))
    if cfg_payload is None:
        raise ValueError(f"Invalid candidate config (not JSON object): {cfg_path}")

    model_name = _extract_candidate_model_name(cfg_payload)
    if model_name is None:
        raise RuntimeError("Invalid candidate config: missing training.model_name")

    attempt = 0
    success = False
    last_exc: Exception | None = None
    cmd = [
        sys.executable,
        "tools/ab_test_runner.py",
        "--configs",
        cfg_path,
        "--seeds",
        str(seeds),
        "--timesteps",
        str(timesteps),
    ]
    base_delay = 1

    while attempt <= retries and not success:
        attempt += 1
        try:
            log.info("Running candidate: %s attempt %s/%s", cfg_path, attempt, retries + 1)
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ab_test_runner failed (rc={proc.returncode}): {proc.stderr}"
                )
            success = True
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            log.warning("Candidate run attempt %s timed out: %s", attempt, exc)
            _cleanup_partial_reports(model_name, report_dir)
            time.sleep(base_delay * attempt)
            continue
        except Exception as exc:
            last_exc = exc
            log.warning("Candidate run attempt %s failed: %s", attempt, exc)
            _cleanup_partial_reports(model_name, report_dir)
            time.sleep(base_delay * attempt)

    if not success:
        raise RuntimeError(
            f"Candidate evaluation failed after {attempt} attempts: {last_exc}"
        )

    report_dir_path = Path(report_dir or "reports")
    total_sharpe = 0.0
    total_return = 0.0
    report_count = 0
    report_paths: list[str] = []

    for report_path in report_dir_path.glob("training_report_*.json"):
        payload = _load_json_object(report_path)
        if payload is None:
            continue
        if _extract_report_model_name(payload) != model_name:
            continue

        training_stats = payload.get("training_stats")
        if isinstance(training_stats, dict):
            sharpe = _safe_float(training_stats.get("sharpe_ratio"))
            total_ret = _safe_float(training_stats.get("total_return"))
        else:
            sharpe = 0.0
            total_ret = 0.0

        total_sharpe += sharpe
        total_return += total_ret
        report_count += 1
        report_paths.append(str(report_path))

    if report_count == 0:
        log.warning(
            "No reports found for candidate model_name=%s. Returning zeros.",
            model_name,
        )
        return _empty_result()

    return {
        "mean_sharpe": total_sharpe / report_count,
        "mean_total_return": total_return / report_count,
        "report_count": report_count,
        "run_artifacts": report_paths,
    }


def _cleanup_partial_reports(
    model_name: Optional[str], report_dir: Optional[str] = "reports"
) -> None:
    if not model_name:
        return
    report_dir_path = Path(report_dir or "reports")
    for report_path in report_dir_path.glob("training_report_*.json"):
        payload = _load_json_object(report_path)
        if payload is None:
            continue
        if _extract_report_model_name(payload) != model_name:
            continue
        try:
            report_path.unlink()
        except OSError:
            continue
