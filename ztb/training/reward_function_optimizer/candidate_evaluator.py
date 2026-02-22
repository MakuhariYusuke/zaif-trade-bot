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

from ztb.utils.safety import ensure_dict, safe_open_json, safe_to_float

ReportSignature = tuple[int, int]


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
    payload = safe_open_json(path)
    if payload is None:
        return None
    return ensure_dict(payload)


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


def _report_signature(report_path: Path) -> ReportSignature | None:
    try:
        stat = report_path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _find_reports_for_model(model_name: str, report_dir_path: Path) -> list[Path]:
    matches: list[Path] = []
    for report_path in report_dir_path.glob("training_report_*.json"):
        payload = _load_json_object(report_path)
        if payload is None:
            continue
        if _extract_report_model_name(payload) != model_name:
            continue
        matches.append(report_path)
    return matches


def _snapshot_report_state(report_paths: list[Path]) -> dict[Path, ReportSignature]:
    state: dict[Path, ReportSignature] = {}
    for report_path in report_paths:
        signature = _report_signature(report_path)
        if signature is not None:
            state[report_path] = signature
    return state


def _is_new_or_updated_report(
    report_path: Path, baseline_state: dict[Path, ReportSignature]
) -> bool:
    current_signature = _report_signature(report_path)
    if current_signature is None:
        return False
    previous_signature = baseline_state.get(report_path)
    if previous_signature is None:
        return True
    return current_signature != previous_signature


def _collect_current_run_reports(
    model_name: str,
    report_dir_path: Path,
    baseline_state: dict[Path, ReportSignature],
) -> list[Path]:
    report_paths = _find_reports_for_model(model_name, report_dir_path)
    current_run_reports = [
        report_path
        for report_path in report_paths
        if _is_new_or_updated_report(report_path, baseline_state)
    ]
    return sorted(current_run_reports)


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

    report_dir_path = Path(report_dir or "reports")
    report_dir_path.mkdir(parents=True, exist_ok=True)
    baseline_state = _snapshot_report_state(
        _find_reports_for_model(model_name, report_dir_path)
    )

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
            _cleanup_partial_reports(model_name, report_dir_path, baseline_state)
            time.sleep(base_delay * attempt)
            continue
        except Exception as exc:
            last_exc = exc
            log.warning("Candidate run attempt %s failed: %s", attempt, exc)
            _cleanup_partial_reports(model_name, report_dir_path, baseline_state)
            time.sleep(base_delay * attempt)

    if not success:
        raise RuntimeError(
            f"Candidate evaluation failed after {attempt} attempts: {last_exc}"
        )

    report_paths = _collect_current_run_reports(model_name, report_dir_path, baseline_state)
    total_sharpe = 0.0
    total_return = 0.0
    report_count = 0
    run_artifacts: list[str] = []

    for report_path in report_paths:
        payload = _load_json_object(report_path)
        if payload is None:
            continue

        training_stats = ensure_dict(payload.get("training_stats"))
        sharpe = safe_to_float(training_stats.get("sharpe_ratio"), 0.0)
        total_ret = safe_to_float(training_stats.get("total_return"), 0.0)

        total_sharpe += sharpe
        total_return += total_ret
        report_count += 1
        run_artifacts.append(str(report_path))

    if report_count == 0:
        log.warning(
            "No new reports found for candidate model_name=%s. Returning zeros.",
            model_name,
        )
        return _empty_result()

    return {
        "mean_sharpe": total_sharpe / report_count,
        "mean_total_return": total_return / report_count,
        "report_count": report_count,
        "run_artifacts": run_artifacts,
    }


def _cleanup_partial_reports(
    model_name: Optional[str],
    report_dir_path: Path,
    baseline_state: dict[Path, ReportSignature] | None = None,
) -> None:
    if not model_name:
        return
    baseline = baseline_state or {}
    for report_path in _find_reports_for_model(model_name, report_dir_path):
        if not _is_new_or_updated_report(report_path, baseline):
            continue
        try:
            report_path.unlink()
        except OSError:
            continue
