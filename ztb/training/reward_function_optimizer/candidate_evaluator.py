"""
CandidateEvaluator: wrapper that runs `tools/ab_test_runner.py` for a candidate config and returns aggregated metrics.

This is a minimal wrapper for MVP: it supports dry_run and basic metric parsing.

Return contract:
 - evaluate_candidate(...) -> Dict[str, float]
 - Common keys returned in the dict:
        - mean_sharpe: float - arithmetic mean of reported sharpe_ratio values across reported runs (0.0 if none)
        - mean_total_return: float - arithmetic mean of reported total_return values across reported runs (0.0 if none)
        - report_count: int - the number of successful reports (training_report_*.json) that matched the
            candidate's `training.model_name`. This is useful to detect partial failures or timeouts — it is
            recommended to gate on `report_count` >= `seeds` (or other tolerances) before trusting other metrics.

Notes:
 - If `report_count` is 0 the returned mean metrics will be zeros to indicate the run did not produce
     matching reports.
 - The evaluator performs partial-report cleanup on failures/timeouts and retries (exponential backoff).
 - The `report_count` is diagnostic and can be used to decide whether to re-run or reject the candidate.

Example:
 >>> evaluate_candidate('config/v448/mtf_candidate_0.json', seeds=3)
 {'mean_sharpe': 0.67, 'mean_total_return': 0.12, 'report_count': 3}
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def evaluate_candidate(
    cfg_path: str,
    seeds: int = 3,
    timesteps: int = 2000,
    dry_run: bool = False,
    retries: int = 2,
    timeout: Optional[int] = None,
    report_dir: Optional[str] = "reports",
) -> Dict[str, float]:
    if dry_run:
        return {"mean_sharpe": 0.0, "mean_total_return": 0.0}

    log = logging.getLogger("CandidateEvaluator")
    attempt = 0
    success = False
    last_exc = None
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
    # Exponential backoff defaults
    base_delay = 1

    while attempt <= retries and not success:
        attempt += 1
        try:
            log.info(f"Running candidate: {cfg_path} attempt {attempt}/{retries+1}")
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if p.returncode != 0:
                raise RuntimeError(
                    f"ab_test_runner failed (rc={p.returncode}): {p.stderr}"
                )
            success = True
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            log.warning(f"Candidate run attempt {attempt} timed out: {exc}")
            # cleanup partial reports
            try:
                cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
                model_name = cfg.get("training", {}).get("model_name")
                if model_name:
                    _cleanup_partial_reports(model_name, report_dir)
            except Exception:
                pass
            time.sleep(base_delay * attempt)
            continue
        except Exception as exc:
            last_exc = exc
            log.warning(f"Candidate run attempt {attempt} failed: {exc}")
            # cleanup any partial reports for candidate model_name to avoid mix
            try:
                cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
                model_name = cfg.get("training", {}).get("model_name")
                if not model_name:
                    log.warning(
                        "Candidate config is missing training.model_name; aborting evaluate_candidate."
                    )
                    raise RuntimeError(
                        "Invalid candidate config: missing training.model_name"
                    )
                _cleanup_partial_reports(model_name, report_dir)
            except Exception:
                pass
            # backoff - increase delay after each failed attempt
            time.sleep(base_delay * attempt)
    if not success:
        raise RuntimeError(
            f"Candidate evaluation failed after {attempt} attempts: {last_exc}"
        )

    # Parse reports for model_name
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    model_name = cfg.get("training", {}).get("model_name")
    if not model_name:
        raise ValueError("Invalid candidate config: missing training.model_name")
    rd_str = report_dir or "reports"
    report_dir_path = Path(rd_str)
    matches = list(report_dir_path.glob("training_report_*.json"))
    relevant = []
    for m in matches:
        try:
            obj = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = obj.get("configuration", {}).get("training", {}).get("model_name")
        if name == model_name:
            # Attach the report path for artifact listing
            obj["__file__"] = str(m)
            relevant.append(obj)
    if not relevant:
        log.warning(
            f"No reports found for candidate model_name={model_name}. Returning zeros."
        )
        return {"mean_sharpe": 0.0, "mean_total_return": 0.0}

    total_sharpe = 0.0
    total_return = 0.0
    n = 0
    report_paths: list[str] = []
    for r in relevant:
        ts = r.get("training_stats") or {}
        try:
            s = float(ts.get("sharpe_ratio", 0.0))
        except Exception:
            s = 0.0
        try:
            tr = float(ts.get("total_return", 0.0))
        except Exception:
            tr = 0.0
        total_sharpe += s
        total_return += tr
        n += 1
        # Add the path if available in the report object metadata
        if isinstance(r.get("__file__"), str):
            report_paths.append(r.get("__file__"))
        mean_sharpe = total_sharpe / n if n > 0 else 0.0
    mean_return = total_return / n if n > 0 else 0.0
    # Also return paths of the relevant reports for debugging
    return {
        "mean_sharpe": mean_sharpe,
        "mean_total_return": mean_return,
        "report_count": n,
        "run_artifacts": report_paths,
    }


def _cleanup_partial_reports(
    model_name: Optional[str], report_dir: Optional[str] = "reports"
) -> None:
    if not model_name:
        return
    rd = Path(report_dir or "reports")
    for p in rd.glob("training_report_*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            obj.get("configuration", {}).get("training", {}).get("model_name")
            == model_name
        ):
            try:
                p.unlink()
            except Exception:
                pass
