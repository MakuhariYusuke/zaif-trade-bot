import json
import subprocess
from pathlib import Path

import pytest

from ztb.training.reward_function_optimizer.candidate_evaluator import (
    evaluate_candidate,
)


class _RunResult:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _write_candidate_config(tmp_path: Path, model_name: str = "mtf_test_candidate") -> Path:
    cfg = {
        "training": {"model_name": model_name, "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "cand.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _write_report(
    reports_dir: Path,
    file_name: str,
    model_name: str,
    sharpe: float | None = None,
    total_return: float | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_stats = {}
    if sharpe is not None:
        training_stats["sharpe_ratio"] = sharpe
    if total_return is not None:
        training_stats["total_return"] = total_return
    report = {
        "configuration": {"training": {"model_name": model_name}},
        "training_stats": training_stats,
    }
    report_path = reports_dir / file_name
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_evaluate_candidate_dry_run(tmp_path: Path) -> None:
    cfg_path = _write_candidate_config(tmp_path)
    metrics = evaluate_candidate(str(cfg_path), seeds=1, timesteps=100, dry_run=True)
    assert metrics["mean_sharpe"] == 0.0
    assert metrics["mean_total_return"] == 0.0
    assert metrics["report_count"] == 0


def test_evaluate_candidate_parses_only_current_run_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_candidate_config(tmp_path)
    reports_dir = tmp_path / "reports"

    # Pre-existing report for the same model should not be included.
    old_report = _write_report(
        reports_dir,
        "training_report_old.json",
        "mtf_test_candidate",
        sharpe=9.9,
        total_return=9.9,
    )

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        _write_report(
            reports_dir,
            "training_report_new_0.json",
            "mtf_test_candidate",
            sharpe=0.5,
            total_return=0.05,
        )
        _write_report(
            reports_dir,
            "training_report_new_1.json",
            "mtf_test_candidate",
            sharpe=0.6,
            total_return=0.07,
        )
        _write_report(
            reports_dir,
            "training_report_other.json",
            "other_model",
            sharpe=0.8,
            total_return=0.08,
        )
        return _RunResult(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    metrics = evaluate_candidate(
        str(cfg_path),
        seeds=1,
        timesteps=100,
        dry_run=False,
        report_dir=str(reports_dir),
    )
    assert old_report.exists()
    assert metrics["report_count"] == 2
    assert metrics["mean_sharpe"] == pytest.approx((0.5 + 0.6) / 2)
    assert metrics["mean_total_return"] == pytest.approx((0.05 + 0.07) / 2)
    assert len(metrics["run_artifacts"]) == 2


def test_evaluate_candidate_retries_and_cleans_partial_new_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_candidate_config(tmp_path)
    reports_dir = tmp_path / "reports"
    calls = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            _write_report(
                reports_dir,
                "training_report_partial.json",
                "mtf_test_candidate",
                sharpe=0.2,
                total_return=0.02,
            )
            return _RunResult(returncode=1, stderr="failed")

        _write_report(
            reports_dir,
            "training_report_final.json",
            "mtf_test_candidate",
            sharpe=0.7,
            total_return=0.07,
        )
        return _RunResult(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("time.sleep", lambda _: None)

    metrics = evaluate_candidate(
        str(cfg_path),
        seeds=1,
        timesteps=100,
        dry_run=False,
        report_dir=str(reports_dir),
        retries=2,
    )
    assert calls["n"] == 2
    assert not (reports_dir / "training_report_partial.json").exists()
    assert metrics["report_count"] == 1
    assert metrics["mean_sharpe"] == pytest.approx(0.7)


def test_evaluate_candidate_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = _write_candidate_config(tmp_path)

    def fake_run_timeout(cmd, capture_output=True, text=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("subprocess.run", fake_run_timeout)
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        evaluate_candidate(
            str(cfg_path),
            seeds=1,
            timesteps=100,
            dry_run=False,
            report_dir=str(tmp_path / "reports"),
            retries=1,
            timeout=1,
        )


def test_evaluate_candidate_preserves_preexisting_reports_during_retry_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_candidate_config(tmp_path)
    reports_dir = tmp_path / "reports"
    preexisting = _write_report(
        reports_dir,
        "training_report_preexisting.json",
        "mtf_test_candidate",
        sharpe=0.4,
        total_return=0.04,
    )
    calls = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            _write_report(
                reports_dir,
                "training_report_attempt1_partial.json",
                "mtf_test_candidate",
                sharpe=0.1,
                total_return=0.01,
            )
            return _RunResult(returncode=1, stderr="failed")
        _write_report(
            reports_dir,
            "training_report_attempt2_final.json",
            "mtf_test_candidate",
            sharpe=0.8,
            total_return=0.08,
        )
        return _RunResult(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("time.sleep", lambda _: None)

    metrics = evaluate_candidate(
        str(cfg_path),
        seeds=1,
        timesteps=100,
        dry_run=False,
        report_dir=str(reports_dir),
        retries=2,
    )

    assert preexisting.exists()
    assert not (reports_dir / "training_report_attempt1_partial.json").exists()
    assert metrics["report_count"] == 1
    assert metrics["mean_sharpe"] == pytest.approx(0.8)


def test_evaluate_candidate_missing_model_name(tmp_path: Path) -> None:
    cfg = tmp_path / "base_config.json"
    cfg.write_text(json.dumps({"training": {}, "multi_timeframe": {}}), encoding="utf-8")

    with pytest.raises(RuntimeError):
        evaluate_candidate(str(cfg), seeds=1, timesteps=10, dry_run=False)
