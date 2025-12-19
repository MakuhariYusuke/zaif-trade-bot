import json
from pathlib import Path

from ztb.training.reward_function_optimizer.candidate_evaluator import (
    evaluate_candidate,
)


def test_evaluate_candidate_dry_run(tmp_path: Path):
    cfg = {
        "training": {"model_name": "mtf_test_candidate", "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "cand.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    metrics = evaluate_candidate(str(cfg_path), seeds=1, timesteps=100, dry_run=True)
    assert metrics["mean_sharpe"] == 0.0
    assert metrics["mean_total_return"] == 0.0


def test_evaluate_candidate_parses_reports(tmp_path: Path, monkeypatch):
    cfg = {
        "training": {"model_name": "mtf_test_candidate", "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "cand.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # stub subprocess.run to simulate a successful ab_test_runner run and to generate reports
    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        # Create synthetic report files
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        for i in range(2):
            report = {
                "configuration": {"training": {"model_name": "mtf_test_candidate"}},
                "training_stats": {
                    "sharpe_ratio": 0.5 + i * 0.1,
                    "total_return": 0.05 + i * 0.02,
                },
            }
            rpt_file = reports_dir / f"training_report_fake_{i}.json"
            rpt_file.write_text(json.dumps(report), encoding="utf-8")

        class R:
            returncode = 0
            stdout = "OK"
            stderr = ""

        return R()

    monkeypatch.setattr("subprocess.run", fake_run)
    # Run evaluate against reports dir set to tmp_path/reports
    from ztb.training.reward_function_optimizer.candidate_evaluator import (
        evaluate_candidate,
    )

    metrics = evaluate_candidate(
        str(cfg_path),
        seeds=1,
        timesteps=100,
        dry_run=False,
        report_dir=str(tmp_path / "reports"),
    )
    assert metrics["mean_sharpe"] > 0
    assert metrics["mean_total_return"] > 0
    assert metrics.get("report_count", 0) > 0
    assert isinstance(metrics.get("run_artifacts"), list)


def test_evaluate_candidate_retries(tmp_path: Path, monkeypatch):
    cfg = {
        "training": {"model_name": "mtf_test_candidate", "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "cand.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    calls = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        calls["n"] += 1
        # First call fails with non-zero rc
        if calls["n"] == 1:

            class R:
                returncode = 1
                stdout = ""
                stderr = "failed"

            return R()
        # Second call succeeds and writes reports
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        for i in range(1):
            report = {
                "configuration": {"training": {"model_name": "mtf_test_candidate"}},
                "training_stats": {"sharpe_ratio": 0.7, "total_return": 0.07},
            }
            rpt_file = reports_dir / f"training_report_fake_{i}.json"
            rpt_file.write_text(json.dumps(report), encoding="utf-8")


        return R()

    monkeypatch.setattr("subprocess.run", fake_run)
    metrics = evaluate_candidate(
        str(cfg_path),
        seeds=1,
        timesteps=100,
        dry_run=False,
        report_dir=str(tmp_path / "reports"),
        retries=2,
    )
    assert metrics["mean_sharpe"] > 0
    assert calls["n"] >= 2
    assert metrics.get("report_count", 0) > 0


def test_evaluate_candidate_timeout(tmp_path: Path, monkeypatch):
    cfg = {
        "training": {"model_name": "mtf_test_candidate", "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "cand.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    def fake_run_timeout(cmd, capture_output=True, text=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    import subprocess

    monkeypatch.setattr("subprocess.run", fake_run_timeout)
    import pytest

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


def test_evaluate_candidate_partial_report_cleanup(tmp_path: Path, monkeypatch):
    cfg = {
        "training": {"model_name": "mtf_test_candidate", "timesteps": 100},
        "multi_timeframe": {
            "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
        },
    }
    cfg_path = tmp_path / "cand.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # create a partial report before running
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    partial = reports_dir / "training_report_partial.json"
    partial.write_text(
        json.dumps(
            {
                "configuration": {"training": {"model_name": "mtf_test_candidate"}},
                "training_stats": {},
            }
        ),
        encoding="utf-8",
    )

    calls = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        calls["n"] += 1
        # Make the first run fail, second succeed and write final report
        if calls["n"] == 1:
            class R:
                returncode = 1
                stdout = ""
                stderr = "failed"
            return R()
        else:
            report = {
                "configuration": {"training": {"model_name": "mtf_test_candidate"}},
                "training_stats": {"sharpe_ratio": 0.6, "total_return": 0.06},
            }
            rpt_file = reports_dir / "training_report_final.json"
            rpt_file.write_text(json.dumps(report), encoding="utf-8")

            class R:
                returncode = 0
                stdout = "OK"
                stderr = ""
            return R()
    metrics = evaluate_candidate(
        str(cfg_path),
        seeds=1,
        timesteps=100,
        dry_run=False,
        report_dir=str(reports_dir),
        retries=2,
    )
    # partial report should be cleaned up
    assert not partial.exists()
    assert metrics.get("report_count", 0) > 0


def test_evaluate_candidate_missing_model_name(tmp_path: Path):
    cfg = tmp_path / "base_config.json"
    cfg.write_text(
        json.dumps({"training": {}, "multi_timeframe": {}}), encoding="utf-8"
    )
    import pytest

    with pytest.raises(RuntimeError):
        evaluate_candidate(str(cfg), seeds=1, timesteps=10, dry_run=False)
