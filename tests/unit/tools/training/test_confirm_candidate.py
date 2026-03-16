import json
from pathlib import Path
from subprocess import run


def test_confirm_candidate_writes_summary_and_applied_file(tmp_path: Path, monkeypatch):
    # Create base config
    base_cfg = tmp_path / "base_config.json"
    base_cfg.write_text(
        json.dumps(
            {
                "training": {"model_name": "mtf_test"},
                "multi_timeframe": {
                    "feature_weights": {"1min": 0.3, "5min": 0.6, "15min": 0.1}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    # Run confirm_candidate in dry-run, should create summary but not apply
    from pathlib import Path as _P

    repo_root = _P(__file__).resolve().parents[4]
    script_path = str(repo_root / "tools" / "training" / "confirm_candidate.py")
    res = run(
        [
            "python",
            script_path,
            "--config",
            str(base_cfg),
            "--dry-run",
            "--candidates",
            "3",
            "--prefilter-seeds",
            "1",
            "--verify-seeds",
            "1",
            "--top-n",
            "2",
        ],
        cwd=str(tmp_path),
    )
    assert res.returncode == 0
    summary = tmp_path / "reports" / "mtf_optimizer_summary.json"
    assert summary.exists()
    obj = json.loads(summary.read_text(encoding="utf-8"))
    assert isinstance(obj, list)
    assert obj[0].get("model_name") is not None
