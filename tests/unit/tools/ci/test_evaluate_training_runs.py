import json
from subprocess import run


def test_evaluate_training_runs_creates_summary(tmp_path, monkeypatch):
    # Create a fake reports directory with a training report
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    rpt = {
        "configuration": {"training": {"model_name": "test_model"}},
        "training_stats": {"sharpe_ratio": 0.85, "total_return": 0.12},
    }
    rpt_path = rpt_dir / "training_report_test.json"
    rpt_path.write_text(json.dumps(rpt), encoding="utf-8")

    # Change working dir to tmp_path for script to find reports/
    monkeypatch.chdir(tmp_path)

    out_path = tmp_path / "reports" / "ab_summary.json"
    # Run the script
    # Use the repository root file path so the script can be found regardless of working dir
    from pathlib import Path as _P

    repo_root = _P(__file__).resolve().parents[4]
    script_path = str(repo_root / "tools" / "ci" / "evaluate_training_runs.py")
    res = run(["python", script_path, "--out", str(out_path)], cwd=str(tmp_path))
    assert res.returncode == 0
    assert out_path.exists()

    obj = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(obj, list)
    # The new output groups by model_name and reports aggregated mean values
    assert obj[0].get("model_name") == "test_model"
    assert obj[0].get("mean_sharpe") == 0.85
    assert obj[0].get("report_count") == 1
