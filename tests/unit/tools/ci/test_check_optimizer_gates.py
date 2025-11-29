import json
from pathlib import Path
from subprocess import run


def test_check_optimizer_gates_min_reports(tmp_path: Path):
    # Create a summary file with two candidates
    data = [
        {
            "model_name": "a",
            "mean_sharpe": 0.6,
            "mean_total_return": 0.06,
            "report_count": 3,
        },
        {
            "model_name": "b",
            "mean_sharpe": 0.7,
            "mean_total_return": 0.07,
            "report_count": 1,
        },
    ]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(data), encoding="utf-8")
    # Run gate requiring at least 2 reports; candidate 'b' should be filtered
    res = run(
        [
            "python",
            "tools/ci/check_optimizer_gates.py",
            "--summary",
            str(summary),
            "--min-reports",
            "2",
            "--sharpe",
            "0.5",
            "--return",
            "0.05",
        ]
    )
    assert res.returncode == 0
    # Run gate requiring at least 4 reports; no candidate should meet the count
    res2 = run(
        [
            "python",
            "tools/ci/check_optimizer_gates.py",
            "--summary",
            str(summary),
            "--min-reports",
            "4",
            "--sharpe",
            "0.5",
            "--return",
            "0.05",
        ]
    )
    assert res2.returncode != 0
