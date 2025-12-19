import json
import subprocess
import sys


def test_ab_runner_parses_multiple_configs(tmp_path):
    # Create two dummy config files with minimal training.model_name
    c1 = tmp_path / "cfg1.json"
    c2 = tmp_path / "cfg2.json"
    c1.write_text(json.dumps({"training": {"model_name": "test_model_1"}}))
    c2.write_text(json.dumps({"training": {"model_name": "test_model_2"}}))

    # Call the ab_test_runner with seeds=0 so it doesn't attempt to run training.
    cmd = [
        sys.executable,
        "tools/ab_test_runner.py",
        "--configs",
        str(c1),
        str(c2),
        "--seeds",
        "0",
        "--jobs",
        "2",
    ]

    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # should exit without error
    assert completed.returncode == 0, f"ab_test_runner failed: {completed.stderr}"
    # It should mention the results for both model names in output
    out = completed.stdout
    assert "Results for test_model_1" in out
    assert "Results for test_model_2" in out
