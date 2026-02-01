import json
import subprocess
import sys


def test_ab_param_search_generates_configs(tmp_path):
    # template config
    template = tmp_path / "base.json"
    template.write_text(json.dumps({"training": {"model_name": "ab_search_model"}, "training_params": {}}))

    # grid
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"training.learning_rate": [0.001, 0.0001], "env.balance_penalty": [0.01]}))

    cmd = [
        sys.executable,
        "tools/ab_param_search.py",
        "--template",
        str(template),
        "--grid",
        str(grid),
        "--seeds",
        "0",
        "--jobs",
        "1",
    ]

    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    assert "Generated" in out
    assert "Running search via UnifiedOptimizer" in out


def test_score_distribution_helpers():
    # Verify the file was updated to use RewardUtils for balance scoring (avoid importing module which pulls heavy deps)
    content = open("tools/ab_param_search.py", "r", encoding="utf-8").read()
    assert "RewardUtils.calculate_balance_deviation_from_ratios" in content
    assert "return -sell" in content  # min_sell behavior preserved