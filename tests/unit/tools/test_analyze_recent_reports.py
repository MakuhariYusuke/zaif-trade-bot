import json
import pytest

import tools.analyze_recent_reports as recent_reports_module
from ztb.trading.environment.components.rewards.utils import RewardUtils


def test_analyze_reports_uses_rewardutils_and_parses_reward(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_file = reports_dir / "training_report_1.json"
    data = {
        "training_stats": {
            "action_distribution": {"BUY": 0.6, "SELL": 0.3, "HOLD": 0.1},
            "final_reward": "1.23",
        }
    }
    report_file.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        recent_reports_module,
        "get_recent_training_reports",
        lambda limit, reports_dir: [report_file],
    )
    monkeypatch.setattr(
        recent_reports_module,
        "load_training_report",
        lambda path: data,
    )

    results = recent_reports_module.analyze_reports(limit=1)
    assert len(results) == 1
    r = results[0]
    assert r["file"] == report_file.name
    assert pytest.approx(r["reward"], rel=1e-6) == 1.23
    expected_diff = RewardUtils.calculate_buy_sell_diff(0.6, 0.3)
    assert r["buy_sell_diff"] == expected_diff
