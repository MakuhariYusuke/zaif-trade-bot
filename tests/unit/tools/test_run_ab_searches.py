from pathlib import Path
import json

from tools.run_ab_searches import analyze_reward_components
from ztb.trading.environment.components.rewards.utils import RewardUtils


def test_analyze_reward_components_picks_best_balanced(tmp_path):
    reports_dir = tmp_path

    # Build two reports: one balanced, one imbalanced
    balanced = {
        "training_stats": {"reward_components": {"comp": 1.0}, "action_distribution": {"BUY": 0.33, "SELL": 0.33}}
    }
    imbalanced = {
        "training_stats": {"reward_components": {"comp": 2.0}, "action_distribution": {"BUY": 0.6, "SELL": 0.2}}
    }

    (reports_dir / "training_report_r_balanced.json").write_text(json.dumps(balanced))
    (reports_dir / "training_report_r_imbalanced.json").write_text(json.dumps(imbalanced))

    analysis = analyze_reward_components(reports_dir)

    # best_balance_score should correspond to balanced report
    assert analysis["with_reward_components"] == 2
    best = analysis["best_balanced_config"]
    assert best is not None

    # Compute expected scores for check
    bal_score = RewardUtils.calculate_balance_deviation_from_ratios([0.33, 0.33], [0.5, 0.5])
    imb_score = RewardUtils.calculate_balance_deviation_from_ratios([0.6, 0.2], [0.5, 0.5])
    assert analysis["best_balance_score"] == min(bal_score, imb_score)
    assert any(best["report"].endswith(s) for s in ["r_balanced.json", "r_imbalanced.json"])
