from pathlib import Path


def test_quick_train_uses_rewardutils():
    p = Path("scripts/training/quick_train_v444_optimized.py")
    text = p.read_text(encoding="utf-8")

    assert "RewardUtils.calculate_buy_sell_diff" in text
    assert "abs(action_ratios[1] - action_ratios[2])" not in text
