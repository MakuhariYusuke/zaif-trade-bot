import pytest
from ztb.trading.environment.components.rewards.utils import RewardUtils


def test_calculate_balance_penalty_min_actions_behavior():
    # total < 10 should return 0.0 according to implementation
    counts = [0, 3, 1]  # total 4
    penalty = RewardUtils.calculate_balance_penalty(counts, [0.4, 0.3, 0.3], 0.05, 100.0)
    assert penalty == 0.0

    # total == 10 should compute penalty
    counts = [0, 6, 4]
    penalty = RewardUtils.calculate_balance_penalty(counts, [0.4, 0.3, 0.3], 0.05, 100.0)
    assert penalty >= 0.0


@pytest.mark.parametrize(
    "counts,target_ratios,tol,coeff,expected_gt",
    [
        ([0, 5, 5], [0.4, 0.3, 0.3], 0.05, 10.0, True),  # balanced-ish -> small penalty
        ([0, 9, 1], [0.4, 0.3, 0.3], 0.05, 10.0, True),  # heavily imbalanced -> penalty > 0
        ([10, 0, 0], [0.4, 0.3, 0.3], 0.05, 5.0, True),  # all holds -> penalty expected
    ],
)
def test_calculate_balance_penalty_parametric(counts, target_ratios, tol, coeff, expected_gt):
    penalty = RewardUtils.calculate_balance_penalty(counts, target_ratios, tol, coeff)
    assert (penalty > 0) == expected_gt


def test_calculate_balance_penalty_target_length_mismatch():
    # target_ratios shorter than counts -> should stop at shortest length without error
    counts = [0, 8, 2]
    target_ratios = [0.33, 0.67]  # only two targets
    penalty = RewardUtils.calculate_balance_penalty(counts, target_ratios, 0.01, 50.0)
    assert isinstance(penalty, float)
    assert penalty >= 0.0
