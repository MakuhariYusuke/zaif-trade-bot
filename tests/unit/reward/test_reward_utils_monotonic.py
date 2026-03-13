from ztb.trading.environment.components.rewards.utils import RewardUtils


def test_balance_penalty_monotonic():
    target = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    tol = 0.05

    counts_less_skewed = [50, 25, 25]
    counts_more_skewed = [90, 5, 5]

    penalty_less = RewardUtils.calculate_balance_penalty(counts_less_skewed, target, tol, 100.0)
    penalty_more = RewardUtils.calculate_balance_penalty(counts_more_skewed, target, tol, 100.0)

    assert penalty_more >= penalty_less
    assert penalty_less >= 0

    # Scaling coefficient increases penalty magnitude
    penalty_scale_1 = RewardUtils.calculate_balance_penalty(counts_more_skewed, target, tol, 1.0)
    penalty_scale_100 = RewardUtils.calculate_balance_penalty(counts_more_skewed, target, tol, 100.0)

    assert penalty_scale_100 >= penalty_scale_1
