import pytest

from ztb.trading.environment.components.rewards.utils import RewardUtils


def test_calculate_balance_deviation_from_ratios_simple():
    ratios = [0.4, 0.4, 0.2]
    targets = [1 / 3, 1 / 3, 1 / 3]
    dev = RewardUtils.calculate_balance_deviation_from_ratios(ratios, targets)
    expected = sum(abs(r - t) for r, t in zip(ratios, targets))
    assert dev == pytest.approx(expected)


def test_calculate_balance_deviation_from_percentages_simple():
    percentages = [40.0, 35.0, 25.0]
    target = 33.3333
    dev = RewardUtils.calculate_balance_deviation_from_percentages(percentages, target)
    expected = sum(abs(p - target) for p in percentages)
    assert dev == pytest.approx(expected)


def test_deviation_helpers_empty_inputs():
    assert RewardUtils.calculate_balance_deviation_from_ratios([], [0.33, 0.33, 0.34]) == 0.0
    assert RewardUtils.calculate_balance_deviation_from_ratios([0.33, 0.33, 0.34], []) == 0.0
    assert RewardUtils.calculate_balance_deviation_from_percentages([], 33.3) == 0.0


def test_calculate_buy_sell_diff():
    assert RewardUtils.calculate_buy_sell_diff(0.6, 0.3) == 0.3
    assert RewardUtils.calculate_buy_sell_diff(0.33, 0.33) == 0.0
    # numeric robustness
    assert RewardUtils.calculate_buy_sell_diff(0, 1) == 1.0
