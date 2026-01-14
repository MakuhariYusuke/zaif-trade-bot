import pytest

from ztb.trading.environment.components.rewards.utils import RewardUtils


def test_calculate_balance_penalty():
    # Perfect balance
    action_counts = [20, 40, 40]
    target_ratios = [0.2, 0.4, 0.4]
    tolerance = 0.05
    penalty_coeff = 1.0

    penalty = RewardUtils.calculate_balance_penalty(
        action_counts, target_ratios, tolerance, penalty_coeff
    )
    assert penalty == 0.0

    # Deviation within tolerance
    action_counts = [24, 38, 38]  # 0.24, 0.38, 0.38. Deviation 0.04, 0.02, 0.02
    penalty = RewardUtils.calculate_balance_penalty(
        action_counts, target_ratios, tolerance, penalty_coeff
    )
    assert penalty == 0.0

    # Deviation outside tolerance
    action_counts = [30, 35, 35]  # 0.3, 0.35, 0.35.
    # Hold: |0.3 - 0.2| = 0.1 > 0.05. Excess = 0.05
    # Buy: |0.35 - 0.4| = 0.05 <= 0.05. Excess = 0
    # Sell: |0.35 - 0.4| = 0.05 <= 0.05. Excess = 0

    penalty = RewardUtils.calculate_balance_penalty(
        action_counts, target_ratios, tolerance, penalty_coeff
    )
    assert penalty == pytest.approx(0.05)


def test_calculate_trading_bonus():
    assert RewardUtils.calculate_trading_bonus(1, 0.1) == 0.1  # BUY
    assert RewardUtils.calculate_trading_bonus(2, 0.1) == 0.1  # SELL
    assert RewardUtils.calculate_trading_bonus(0, 0.1) == 0.0  # HOLD


def test_calculate_position_penalty():
    # Below threshold
    assert RewardUtils.calculate_position_penalty(0.4, 1.0, 0.5, 0.2) == 0.0

    # Above threshold
    # Util = 0.6. Excess = 0.1. Penalty = 0.1 * 0.2 = 0.02
    assert RewardUtils.calculate_position_penalty(0.6, 1.0, 0.5, 0.2) == pytest.approx(
        0.02
    )


def test_calculate_activity_bonus():
    # Window size 5, min trades 2

    # Not enough trades
    recent_actions = [0, 0, 0, 0, 1]  # 1 trade
    assert RewardUtils.calculate_activity_bonus(recent_actions) == 0.0

    # Enough trades
    recent_actions = [0, 0, 0, 1, 2]  # 2 trades
    # Bonus rate 0.02. Ratio 2/5 = 0.4. Bonus = 0.02 * 0.4 = 0.008
    assert RewardUtils.calculate_activity_bonus(recent_actions) == pytest.approx(0.008)
