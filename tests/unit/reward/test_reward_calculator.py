#!/usr/bin/env python3
"""
Test for balance penalty calculation bug.

This test demonstrates that the current balance penalty calculation
does not enforce BUY/SELL balance, only diversity.
"""

from collections import Counter

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.rewards.utils import RewardUtils


def _action_counts_from_actions(actions):
    c = [0, 0, 0]
    for a in actions:
        if a == ACTION_HOLD:
            c[0] += 1
        elif a == ACTION_BUY:
            c[1] += 1
        elif a == ACTION_SELL:
            c[2] += 1
    return c

# Note: tests below use the canonical RewardUtils.calculate_balance_penalty
# to avoid duplicating balance penalty logic in tests.

def test_balance_penalty_bug():
    """Test that demonstrates the balance penalty behaviour using RewardUtils."""
    scale = 1000.0

    # All SELL actions
    all_sell = [ACTION_SELL] * 10
    counts_sell = _action_counts_from_actions(all_sell)

    # All BUY actions
    all_buy = [ACTION_BUY] * 10
    counts_buy = _action_counts_from_actions(all_buy)

    # Balanced actions
    balanced = [ACTION_BUY, ACTION_SELL, ACTION_HOLD] * 3 + [ACTION_BUY]  # 4 BUY, 3 SELL, 3 HOLD
    counts_balanced = _action_counts_from_actions(balanced)

    # Use asymmetric targets (hold=0.2, buy=0.4, sell=0.35) and normalize
    hold_target = 0.2
    buy_target = 0.4
    sell_target = 0.35
    tr = [hold_target, buy_target, sell_target]
    total_tr = sum(tr)
    target_ratios = [t / total_tr for t in tr]

    penalty_sell = RewardUtils.calculate_balance_penalty(counts_sell, target_ratios, 0.05, scale)
    penalty_buy = RewardUtils.calculate_balance_penalty(counts_buy, target_ratios, 0.05, scale)
    penalty_balanced = RewardUtils.calculate_balance_penalty(counts_balanced, target_ratios, 0.05, scale)

    print(f"All SELL penalty: {penalty_sell}")
    print(f"All BUY penalty: {penalty_buy}")
    print(f"Balanced penalty: {penalty_balanced}")

    # Sanity checks: all-BUY and all-SELL should differ, and balanced should be lower
    assert penalty_sell != penalty_buy, f"Penalties should differ, got SELL: {penalty_sell}, BUY: {penalty_buy}"
    assert penalty_balanced < min(penalty_sell, penalty_buy), (
        f"Balanced penalty {penalty_balanced} should be less than both {penalty_sell} and {penalty_buy}"
    )


def test_improved_balance_penalty():
    """Test improved balance penalty via RewardUtils (sanity checks)."""
    scale = 1000.0

    # All SELL actions
    all_sell = [ACTION_SELL] * 10
    counts_sell = _action_counts_from_actions(all_sell)

    # All BUY actions
    all_buy = [ACTION_BUY] * 10
    counts_buy = _action_counts_from_actions(all_buy)

    # Balanced actions
    balanced = [ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_BUY, ACTION_SELL] * 2  # 4 BUY, 4 SELL, 2 HOLD
    counts_balanced = _action_counts_from_actions(balanced)

    target_ratios = [0.333, 0.333, 0.334]
    penalty_sell = RewardUtils.calculate_balance_penalty(counts_sell, target_ratios, 0.05, scale)
    penalty_buy = RewardUtils.calculate_balance_penalty(counts_buy, target_ratios, 0.05, scale)
    penalty_balanced = RewardUtils.calculate_balance_penalty(counts_balanced, target_ratios, 0.05, scale)

    print(f"Improved - All SELL penalty: {penalty_sell}")
    print(f"Improved - All BUY penalty: {penalty_buy}")
    print(f"Improved - Balanced penalty: {penalty_balanced}")

    # Now all-SELL and all-BUY should have different penalties
    assert penalty_sell != penalty_buy, f"Penalties should differ, got SELL: {penalty_sell}, BUY: {penalty_buy}"

    # Balanced should have lower penalty
    assert penalty_balanced < penalty_sell, f"Balanced penalty {penalty_balanced} should be less than {penalty_sell}"


if __name__ == "__main__":
    test_balance_penalty_bug()
    test_improved_balance_penalty()