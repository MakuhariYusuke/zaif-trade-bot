#!/usr/bin/env python3
"""
Test for balance penalty calculation bug.

This test demonstrates that the current balance penalty calculation
does not enforce BUY/SELL balance, only diversity.
"""

import pytest
from collections import Counter

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL


def calculate_balance_penalty(actions, balance_penalty_scale=1000.0, target_ratio=0.333):
    """
    Calculate balance penalty as currently implemented (fixed).
    """
    counter = Counter(actions)
    total_actions = len(actions)

    buy_count = counter[ACTION_BUY]
    sell_count = counter[ACTION_SELL]
    hold_count = counter[ACTION_HOLD]

    buy_ratio = buy_count / total_actions
    sell_ratio = sell_count / total_actions
    hold_ratio = hold_count / total_actions

    penalty = abs(buy_ratio - sell_ratio) * balance_penalty_scale

    return penalty


def test_balance_penalty_bug():
    """Test that demonstrates the balance penalty bug."""
    scale = 1000.0

    # All SELL actions
    all_sell = [ACTION_SELL] * 10
    penalty_sell = calculate_balance_penalty(all_sell, scale)
    print(f"All SELL penalty: {penalty_sell}")

    # All BUY actions
    all_buy = [ACTION_BUY] * 10
    penalty_buy = calculate_balance_penalty(all_buy, scale)
    print(f"All BUY penalty: {penalty_buy}")

    # Balanced actions
    balanced = [ACTION_BUY, ACTION_SELL, ACTION_HOLD] * 3 + [ACTION_BUY]  # 4 BUY, 3 SELL, 3 HOLD
    penalty_balanced = calculate_balance_penalty(balanced, scale)
    print(f"Balanced penalty: {penalty_balanced}")

    # Assert that all-SELL and all-BUY have same penalty (bug)
    assert penalty_sell == penalty_buy, f"Expected same penalty, got SELL: {penalty_sell}, BUY: {penalty_buy}"

    # Balanced should have lower penalty
    assert penalty_balanced < penalty_sell, f"Balanced penalty {penalty_balanced} should be less than {penalty_sell}"

    print("Bug confirmed: all-SELL and all-BUY have identical penalty")


def calculate_improved_balance_penalty(actions, balance_penalty_scale=1000.0):
    """
    Improved balance penalty that enforces BUY/SELL balance.
    """
    counter = Counter(actions)
    total_actions = len(actions)

    buy_count = counter[ACTION_BUY]
    sell_count = counter[ACTION_SELL]
    hold_count = counter[ACTION_HOLD]

    buy_ratio = buy_count / total_actions
    sell_ratio = sell_count / total_actions
    hold_ratio = hold_count / total_actions

    # Enforce BUY/SELL balance (target 0.4 each) and HOLD (0.2)
    buy_sell_target = 0.4
    hold_target = 0.2

    # Calculate penalties for deviation from targets
    buy_penalty = abs(buy_ratio - buy_sell_target)
    sell_penalty = abs(sell_ratio - buy_sell_target)
    hold_penalty = abs(hold_ratio - hold_target)

    # BUY actions are more expensive (transaction costs, position management)
    # so penalize BUY deviations more heavily
    buy_penalty *= 1.5

    # Additional penalty for BUY/SELL imbalance - penalize the excessive action more
    if buy_ratio > sell_ratio:
        # Too many BUYs - increase BUY penalty
        buy_penalty *= 2.0
    elif sell_ratio > buy_ratio:
        # Too many SELLs - increase SELL penalty
        sell_penalty *= 2.0

    penalty = (buy_penalty + sell_penalty + hold_penalty) * balance_penalty_scale

    return penalty


def test_improved_balance_penalty():
    """Test improved balance penalty."""
    scale = 1000.0

    # All SELL actions
    all_sell = [ACTION_SELL] * 10
    penalty_sell = calculate_improved_balance_penalty(all_sell, scale)
    print(f"Improved - All SELL penalty: {penalty_sell}")

    # All BUY actions
    all_buy = [ACTION_BUY] * 10
    penalty_buy = calculate_improved_balance_penalty(all_buy, scale)
    print(f"Improved - All BUY penalty: {penalty_buy}")

    # Balanced actions
    balanced = [ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_BUY, ACTION_SELL] * 2  # 4 BUY, 4 SELL, 2 HOLD
    penalty_balanced = calculate_improved_balance_penalty(balanced, scale)
    print(f"Improved - Balanced penalty: {penalty_balanced}")

    # Now all-SELL and all-BUY should have different penalties
    assert penalty_sell != penalty_buy, f"Penalties should differ, got SELL: {penalty_sell}, BUY: {penalty_buy}"

    # Balanced should have lower penalty
    assert penalty_balanced < penalty_sell, f"Balanced penalty {penalty_balanced} should be less than {penalty_sell}"


if __name__ == "__main__":
    test_balance_penalty_bug()
    test_improved_balance_penalty()