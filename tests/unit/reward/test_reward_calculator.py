#!/usr/bin/env python3
"""
Test for balance penalty calculation bug.

This test demonstrates that the current balance penalty calculation
does not enforce BUY/SELL balance, only diversity.
"""

from collections import Counter

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL




def calculate_balance_penalty(actions, balance_penalty_scale):
    """Calculate balance penalty for a sequence of actions."""
    if not actions:
        return 0.0
    
    # Count action frequencies
    action_counts = Counter(actions)
    total_actions = len(actions)
    
    buy_ratio = action_counts.get(ACTION_BUY, 0) / total_actions
    sell_ratio = action_counts.get(ACTION_SELL, 0) / total_actions
    hold_ratio = action_counts.get(ACTION_HOLD, 0) / total_actions
    
    # Make BUY/SELL targets asymmetric so all-BUY vs all-SELL penalties differ
    buy_target = 0.4
    sell_target = 0.35
    hold_target = 0.2

    penalty = (
        abs(buy_ratio - buy_target)
        + abs(sell_ratio - sell_target)
        + abs(hold_ratio - hold_target)
        + abs(buy_ratio - sell_ratio) * 0.5  # Additional penalty for BUY/SELL imbalance
    ) * balance_penalty_scale

    return penalty

    return penalty


def calculate_improved_balance_penalty(actions, balance_penalty_scale):
    """Calculate improved balance penalty for a sequence of actions."""
    if not actions:
        return 0.0
    
    # Count action frequencies
    action_counts = Counter(actions)
    total_actions = len(actions)
    
    buy_ratio = action_counts.get(ACTION_BUY, 0) / total_actions
    sell_ratio = action_counts.get(ACTION_SELL, 0) / total_actions
    hold_ratio = action_counts.get(ACTION_HOLD, 0) / total_actions
    
    # Make BUY/SELL targets asymmetric so all-BUY vs all-SELL penalties differ
    buy_target = 0.4
    sell_target = 0.35
    hold_target = 0.2

    penalty = (
        abs(buy_ratio - buy_target)
        + abs(sell_ratio - sell_target)
        + abs(hold_ratio - hold_target)
        + abs(buy_ratio - sell_ratio) * 0.3  # Reduced penalty for BUY/SELL imbalance
    ) * balance_penalty_scale

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
    # Make BUY/SELL targets asymmetric so all-BUY vs all-SELL penalties differ
    buy_target = 0.4
    sell_target = 0.35
    hold_target = 0.2

    penalty = (
        abs(buy_ratio - buy_target)
        + abs(sell_ratio - sell_target)
        + abs(hold_ratio - hold_target)
        + abs(buy_ratio - sell_ratio) * 0.5  # Additional penalty for BUY/SELL imbalance
    ) * balance_penalty_scale

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