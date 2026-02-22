#!/usr/bin/env python3
"""
v3.6.5結果分析 - コンソール出力から抽出
"""

import re
from typing import List, Tuple

CONSOLE_OUTPUT = """
|    pan_action_counts             | [19, 6, 7]                           |

|    pan_action_counts             | [28, 3, 1]                           |
"""




def main() -> None:
    print("=" * 80)
    print("v3.6.5 RESULT ANALYSIS")
    print("=" * 80)

    counts = parse_pan_counts(CONSOLE_OUTPUT)

    total_hold = 0
    total_buy = 0
    total_sell = 0

    print("\nPer-Iteration Breakdown:")
    for i, (hold, buy, sell) in enumerate(counts, 1):
        total = hold + buy + sell
        total_hold += hold
        total_buy += buy
        total_sell += sell

        print(
            f"  Iteration {i}: HOLD={hold:2d} ({hold/total:5.1%})  "
            f"BUY={buy:2d} ({buy/total:5.1%})  "
            f"SELL={sell:2d} ({sell/total:5.1%})"
        )

    grand_total = total_hold + total_buy + total_sell

    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"  Total Actions: {grand_total}")
    print(f"  HOLD: {total_hold:3d} ({total_hold/grand_total:6.1%})")
    print(f"  BUY:  {total_buy:3d} ({total_buy/grand_total:6.1%})")
    print(f"  SELL: {total_sell:3d} ({total_sell/grand_total:6.1%})")

    print("\n" + "=" * 80)
    print("COMPARISON: v3.6.4 vs v3.6.5")
    print("=" * 80)

    # v3.6.4 results
    v364_hold = 79.2
    v364_buy = 8.3
    v364_sell = 12.5

    v365_hold = total_hold / grand_total * 100
    v365_buy = total_buy / grand_total * 100
    v365_sell = total_sell / grand_total * 100

    print("                v3.6.4    v3.6.5    Change")
    print(
        f"  HOLD:         {v364_hold:5.1f}%    {v365_hold:5.1f}%    {v365_hold - v364_hold:+6.1f}pp"
    )
    print(
        f"  BUY:          {v364_buy:5.1f}%    {v365_buy:5.1f}%    {v365_buy - v364_buy:+6.1f}pp"
    )
    print(
        f"  SELL:         {v364_sell:5.1f}%    {v365_sell:5.1f}%    {v365_sell - v364_sell:+6.1f}pp"
    )

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if v365_sell < v364_sell:
        print("  ❌ REGRESSION: SELL rate DECREASED")
        print(
            f"     {v364_sell:.1f}% → {v365_sell:.1f}% ({v365_sell - v364_sell:+.1f}pp)"
        )

    if v365_hold > 80:
        print("  ❌ HOLD dominance WORSENED")
        print(
            f"     {v364_hold:.1f}% → {v365_hold:.1f}% ({v365_hold - v364_hold:+.1f}pp)"
        )

    if v365_buy < v364_buy:
        print("  ❌ BUY rate DECREASED (unexpected)")
        print(f"     {v364_buy:.1f}% → {v365_buy:.1f}% ({v365_buy - v364_buy:+.1f}pp)")

    print("\n" + "=" * 80)
    print("HYPOTHESIS: Why did v3.6.5 fail?")
    print("=" * 80)
    print(
        """
  1. HOLD multiplier 0.5 may be too low:
     - If base rewards are negative or small, 0.5x still beats risky actions
     - HOLD becomes "safe" default even with penalty

  2. BUY multiplier 2.0 alone is insufficient:
     - BUY still has base action penalty (0.015)
     - Position opening is risky without guaranteed profit

  3. Reward structure may be dominated by penalties:
     - Action penalties, position penalties, loss penalties
     - Even with 2x multiplier, net reward may still be negative

  4. Short training time (10k steps):
     - Not enough time for policy to adapt to new reward structure
     - May need 30k+ steps for convergence

  5. Lagrange constraint interference:
     - Lambda=30.0 trying to enforce SELL
     - Conflicts with new multiplier structure
     - May need to disable Lagrange temporarily
    """
    )

    print("\n" + "=" * 80)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 80)
    print(
        """
  Option 1: MORE AGGRESSIVE MULTIPLIERS
    profit_bonus_multipliers: [3.0, 4.0, 0.3]
    - Stronger BUY/SELL incentives
    - Lower HOLD to discourage passivity

  Option 2: DISABLE PENALTIES + STRONGER MULTIPLIERS
    profit_bonus_multipliers: [3.0, 3.0, 0.5]
    action_penalty_scale: 0.0  (remove base penalties)
    position_penalty_scale: 0.0

  Option 3: LONGER TRAINING
    - Run v3.6.5 again with 30k steps
    - Check if trend improves over time

  Option 4: DISABLE LAGRANGE TEMPORARILY
    - Test multipliers without Lagrange constraint
    - See if multipliers alone can balance actions

  Option 5: ADD EXPLICIT PENALTIES (use new reward_calculator features)
    hold_action_penalty: 0.2  (strong penalty)
    buy_action_penalty: -0.1  (reward)
    sell_action_penalty: -0.15  (stronger reward)
    profit_bonus_multipliers: [2.0, 3.0, 1.0]  (revert HOLD)

  IMMEDIATE RECOMMENDATION: Option 5
    - Use newly implemented per-action penalties
    - More direct control over action preferences
    - Combines multipliers + explicit penalties
    """
    )


if __name__ == "__main__":
    main()
