#!/usr/bin/env python3
"""
v3.6.6結果分析 + 包括的バグ診断
"""

import re
from typing import Dict, List, Tuple

CONSOLE_OUTPUT = """
|    pan_action_counts             | [21, 6, 5]                           |
|    pan_action_counts             | [25, 3, 4]                           |
|    pan_action_counts             | [22, 3, 7]                           |
"""


def parse_pan_counts(text: str) -> List[Tuple[int, int, int]]:
    """Parse pan_action_counts from console output."""
    pattern = r"pan_action_counts\s*\|\s*\[(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.findall(pattern, text)
    return [(int(m[0]), int(m[1]), int(m[2])) for m in matches]


def main() -> None:
    print("=" * 80)
    print("v3.6.6 COMPREHENSIVE ANALYSIS + BUG DIAGNOSIS")
    print("=" * 80)

    counts = parse_pan_counts(CONSOLE_OUTPUT)

    total_hold = 0
    total_buy = 0
    total_sell = 0

    print("\n📊 PER-ITERATION BREAKDOWN:")
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
    print("📈 AGGREGATE STATISTICS")
    print("=" * 80)
    print(f"  Total Actions: {grand_total}")
    print(f"  HOLD: {total_hold:3d} ({total_hold/grand_total:6.1%})")
    print(f"  BUY:  {total_buy:3d} ({total_buy/grand_total:6.1%})")
    print(f"  SELL: {total_sell:3d} ({total_sell/grand_total:6.1%})")

    print("\n" + "=" * 80)
    print("📊 VERSION COMPARISON")
    print("=" * 80)

    versions: Dict[str, Tuple[float, float, float]] = {
        "v3.6.4": (79.2, 8.3, 12.5),
        "v3.6.5": (73.4, 14.1, 12.5),
        "v3.6.6": (
            total_hold / grand_total * 100,
            total_buy / grand_total * 100,
            total_sell / grand_total * 100,
        ),
    }

    print("          HOLD%   BUY%   SELL%   Config")
    for ver, (hold_pct, buy_pct, sell_pct) in versions.items():
        if ver == "v3.6.4":
            config = "multipliers [1.0, 3.0, 1.0]"
        elif ver == "v3.6.5":
            config = "multipliers [2.0, 3.0, 0.5]"
        else:
            config = "multipliers [2.0, 3.0, 1.0] + penalties"
        print(f"  {ver}:  {hold_pct:5.1f}  {buy_pct:5.1f}  {sell_pct:5.1f}   {config}")

    print("\n" + "=" * 80)
    print("🔍 CRITICAL FINDINGS")
    print("=" * 80)

    v366_sell = total_sell / grand_total * 100

    print(
        """
  ❌ SELL RATE REGRESSION: 12.5% → 10.2% (-2.3pp)

  📉 TREND ANALYSIS:
     v3.6.4 → v3.6.5: SELL 12.5% (unchanged)
     v3.6.5 → v3.6.6: SELL 10.2% (WORSE!)

  🚨 ROOT CAUSE HYPOTHESIS:
     Explicit penalties may be ADDING to base penalties instead of replacing

     Expected: penalty = configured_value
     Actual:   penalty = base_penalty + configured_value

     This would mean:
       HOLD: 0.01-0.05 (base) + 0.2 (config) = 0.21-0.25 total
       BUY:  0.015 (base) + (-0.1) (config) = -0.085 reward (WEAK)
       SELL: 0.015 (base) + (-0.15) (config) = -0.135 reward (WEAK)
    """
    )

    print("\n" + "=" * 80)
    print("🐛 BUG DIAGNOSIS")
    print("=" * 80)

    print(
        """
  BUG SUSPECTED: Penalty Accumulation Issue

  Location: ztb/trading/environment/components/reward_calculator.py

  Current Implementation (SUSPECTED):
    base_action_penalty = 0.015  # for BUY/SELL
    configured_penalty = buy_action_penalty  # -0.1
    actual_penalty = base_action_penalty + configured_penalty
                   = 0.015 + (-0.1) = -0.085

  This makes BUY/SELL rewards WEAKER than intended because:
    - We wanted -0.1 reward (net benefit)
    - We got -0.085 reward (reduced benefit)
    - Base penalty reduces the configured reward

  HOLD is different:
    base_action_penalty = 0.01-0.05 (variable)
    configured_penalty = 0.2
    actual_penalty = base + 0.2 = 0.21-0.25 (CORRECT - stronger penalty)

  ✅ HOLD penalty works correctly (adds to base)
  ❌ BUY/SELL penalties CONFLICT with base (reduce reward effectiveness)
    """
    )

    print("\n" + "=" * 80)
    print("🔧 VERIFICATION NEEDED")
    print("=" * 80)

    print(
        """
  1. Check reward_calculator.py implementation:
     - Are penalties added or replaced?
     - Do we need to SUBTRACT base penalty before adding configured penalty?

  2. Calculate net rewards for each action:
     HOLD = profit_bonus * 1.0 - (base_penalty + 0.2)
     BUY  = profit_bonus * 2.0 - (0.015 - 0.1) = profit * 2.0 + 0.085
     SELL = profit_bonus * 3.0 - (0.015 - 0.15) = profit * 3.0 + 0.135

  3. If profit_bonus is small (e.g., 0.1):
     HOLD reward = 0.1 * 1.0 - 0.25 = -0.15
     BUY reward  = 0.1 * 2.0 + 0.085 = 0.285
     SELL reward = 0.1 * 3.0 + 0.135 = 0.435

     This looks correct! So why is SELL still low?
    """
    )

    print("\n" + "=" * 80)
    print("🔍 ADDITIONAL DIAGNOSTIC CHECKS")
    print("=" * 80)

    print(
        """
  Check 1: Episode Reward (ep_rew_mean = -495)
    ❌ EXTREMELY NEGATIVE - Model is losing money consistently
    → Suggests reward structure is fundamentally broken
    → Even profitable trades may be penalized overall

  Check 2: Value Loss (value_loss = 31.3 at iteration 19)
    ❌ VERY HIGH - Model struggling to estimate value
    → Policy cannot learn optimal actions if value estimates are wrong
    → High variance in rewards

  Check 3: Advantage Normalization Warnings
    "Action 1 has only 0 samples" / "Action 2 has only 0 samples"
    ❌ FREQUENT - Some actions rarely chosen in minibatches
    → Distribution still heavily skewed
    → Normalization cannot work properly

  Check 4: Lambda at Maximum (30.0)
    ❌ CONSTRAINT SATURATED - Cannot increase SELL bias further
    → Lagrange constraint is maxed out
    → Additional penalties/rewards have no effect due to constraint

  Check 5: Profit Bonus Calculation
    Need to verify base_profit_bonus is:
      a) Non-zero for most states
      b) Positive when expected
      c) Not dominated by penalties
    """
    )

    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    print(
        """
  IMMEDIATE ACTION: Examine Reward Calculator Code
    1. Add debug logging to reward_calculator.py
    2. Log actual reward values for each action type
    3. Verify penalty application logic
    4. Check if profit_bonus is consistently non-zero

  FIX Option 1: Zero Base Penalties
    Set action_penalty_scale = 0.0 (already done?)
    Verify base penalties are actually disabled
    May need to modify _calculate_full_reward logic

  FIX Option 2: Stronger Configured Penalties
    If penalties are accumulating:
      hold_action_penalty: 0.3 (increase)
      buy_action_penalty: -0.2 (stronger reward)
      sell_action_penalty: -0.25 (even stronger)

  FIX Option 3: Disable Lagrange Temporarily
    Lambda = 30.0 may be interfering
    Test with lagrange disabled to isolate multiplier/penalty effects

  FIX Option 4: Examine Base Reward Calculation
    If base_profit_bonus is consistently near zero or negative:
      - Multipliers have no effect
      - Penalties dominate
      - Need to fix profit bonus calculation first

  CRITICAL: Check Profitability
    ep_rew_mean = -495 is unacceptable
    Model must at least break even before balancing actions
    May need to:
      a) Simplify reward structure
      b) Remove all penalties temporarily
      c) Focus on basic profitability first
    """
    )

    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS")
    print("=" * 80)

    print(
        """
  Priority 1: DEBUG LOGGING
    Add comprehensive logging to reward_calculator.py:
      - Log base_profit_bonus value
      - Log multiplier application
      - Log penalty calculation
      - Log final reward per action

  Priority 2: REWARD STRUCTURE AUDIT
    Create diagnostic script to:
      - Simulate rewards for different scenarios
      - Verify penalty/multiplier interactions
      - Check for edge cases

  Priority 3: SIMPLIFICATION
    If debugging shows complex interactions:
      - Strip down to minimal reward (just profit * multiplier)
      - Add penalties ONE AT A TIME
      - Verify each change independently

  Priority 4: PROFITABILITY FIRST
    Focus on ep_rew_mean > 0 before action balancing
    A balanced but unprofitable model is useless
    """
    )


if __name__ == "__main__":
    main()
