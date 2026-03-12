#!/usr/bin/env python3
"""
コンソール出力からアクション分布を解析

v3.6.4のトレーニングログから:
- pan_action_counts: 実際のHOLD/BUY/SELL分布
- lagrange metrics: SELL機会とLambda値
を抽出して分析
"""

import re
from typing import List, Tuple

CONSOLE_OUTPUT = """
| train/                           |                                      |
|    pan_action_counts             | [25, 3, 4]                           |
|    pan_action_means              | [-1.384340524673462, -0.692740440... |
|    pan_action_stds               | [1.2279605865478516, 0.9712451696... |
|    pan_total_samples             | 32                                   |

| train/                           |                                      |
|    pan_action_counts             | [27, 1, 4]                           |
|    pan_action_means              | [-0.8168232440948486, 1.094576835... |
|    pan_action_stds               | [3.582109212875366, 0.0, 1.858707... |
|    pan_total_samples             | 32                                   |

| train/                           |                                      |
|    pan_action_counts             | [24, 4, 4]                           |
|    pan_action_means              | [-1.5451091527938843, -1.78997576... |
|    pan_action_stds               | [1.413691759109497, 0.99322754144... |
|    pan_total_samples             | 32                                   |
"""


def parse_pan_counts(text: str) -> List[Tuple[int, int, int]]:
    """Parse pan_action_counts from console output."""
    pattern = r"pan_action_counts\s*\|\s*\[(\d+),\s*(\d+),\s*(\d+)\]"
    matches = re.findall(pattern, text)
    return [(int(m[0]), int(m[1]), int(m[2])) for m in matches]


def analyze_distribution(counts: List[Tuple[int, int, int]]) -> None:
    """Analyze action distribution."""
    print("=" * 80)
    print("ACTUAL ACTION DISTRIBUTION ANALYSIS")
    print("=" * 80)

    total_hold = 0
    total_buy = 0
    total_sell = 0

    print("\nPer-Iteration Breakdown (from console output):")
    print("  Format: [HOLD, BUY, SELL]")
    print()

    for i, (hold, buy, sell) in enumerate(counts, 1):
        total = hold + buy + sell
        total_hold += hold
        total_buy += buy
        total_sell += sell

        print(
            f"  Iteration {i}: [{hold:2d}, {buy:2d}, {sell:2d}]  "
            f"HOLD={hold/total:5.1%}  BUY={buy/total:5.1%}  SELL={sell/total:5.1%}"
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
    print("DIAGNOSIS")
    print("=" * 80)

    sell_rate = total_sell / grand_total
    target_rate = 0.33

    print(f"  Current SELL Rate: {sell_rate:.1%}")
    print(f"  Target SELL Rate:  {target_rate:.1%}")
    print(
        f"  Gap:               {(target_rate - sell_rate):.1%} ({(target_rate - sell_rate)*100:.1f} percentage points)"
    )

    # Analyze HOLD dominance
    hold_rate = total_hold / grand_total
    if hold_rate > 0.60:
        print(f"\n  ⚠️  HOLD is dominant at {hold_rate:.1%}")
        print("      → Agent is overly conservative")
        print(
            "      → Consider reducing HOLD incentives or increasing action penalties"
        )

    # Analyze BUY scarcity
    buy_rate = total_buy / grand_total
    if buy_rate < 0.15:
        print(f"\n  ⚠️  BUY is rare at {buy_rate:.1%}")
        print("      → Limited SELL opportunities (need position to sell)")
        print("      → SELL cannot exceed BUY in long term")
        print("      → Consider increasing BUY incentives")

    # Calculate theoretical max SELL
    print("\n  📊 Theoretical Maximum SELL Rate:")
    print(
        f"      With BUY={buy_rate:.1%}, SELL cannot sustainably exceed {buy_rate:.1%}"
    )
    print(f"      Current SELL={sell_rate:.1%} is {sell_rate/buy_rate:.1f}x BUY rate")

    if sell_rate > buy_rate * 0.8:
        print("      → SELL is close to BUY - position turnover is high")
        print("      → To increase SELL further, must increase BUY first")


def suggest_fixes() -> None:
    """Suggest specific configuration fixes."""
    print("\n" + "=" * 80)
    print("RECOMMENDED FIXES")
    print("=" * 80)

    print(
        """
ROOT CAUSE IDENTIFIED: BUY scarcity limits SELL opportunities

Fix Strategy 1: Balance BUY/SELL Multipliers
  Current: profit_bonus_multipliers = [1.0, 3.0, 1.0]  # [BUY, SELL, HOLD]
  Proposed: profit_bonus_multipliers = [2.0, 3.0, 0.5]

  Rationale:
    - Increase BUY to 2.0 → Create more SELL opportunities
    - Keep SELL at 3.0 → Maintain SELL incentive
    - Reduce HOLD to 0.5 → Discourage excessive holding

  Expected Impact: BUY 20-25%, SELL 15-20%, HOLD 55-60%

Fix Strategy 2: Increase Lambda Max + Multiplier Adjustment
  Current: lagrange_lambda_max = 30.0
  Proposed: lagrange_lambda_max = 50.0
  Plus: profit_bonus_multipliers = [1.5, 4.0, 0.5]

  Rationale:
    - Higher lambda allows stronger constraint enforcement
    - Stronger SELL multiplier (4.0) with BUY support (1.5)
    - Heavily penalize HOLD (0.5)

  Expected Impact: BUY 25-30%, SELL 18-25%, HOLD 45-55%

Fix Strategy 3: Add Action Penalties
  Current: All penalties = 0.0
  Proposed:
    - hold_action_penalty = 0.1
    - buy_action_penalty = -0.05  (small reward)
    - sell_action_penalty = -0.1  (reward)

  Rationale:
    - Direct penalty for HOLD actions
    - Small reward for BUY/SELL to encourage trading

  Expected Impact: BUY 20-25%, SELL 15-20%, HOLD 55-60%

RECOMMENDED COMBINATION (Strategy 1 + Partial Strategy 3):
  profit_bonus_multipliers = [2.0, 3.0, 0.5]
  hold_action_penalty = 0.05
  buy_action_penalty = 0.0
  sell_action_penalty = 0.0
  lagrange_lambda_max = 30.0  (keep current)

This provides balanced incentives without aggressive changes.
    """
    )


def main() -> None:
    """Main analysis."""
    print("CONSOLE LOG ANALYSIS - v3.6.4 SELL Rate Investigation")
    print()

    counts = parse_pan_counts(CONSOLE_OUTPUT)

    if not counts:
        print("❌ No pan_action_counts found in console output")
        return

    analyze_distribution(counts)
    suggest_fixes()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - Ready for v3.6.5 configuration")
    print("=" * 80)


if __name__ == "__main__":
    main()
