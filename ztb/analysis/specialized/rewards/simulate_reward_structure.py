#!/usr/bin/env python3
"""
報酬構造のシミュレーション診断

実際の報酬計算をシミュレートして、なぜep_rew_mean=-495なのかを調査
"""


def simulate_rewards() -> None:
    """Simulate rewards for different scenarios."""

    print("=" * 80)
    print("REWARD STRUCTURE SIMULATION")
    print("=" * 80)

    # v3.6.6 configuration
    multipliers = [2.0, 3.0, 1.0]  # [BUY, SELL, HOLD]
    hold_penalty = 0.2
    buy_penalty = -0.1
    sell_penalty = -0.15

    # Simulate different profit scenarios
    scenarios = [
        ("Small Profit", 0.1, 1.0),
        ("Medium Profit", 0.5, 1.0),
        ("Large Profit", 1.0, 1.0),
        ("Small Loss", -0.1, 1.0),
        ("Medium Loss", -0.5, 1.0),
        ("Zero Profit", 0.0, 1.0),
    ]

    print("\nScenario Analysis (trend_multiplier=1.0, position=0.5):")
    print("-" * 80)

    for name, base_profit, trend_mult in scenarios:
        print(f"\n{name} (base_profit_bonus = {base_profit}):")

        # HOLD reward
        hold_profit_bonus = base_profit * multipliers[2] * trend_mult
        hold_base_penalty = 0.01 + (0.04 * 0.5 * 1.0)  # pos=0.5, vol=1.0
        hold_action_penalty = hold_base_penalty + hold_penalty
        hold_loss_penalty = -0.2 * abs(base_profit) if base_profit < 0 else 0.0
        hold_reward = hold_profit_bonus - hold_action_penalty + hold_loss_penalty

        # BUY reward
        buy_profit_bonus = base_profit * multipliers[0] * trend_mult
        buy_base_penalty = 0.015
        buy_action_penalty = buy_base_penalty + buy_penalty
        buy_loss_penalty = -0.2 * abs(base_profit) if base_profit < 0 else 0.0
        buy_reward = buy_profit_bonus - buy_action_penalty + buy_loss_penalty

        # SELL reward
        sell_profit_bonus = base_profit * multipliers[1] * trend_mult
        sell_base_penalty = 0.015
        sell_action_penalty = sell_base_penalty + sell_penalty
        sell_loss_penalty = -0.2 * abs(base_profit) if base_profit < 0 else 0.0
        sell_reward = sell_profit_bonus - sell_action_penalty + sell_loss_penalty

        print(
            f"  HOLD:  profit={hold_profit_bonus:+.3f} - penalty={hold_action_penalty:.3f} + loss={hold_loss_penalty:+.3f} = {hold_reward:+.3f}"
        )
        print(
            f"  BUY:   profit={buy_profit_bonus:+.3f} - penalty={buy_action_penalty:.3f} + loss={buy_loss_penalty:+.3f} = {buy_reward:+.3f}"
        )
        print(
            f"  SELL:  profit={sell_profit_bonus:+.3f} - penalty={sell_action_penalty:.3f} + loss={sell_loss_penalty:+.3f} = {sell_reward:+.3f}"
        )

        best = max(
            [("HOLD", hold_reward), ("BUY", buy_reward), ("SELL", sell_reward)],
            key=lambda x: x[1],
        )
        print(f"  → Best action: {best[0]} (reward={best[1]:+.3f})")

    print("\n" + "=" * 80)
    print("CRITICAL OBSERVATIONS")
    print("=" * 80)

    print(
        """
  1. If base_profit_bonus is consistently SMALL (< 0.1):
     - Penalties dominate rewards
     - All actions have negative rewards
     - Agent chooses "least bad" option (often HOLD)

  2. HOLD penalties are MUCH larger than BUY/SELL:
     - HOLD: 0.23-0.25 penalty
     - BUY:  -0.085 reward
     - SELL: -0.135 reward

  3. For zero/small profit states:
     - HOLD: 0.0 - 0.25 = -0.25
     - BUY:  0.0 + 0.085 = +0.085  ← Best choice!
     - SELL: 0.0 + 0.135 = +0.135  ← Even better!

  4. This explains BUY/SELL increase in v3.6.6:
     - Penalties make BUY/SELL REWARDING even without profit
     - HOLD becomes the worst option

  5. But ep_rew_mean = -495 suggests:
     - Most states have NEGATIVE base_profit_bonus
     - Or penalties are applied every step regardless of action
     - Or loss_penalty dominates
    """
    )

    print("\n" + "=" * 80)
    print("HYPOTHESIS: Why ep_rew_mean = -495?")
    print("=" * 80)

    print(
        """
  Possibility 1: Most trades are UNPROFITABLE
    - base_profit_bonus < 0 for most steps
    - Multipliers amplify losses
    - Loss penalties add additional negative rewards

  Possibility 2: Episode length penalty
    - Episode runs for 999 steps
    - Each step accumulates penalties
    - Even small per-step penalty becomes large
    - -495 / 999 ≈ -0.5 per step

  Possibility 3: Position penalties
    - position_penalty may be large
    - Applied every step
    - Not visible in our simulation

  Possibility 4: Reward structure mismatch
    - Environment may be using different reward calculation
    - Logged rewards may not match actual training rewards
    - Need to check actual environment code
    """
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ACTIONS NEEDED")
    print("=" * 80)

    print(
        """
  1. ADD REWARD LOGGING:
     In reward_calculator.py, log:
       - base_profit_bonus distribution
       - profit_bonus after multiplier
       - action_penalty values
       - loss_penalty values
       - position_penalty values
       - FINAL reward per step

  2. ANALYZE EPISODE REWARDS:
     - Sum of all step rewards should equal ep_rew
     - Identify which component dominates
     - Check for cumulative penalty effects

  3. CHECK PROFITABILITY BASELINE:
     - Run simple strategy (always HOLD)
     - Compare reward with current policy
     - If HOLD is also -495, problem is base reward calculation

  4. VERIFY CURRICULUM STAGE:
     - Check if forced_balance stage is applying additional penalties
     - Verify curriculum_stage is not interfering
    """
    )


if __name__ == "__main__":
    simulate_rewards()
