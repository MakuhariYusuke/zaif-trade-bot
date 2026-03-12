"""
Quick simulation runner for v448 verification without requiring heavy ML frameworks.

This script exercises RewardCalculator, BehavioralPenaltyCalculator, TrendDetector,
and BalanceCurriculumManager in a light simulation loop to verify integration and
look for obvious regressions or logic issues in Layer 0-3.
"""

import math
import random
from types import SimpleNamespace

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.behavioral_penalty_calculator import (
    BehavioralPenaltyCalculator,
)
from ztb.trading.environment.components.reward.balance_curriculum import (
    BalanceCurriculumManager,
)
from ztb.trading.environment.components.reward.trend_detector import TrendDetector

# Use standard package imports; ztb.trading.environment is guarded against heavy imports


def make_config() -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.curriculum_stage = "forced_balance"
    cfg.max_position_size = 1.0
    cfg.action_bonuses = {
        "buy_action_bonus": 0.0,
        "sell_action_bonus": 0.0,
        "hold_action_bonus": 0.0,
    }
    cfg.signal_guidance = {}
    cfg.curriculum_learning = {
        "enabled": True,
        "auto_progression": True,
        "emergency_revert": True,
    }

    # Reward settings (nested dict for get_setting_x helpers)
    cfg.reward_settings = {
        "behavior_optimization": {
            "action_balance_target": 0.45,
            "entropy_regularization": 0.05,
            "action_smoothing": 0.06,
            "consistency_penalty": 0.03,
        },
        "forced_balance_min_actions": 10,
        "forced_balance_exploration_reward": 2.0,
        "forced_balance_threshold": 0.15,
    }
    return cfg


def simulate_steps(steps=300) -> None:
    cfg = make_config()
    # Instantiate components directly to avoid heavy imports (torch)
    tdet = TrendDetector(min_samples=5)
    bpc = BehavioralPenaltyCalculator(config=cfg)
    bpc.trend_detector = tdet
    cm = BalanceCurriculumManager(
        config=cfg, enabled=True, auto_progression=True, emergency_revert=True
    )

    # Simulate price data and a naive policy (biased towards BUY early, balanced later)
    prices = [1000.0 + 0.5 * math.sin(i / 10.0) for i in range(steps)]
    position = 0.0
    portfolio_value = 100000.0
    atr = 1.0
    transaction_cost = 0.001
    reward_history = []
    portfolio_value_history = []
    old_position = 0.0

    logs = {
        "stage_changes": [],
        "emergency_triggers": 0,
        "stage_history": [],
    }

    for step in range(steps):
        price = prices[step]
        # Update trend detector
        tdet.update(price)

        # Decide action: early bias to BUY, then balanced
        if step < max(20, steps // 5):
            action = 1 if random.random() < 0.8 else 0  # BUY biased
        else:
            p = random.random()
            if p < 0.45:
                action = ACTION_BUY
            elif p < 0.9:
                action = ACTION_HOLD
            else:
                action = ACTION_SELL

        # Simple pnl estimator: small proportion of position
        if action == ACTION_BUY:  # buy
            position += 0.1
        elif action == ACTION_SELL:  # sell
            position -= 0.1
        position = max(-cfg.max_position_size, min(cfg.max_position_size, position))

        pnl = (price - prices[max(0, step - 1)]) * position
        old_position = position

        # Compute reward-like composite using behavior calculator and shaping
        # Base shaping: small exploration reward, then penalize balance and skew
        base_reward = 0.0
        if len(bpc.recent_actions) < bpc.lookback:
            base_reward = 2.0
        else:
            base_reward = 0.0

        balance_penalty = bpc.calculate_balance_penalty(action)
        shaping = bpc.calculate_balance_shaping(action)
        skew_penalty = bpc.calculate_skewness_penalty()
        entropy_shaping = bpc.calculate_action_entropy_shaping()
        emergency_penalty = bpc.calculate_emergency_intervention()

        reward = (
            base_reward
            + shaping
            + entropy_shaping
            + balance_penalty
            + skew_penalty
            + emergency_penalty
        )

        # Track
        reward_history.append(reward)
        portfolio_value += pnl
        portfolio_value_history.append(portfolio_value)

        # Check stage changes and emergency
        # Update curriculum manager with latest state
        status = cm.update(
            step=step,
            action_counts=bpc._get_recent_counts(),
            recent_rewards=[reward],
            portfolio_values=portfolio_value_history,
        )
        if status["emergency"]:
            logs["emergency_triggers"] += 1
        logs["stage_history"] = cm.get_stage_info()["stage_history"]

        if step % 50 == 0:
            print(
                f"Step {step}: action={action}, reward={reward:.3f}, stage={cm.get_current_stage()}, emergency={cm.emergency_count}"
            )

    print("Simulation complete")
    print("Stage history entries:", len(logs["stage_history"]))
    print("Emergency triggers:", logs["emergency_triggers"])


if __name__ == "__main__":
    simulate_steps(steps=300)
