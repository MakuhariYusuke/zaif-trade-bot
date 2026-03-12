"""
SAC v397c_fixed_scale Deep Dive Analysis
Collects per-step reward component contributions, action dynamics, and cost drivers.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.environment.environment import HeavyTradingEnv


def instrument_reward_calculator() -> None:
    """Injects tracking into RewardCalculator.calculate_reward_simple."""
    original_method = RewardCalculator.calculate_reward_simple

    def wrapped(
        self: RewardCalculator,
        pnl: float,
        portfolio_value: float,
        position: float,
        old_position: float,
        action: int = 0,
    ) -> float:
        reward_scale = self.get_setting_float("reward_scale", 1000.0)
        reward_clip_min = self.get_setting_float("reward_clip_min", -10.0)
        reward_clip_max = self.get_setting_float("reward_clip_max", 10.0)
        inactivity_enabled = self.get_setting_bool("enable_inactivity_penalty", False)
        opportunity_enabled = self.get_setting_bool("enable_opportunity_cost", False)
        bonus_enabled = self.get_setting_bool("enable_trade_execution_bonus", False)

        pnl_ratio = pnl / max(portfolio_value, 1.0)
        base_unclipped = pnl_ratio * reward_scale
        base_clipped = max(reward_clip_min, min(reward_clip_max, base_unclipped))

        total_reward = base_clipped
        inactivity_penalty = 0.0
        opportunity_penalty = 0.0
        trade_bonus = 0.0

        max_position_size = max(abs(self.config.max_position_size), 1e-8)

        if inactivity_enabled:
            hold_threshold = self.get_setting_float("inactivity_hold_threshold", 0.0)
            hold_threshold_abs = hold_threshold * max_position_size
            if (
                action == 0
                and abs(position) <= hold_threshold_abs
                and abs(old_position) <= hold_threshold_abs
            ):
                self._consecutive_idle_steps += 1
                rate = self.get_setting_float("inactivity_penalty_rate", 0.001)
                window = max(1, self.get_setting_int("inactivity_penalty_window", 5))
                multiplier = min(self._consecutive_idle_steps, window)
                penalty = rate * multiplier
                total_reward -= penalty
                inactivity_penalty = penalty
            else:
                self._consecutive_idle_steps = 0

        if opportunity_enabled:
            hold_threshold = self.get_setting_float("position_hold_threshold", 0.05)
            hold_threshold_abs = hold_threshold * max_position_size
            if action == 0 and abs(position) > hold_threshold_abs:
                self._consecutive_position_hold_steps += 1
                rate = self.get_setting_float("opportunity_cost_rate", 0.0005)
                window = max(1, self.get_setting_int("opportunity_cost_window", 5))
                multiplier = min(self._consecutive_position_hold_steps, window)
                intensity = min(abs(position) / max_position_size, 1.0)
                penalty = rate * multiplier * intensity
                total_reward -= penalty
                opportunity_penalty = penalty
            else:
                self._consecutive_position_hold_steps = 0

        if bonus_enabled:
            threshold = self.get_setting_float(
                "trade_execution_position_threshold", 0.01
            )
            threshold_abs = max(threshold * max_position_size, 1e-8)
            position_change = abs(position - old_position)
            if position_change > threshold_abs:
                rate = self.get_setting_float("trade_execution_bonus_rate", 0.001)
                multiplier = self.get_setting_float(
                    "trade_execution_action_multiplier", 1.0
                )
                intensity = min(position_change / max_position_size, 1.0)
                applied_multiplier = multiplier if action in (1, 2) else 1.0
                bonus = rate * intensity * applied_multiplier
                total_reward += bonus
                trade_bonus = bonus

        setattr(
            self,
            "_debug_components",
            {
                "pnl_component": base_clipped,
                "pnl_unclipped": base_unclipped,
                "inactivity_penalty": -inactivity_penalty,
                "opportunity_penalty": -opportunity_penalty,
                "trade_bonus": trade_bonus,
                "total_reward": total_reward,
            },
        )

        return total_reward

    RewardCalculator.calculate_reward_simple = wrapped  # type: ignore[assignment]


def run_deep_analysis(
    model_path: Path,
    data_path: Path,
    max_steps: int = 5000,
) -> Dict[str, Any]:
    instrument_reward_calculator()

    df = pd.read_csv(data_path)
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)

    config: Dict[str, Any] = {
        "initial_portfolio_value": 200000.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "allow_reverse": True,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.25,
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 1000.0,
            "reward_clip_min": -10.0,
            "reward_clip_max": 10.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.01,
            "inactivity_penalty_window": 3,
            "inactivity_hold_threshold": 0.05,
            "enable_opportunity_cost": True,
            "opportunity_cost_rate": 0.005,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.05,
            "trade_execution_position_threshold": 0.01,
            "trade_execution_action_multiplier": 1.5,
        },
    }

    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    model = SAC.load(str(model_path))

    obs, _ = env.reset()

    records: List[Dict[str, Any]] = []

    step = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            continuous_value = float(action.item())
        else:
            continuous_value = float(action)

        old_position = env.position
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        discrete_action = continuous_to_discrete_action(
            continuous_value,
            threshold=getattr(env, "action_threshold", 0.33),
        )
        pnl = info.get("pnl", 0.0)
        portfolio_value = info.get("portfolio_value", env.portfolio_value)

        components = getattr(env.reward_calculator, "_debug_components", {})

        records.append(
            {
                "step": step,
                "continuous_action": continuous_value,
                "discrete_action": discrete_action,
                "reward": reward,
                "pnl": pnl,
                "portfolio_value": portfolio_value,
                "position": env.position,
                "position_change": env.position - old_position,
                "pnl_component": components.get("pnl_component", 0.0),
                "pnl_unclipped": components.get("pnl_unclipped", 0.0),
                "inactivity_penalty": components.get("inactivity_penalty", 0.0),
                "opportunity_penalty": components.get("opportunity_penalty", 0.0),
                "trade_bonus": components.get("trade_bonus", 0.0),
                "total_reward": components.get("total_reward", reward),
            }
        )

        step += 1
        if step % 1000 == 0:
            print(f"Processed {step} steps")

    df_records = pd.DataFrame(records)

    reward_sum = df_records["reward"].sum()

    component_totals = {
        "pnl_component_sum": float(df_records["pnl_component"].sum()),
        "inactivity_penalty_sum": float(df_records["inactivity_penalty"].sum()),
        "opportunity_penalty_sum": float(df_records["opportunity_penalty"].sum()),
        "trade_bonus_sum": float(df_records["trade_bonus"].sum()),
    }

    action_distribution = (
        df_records["discrete_action"]
        .value_counts(normalize=True)
        .reindex([0, 1, 2], fill_value=0.0)
        .to_dict()
    )

    summary: Dict[str, Any] = {
        "reward_totals": component_totals,
        "reward_total_sum": float(reward_sum),
        "reward_component_ratios": {
            name: float(value / reward_sum * 100.0) if reward_sum else 0.0
            for name, value in component_totals.items()
        },
        "action_distribution": {
            "HOLD": action_distribution.get(0, 0.0) * 100.0,
            "BUY": action_distribution.get(1, 0.0) * 100.0,
            "SELL": action_distribution.get(2, 0.0) * 100.0,
        },
        "continuous_action_stats": {
            "mean": float(df_records["continuous_action"].mean()),
            "std": float(df_records["continuous_action"].std()),
            "min": float(df_records["continuous_action"].min()),
            "max": float(df_records["continuous_action"].max()),
        },
        "pnl_stats": {
            "mean": float(df_records["pnl"].mean()),
            "std": float(df_records["pnl"].std()),
            "min": float(df_records["pnl"].min()),
            "max": float(df_records["pnl"].max()),
            "total": float(df_records["pnl"].sum()),
        },
        "position_stats": {
            "mean": float(df_records["position"].mean()),
            "std": float(df_records["position"].std()),
            "min": float(df_records["position"].min()),
            "max": float(df_records["position"].max()),
        },
        "position_change_stats": {
            "mean": float(df_records["position_change"].abs().mean()),
            "trades_over_threshold": int((df_records["trade_bonus"] > 0).sum()),
        },
        "max_drawdown_reward": float(df_records["reward"].cumsum().min()),
    }

    output_dir = project_root / "docs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v397c_deep_analysis.json"
    df_records.to_csv(output_dir / "v397c_deep_analysis_steps.csv", index=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis summary saved to {output_path}")
    return summary


if __name__ == "__main__":
    model_path = (
        project_root / "checkpoints" / "sac_session" / "sac_v397c_fixed_scale_final.zip"
    )
    data_path = project_root / "btc_jpy_real_dataset.csv"
    summary = run_deep_analysis(model_path, data_path, max_steps=5000)
    print(json.dumps(summary, indent=2))
