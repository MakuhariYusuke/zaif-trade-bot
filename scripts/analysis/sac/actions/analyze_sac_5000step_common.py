#!/usr/bin/env python3
"""Shared action-distribution analysis for 5000-step SAC experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

# Add project root to path
project_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(project_root))

from ztb.utils.constants import DEFAULT_SEED


@dataclass(frozen=True)
class ActionAnalysisProfile:
    title: str
    model_path: str
    output_path: str
    threshold: float
    no_action_penalty: float = 0.0
    action_bonus: float = 0.0
    n_episodes: int = 10
    sell_ratio_warn: float = 0.6
    buy_ratio_warn: float = 0.1
    std_warn: float = 0.1
    reward_warn: float = -100.0
    hold_ratio_warn: float | None = None
    success_criteria: tuple[str, ...] = ()


def _create_env(profile: ActionAnalysisProfile) -> gym.Env:
    np.random.seed(DEFAULT_SEED)
    n_steps = 1000
    base_price = 1000000
    price_changes = np.random.normal(0, 0.001, n_steps)
    prices = base_price * (1 + np.cumsum(price_changes))

    class TradingEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
            self.prices = prices
            self.initial_balance = 200000.0
            self.reset()

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0
            self.prev_portfolio_value = self.initial_balance
            return self._get_observation(), {}

        def step(self, action):
            action_value = float(action[0])
            price = self.prices[self.current_step]
            trade_executed = False

            if action_value > profile.threshold and self.balance > price * 0.1:
                self.position += 0.1
                self.balance -= price * 0.1
                trade_executed = True
            elif action_value < -profile.threshold and self.position > 0.1:
                self.position -= 0.1
                self.balance += price * 0.1
                trade_executed = True

            portfolio_value = self.balance + self.position * price
            portfolio_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
            reward = portfolio_return
            if not trade_executed:
                reward += profile.no_action_penalty
            if trade_executed:
                reward += profile.action_bonus

            self.prev_portfolio_value = portfolio_value
            self.current_step += 1
            done = self.current_step >= len(self.prices) - 1
            return self._get_observation(), reward, done, False, {}

        def _get_observation(self):
            price = self.prices[self.current_step]
            trend = 0.0
            if self.current_step > 0:
                prev_price = self.prices[self.current_step - 1]
                trend = (price - prev_price) / prev_price

            return np.array(
                [
                    price / 10000000,
                    trend,
                    self.position,
                    self.balance / self.initial_balance,
                    self.current_step / len(self.prices),
                ],
                dtype=np.float32,
            )

    return TradingEnv()


def analyze_action_distribution(profile: ActionAnalysisProfile):
    model = SAC.load(profile.model_path)
    env = _create_env(profile)

    actions: list[float] = []
    rewards: list[float] = []
    observations: list[np.ndarray] = []

    for _ in range(profile.n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(float(action[0]))
            observations.append(obs)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(float(reward))

    actions_np = np.array(actions)
    rewards_np = np.array(rewards)
    observations_np = np.array(observations)

    buy_signals = int(np.sum(actions_np > profile.threshold))
    sell_signals = int(np.sum(actions_np < -profile.threshold))
    hold_signals = int(np.sum((actions_np >= -profile.threshold) & (actions_np <= profile.threshold)))

    analysis = {
        "total_actions": len(actions_np),
        "action_stats": {
            "mean": float(np.mean(actions_np)),
            "std": float(np.std(actions_np)),
            "min": float(np.min(actions_np)),
            "max": float(np.max(actions_np)),
            "median": float(np.median(actions_np)),
        },
        "action_distribution": {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "strong_buy": int(np.sum(actions_np > 0.5)),
            "strong_sell": int(np.sum(actions_np < -0.5)),
        },
        "reward_stats": {
            "mean": float(np.mean(rewards_np)),
            "std": float(np.std(rewards_np)),
            "total": float(np.sum(rewards_np)),
        },
        "bias_analysis": {
            "buy_ratio": float(buy_signals / len(actions_np)) if len(actions_np) > 0 else 0.0,
            "sell_ratio": float(sell_signals / len(actions_np)) if len(actions_np) > 0 else 0.0,
            "buy_sell_ratio": float(buy_signals / max(sell_signals, 1)),
        },
    }

    return analysis, actions_np, rewards_np, observations_np


def _detect_issues(profile: ActionAnalysisProfile, analysis: dict) -> list[str]:
    issues: list[str] = []
    if analysis["bias_analysis"]["sell_ratio"] > profile.sell_ratio_warn:
        issues.append(f"SELL ratio is high: > {profile.sell_ratio_warn:.2f}")
    if analysis["bias_analysis"]["buy_ratio"] < profile.buy_ratio_warn:
        issues.append(f"BUY ratio is low: < {profile.buy_ratio_warn:.2f}")
    if analysis["action_stats"]["std"] < profile.std_warn:
        issues.append(f"Action std is low: < {profile.std_warn:.2f}")
    if analysis["reward_stats"]["mean"] < profile.reward_warn:
        issues.append(f"Mean reward is low: < {profile.reward_warn}")
    if profile.hold_ratio_warn is not None:
        hold_ratio = analysis["action_distribution"]["hold_signals"] / max(analysis["total_actions"], 1)
        if hold_ratio > profile.hold_ratio_warn:
            issues.append(f"Hold ratio is high: > {profile.hold_ratio_warn:.2f}")
    return issues


def _print_success_criteria(lines: Iterable[str]) -> None:
    lines = tuple(lines)
    if not lines:
        return
    print("\nSuccess criteria:")
    for line in lines:
        print(f"  - {line}")


def run_profile(profile: ActionAnalysisProfile) -> None:
    if not Path(profile.model_path).exists():
        print(f"Model not found: {profile.model_path}")
        return

    analysis, _, _, _ = analyze_action_distribution(profile)

    output_path = Path(profile.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(profile.title)
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total Actions: {analysis['total_actions']}")
    print(f"Mean Action: {analysis['action_stats']['mean']:.6f}")
    print(f"Action Std: {analysis['action_stats']['std']:.6f}")
    print(f"Buy/Sell/Hold: {analysis['action_distribution']['buy_signals']}/"
          f"{analysis['action_distribution']['sell_signals']}/"
          f"{analysis['action_distribution']['hold_signals']}")
    print(f"Mean Reward: {analysis['reward_stats']['mean']:.6f}")

    issues = _detect_issues(profile, analysis)
    if issues:
        print("\nDetected issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo major issues detected.")

    _print_success_criteria(profile.success_criteria)
    print("=" * 60)
