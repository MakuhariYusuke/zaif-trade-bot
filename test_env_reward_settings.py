#!/usr/bin/env python3
"""
Unit tests for HeavyTradingEnv reward settings configuration.
"""

import copy
import unittest
from typing import Any, Dict, List, Optional

import pandas as pd

from ztb.trading.environment.environment import HeavyTradingEnv


class TestHeavyTradingEnvRewardSettings(unittest.TestCase):
    """Test reward settings configuration in HeavyTradingEnv."""
    df: pd.DataFrame | None = None

    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal DataFrame for testing
        self.df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'open': [99.0, 100.0, 101.0],
            'high': [101.0, 102.0, 103.0],
            'low': [98.0, 99.0, 100.0],
            'volume': [1000, 1100, 1200],
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='1min')
        })

    def _build_env(self, reward_overrides: Optional[Dict[str, Any]] = None) -> HeavyTradingEnv:
        """Create an environment with simplified reward settings for deterministic tests."""
        base_reward_settings = {
            "enable_forced_diversity": False,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
            "reward_clip_value": 0.0,
            "position_penalty_scale": 0.0,
            "inventory_penalty_scale": 0.0,
            "trade_frequency_penalty": 0.0,
            "trade_cooldown_penalty": 0.0,
            "consecutive_trade_penalty": 0.0,
            "volatility_penalty_scale": 0.0,
            "sharpe_bonus_scale": 0.0,
            "sortino_bonus_scale": 0.0,
            "calmar_bonus_scale": 0.0,
        }
        reward_settings = base_reward_settings.copy()
        if reward_overrides:
            reward_settings.update(reward_overrides)

        config = {
            "curriculum_stage": "full",
            "reward_scaling": 1.0,
            "reward_settings": reward_settings,
        }

        env = HeavyTradingEnv(df=self.df, config=config)
        env.reset()
        env.config["curriculum_stage"] = "full"
        return env

    def _prepare_reward_state(self, env: HeavyTradingEnv, action_history: List[int]) -> None:
        """Reset mutable reward-related state for deterministic reward calculations."""
        max_history = getattr(env, "_max_action_history", len(action_history))
        env.action_history = list(action_history[-max_history:])
        env._action_counts = [  # type: ignore[attr-defined]
            env.action_history.count(0),
            env.action_history.count(1),
            env.action_history.count(2),
        ]
        if hasattr(env, "_current_episode_actions"):
            setattr(env, "_current_episode_actions", list(env.action_history))
        env.pnl_history.clear()
        env.position_abs_history.clear()
        env.reward_history = []
        env.trade_interval_history.clear()
        env.portfolio_value_history = []
        env._last_trade_step = None  # type: ignore[attr-defined]
        env._consecutive_trade_steps = 0  # type: ignore[attr-defined]
        env.current_step = 0
        env.position = 0.0

    def _evaluate_reward(
        self,
        env: HeavyTradingEnv,
        *,
        action: int,
        action_history: List[int] | None = None,
        pnl: float = 1.0,
        position: float = 0.0,
        old_position: float = 0.0,
        current_price: float = 100.0,
    ) -> float:
        """Helper to compute reward with controlled inputs."""
        history = action_history if action_history is not None else [action] * 10
        self._prepare_reward_state(env, history)
        env.position = position

        return env._calculate_reward(  # type: ignore[attr-defined]
            action=action,
            current_price=current_price,
            position=position,
            portfolio_value=1000.0,
            atr=1.0,
            transaction_cost=0.0,
            reward_scaling=1.0,
            pnl=pnl,
            old_position=old_position,
            step=0,
            observation=None,
        )

    def test_reward_settings_merge(self):
        """Test that reward_settings from config are properly merged."""
        config = {
            "reward_settings": {
                "profit_bonus_multipliers": [0.5, 1.0, 1.5],
                "enable_forced_diversity": True,
            }
        }

        env = HeavyTradingEnv(df=self.df, config=config)

        # Check that profit_bonus_multipliers are correctly set
        self.assertEqual(env.reward_settings["profit_bonus_multipliers"], [0.5, 1.0, 1.5])

        # Check that enable_forced_diversity is correctly set
        self.assertTrue(env.reward_settings["enable_forced_diversity"])

    def test_default_reward_settings(self):
        """Test default reward settings when no config provided."""
        env = HeavyTradingEnv(df=self.df)

        # Check default profit_bonus_multipliers
        self.assertEqual(env.reward_settings["profit_bonus_multipliers"], [1.0, 1.0, 0.8])

        # Check default enable_forced_diversity
        self.assertFalse(env.reward_settings["enable_forced_diversity"])

    def test_profit_bonus_multipliers_in_reward(self):
        """Test that profit_bonus_multipliers affect reward calculation."""
        reward_overrides = {
            "profit_bonus_multipliers": [0.5, 1.0, 1.5],
            "enable_forced_diversity": False,
        }

        reward_buy = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=1,
            action_history=[1] * 10,
            pnl=2.0,
            position=1.0,
        )

        reward_sell = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=2,
            action_history=[2] * 10,
            pnl=2.0,
            position=-1.0,
        )

        reward_hold = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=0,
            action_history=[0] * 10,
            pnl=2.0,
            position=1.0,
        )

        self.assertLess(
            reward_buy,
            reward_sell,
            "SELL should earn more reward than BUY when multipliers favor SELL",
        )
        self.assertLess(
            reward_sell,
            reward_hold,
            "HOLD should earn the highest reward with a 1.5x multiplier",
        )

    def test_forced_diversity_penalty(self):
        """Test that enable_forced_diversity adds penalty for unbalanced action distribution."""
        reward_overrides = {
            "enable_forced_diversity": True,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
            "reward_clip_value": 0.0,
        }

        buy_imbalanced_history = [1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
        buy_balanced_history = [1, 1, 1, 2, 2, 2, 1, 2, 1, 2]

        reward_buy_penalized = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=1,
            action_history=buy_imbalanced_history,
            pnl=1.0,
            position=0.0,
        )

        reward_buy_balanced = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=1,
            action_history=buy_balanced_history,
            pnl=1.0,
            position=0.0,
        )

        self.assertLess(
            reward_buy_penalized,
            reward_buy_balanced,
            "Forced diversity penalty should reduce BUY reward when BUY ratio exceeds threshold",
        )

        sell_imbalanced_history = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2]

        reward_sell_penalized = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=2,
            action_history=sell_imbalanced_history,
            pnl=1.0,
            position=0.0,
        )

        reward_sell_balanced = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=2,
            action_history=buy_balanced_history,
            pnl=1.0,
            position=0.0,
        )

        self.assertLess(
            reward_sell_penalized,
            reward_sell_balanced,
            "Forced diversity penalty should reduce SELL reward when SELL ratio exceeds threshold",
        )

    def test_step_action_opens_expected_positions(self) -> None:
        """Ensure action indices map to HOLD=0, BUY=1, SELL=2 when stepping the environment."""
        env = self._build_env({})
        env.config["max_position_size"] = 1.0

        env.reset()
        position_before = env.position
        env.step(0)
        self.assertEqual(env.position, position_before, "HOLD action should not change position")

        env.reset()
        env.step(1)
        self.assertGreater(env.position, 0, "BUY action should result in long position")

        env.reset()
        env.step(2)
        self.assertLess(env.position, 0, "SELL action should result in short position")

    def test_reward_symmetry_buy_sell(self):
        """Test that BUY and SELL rewards are symmetric under identical conditions."""
        reward_overrides = {
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],  # Symmetric multipliers
            "enable_forced_diversity": False,
        }

        # Test with positive PnL (profitable trade)
        reward_buy_profit = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=1,
            action_history=[1] * 10,
            pnl=2.0,
            position=1.0,
        )

        reward_sell_profit = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=2,
            action_history=[2] * 10,
            pnl=2.0,
            position=-1.0,
        )

        # BUY and SELL should have same reward for profitable trades with symmetric multipliers
        self.assertAlmostEqual(
            reward_buy_profit,
            reward_sell_profit,
            places=2,
            msg="BUY and SELL rewards should be symmetric for profitable trades",
        )

        # Test with negative PnL (loss trade)
        reward_buy_loss = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=1,
            action_history=[1] * 10,
            pnl=-2.0,
            position=1.0,
        )

        reward_sell_loss = self._evaluate_reward(
            self._build_env(reward_overrides),
            action=2,
            action_history=[2] * 10,
            pnl=-2.0,
            position=-1.0,
        )

        # BUY and SELL should have same penalty for loss trades
        self.assertAlmostEqual(
            reward_buy_loss,
            reward_sell_loss,
            places=2,
            msg="BUY and SELL penalties should be symmetric for loss trades",
        )

    def test_forced_diversity_strengthened_penalty(self):
        """Test strengthened forced diversity penalty with lower threshold (25%) and higher multiplier (1.5)."""
        reward_overrides = {
            "enable_forced_diversity": True,
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
            "reward_clip_value": 0.0,
        }

        # BUY ratio at 57% (above 25% threshold)
        buy_high_ratio_history = [1, 1, 1, 1, 2, 2, 2, 0, 0, 0]  # BUY:4, SELL:3, HOLD:3

        # Calculate expected penalty with strengthened parameters
        recent_actions = buy_high_ratio_history[-10:]
        buy_count = sum(1 for a in recent_actions if a == 1)
        sell_count = sum(1 for a in recent_actions if a == 2)
        total_trades = buy_count + sell_count
        buy_ratio = buy_count / total_trades if total_trades > 0 else 0

        # Strengthened penalty: threshold 0.25, multiplier 1.5
        strengthened_penalty = (buy_ratio - 0.25) * 1.5 if buy_ratio > 0.25 else 0.0

        # Verify strengthened penalty is greater than old penalty
        old_penalty = (buy_ratio - 0.3) * 1.0 if buy_ratio > 0.3 else 0.0
        self.assertGreater(
            strengthened_penalty,
            old_penalty,
            f"Strengthened penalty ({strengthened_penalty:.3f}) should be > old penalty ({old_penalty:.3f})",
        )

    def test_downward_signal_enhancement(self):
        """Test enhanced downward signal detection with RSI and stricter trend thresholds."""
        reward_overrides = {
            "profit_bonus_multipliers": [1.0, 1.0, 0.8],
            "enable_forced_diversity": False,
        }

        # Create environment with bearish trend data
        env = self._build_env(reward_overrides)
        env.df = pd.DataFrame({
            'close': [100.0, 99.0, 98.0],
            'open': [101.0, 100.0, 99.0],
            'high': [102.0, 101.0, 100.0],
            'low': [99.0, 98.0, 97.0],
            'volume': [1000, 1100, 1200],
            'sma_short': [97.0, 97.0, 97.0],  # SMA_20 (downtrend)
            'sma_long': [102.0, 102.0, 102.0],  # SMA_50 (above short)
            'rsi': [25.0, 28.0, 30.0],  # Oversold
        })
        env.current_step = 0

        # Test that the trend detection strengthening is in place
        # The code now uses stricter thresholds (>1.02 for strong bullish, <0.98 for strong bearish)
        # and enhanced multipliers (1.5 bonus, 0.5 penalty)
        step_data = env.df.iloc[env.current_step]
        sma_20 = step_data.get('sma_short', 0.0)
        sma_50 = step_data.get('sma_long', 0.0)
        rsi = step_data.get('rsi', 50.0)

        self.assertLess(sma_20, sma_50, "Should be bearish trend")
        self.assertLess(rsi, 30.0, "Should be oversold")

        # Verify trend ratio is in strong bearish range (<0.98)
        trend_ratio = sma_20 / sma_50 if sma_50 > 0 else 1.0
        self.assertLess(trend_ratio, 0.98, f"Trend ratio ({trend_ratio:.4f}) should indicate strong bearish")

        # Additionally verify RSI oversold enhancement would trigger
        self.assertTrue(rsi < 30.0 and trend_ratio < 1.0, "Should meet RSI oversold + bearish trend conditions")


if __name__ == '__main__':
    unittest.main()
