"""
Balance Curriculum Manager - Dynamic curriculum progression for bias-free training.

SAC v448 Layer 3: Integrates with existing curriculum_stage system while adding
dynamic progression and emergency intervention capabilities.
"""

from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np

from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger


class BalanceCurriculumManager:
    """
    Manages dynamic curriculum progression for balance-focused training.
    
    Features:
    - Automatic stage progression based on performance metrics
    - Emergency revert to forced_balance on bias collapse
    - Integration with existing RewardCalculator curriculum_stage system
    - Backward compatible (can be disabled for v447-style static stages)
    
    Stage Progression:
    1. forced_balance: Enforce action balance until stable
    2. balanced_transition: Gradual introduction of profit-based rewards
    3. pnl_focused: Optimize for profitability while maintaining balance
    
    Emergency Conditions:
    - BUY-SELL deviation > 35%: Revert to forced_balance
    - Sustained negative rewards: Increase balance enforcement
    """

    # Define stage progression order
    STAGE_SEQUENCE = [
        "forced_balance",
        "balanced_transition",
        "pnl_focused",
        "trading_focused",
        "profit_optimized",
    ]

    def __init__(
        self,
        config: EnvironmentConfig,
        enabled: bool = True,
        auto_progression: bool = True,
        emergency_revert: bool = True,
    ):
        """
        Initialize BalanceCurriculumManager.
        
        Args:
            config: Environment configuration
            enabled: Enable dynamic curriculum (False for v447 compatibility)
            auto_progression: Enable automatic stage progression
            emergency_revert: Enable emergency revert to forced_balance
        """
        self.config = config
        self.enabled = enabled
        self.auto_progression = auto_progression
        self.emergency_revert = emergency_revert
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize current stage from config
        self.current_stage = getattr(config, "curriculum_stage", "forced_balance")
        self.stage_start_step = 0
        self.total_steps = 0
        
        # Stage history for analysis
        self.stage_history: List[Dict[str, Any]] = []
        
        # Metrics tracking
        self.recent_rewards = deque(maxlen=100)
        self.stage_rewards = deque(maxlen=50)
        
        # Stage progression conditions (can be overridden by config)
        self.stage_conditions = self._initialize_stage_conditions()
        
        # Emergency state
        self.emergency_count = 0
        self.max_emergency_reverts = 3
        
        self.logger.info(
            f"BalanceCurriculumManager initialized: enabled={enabled}, "
            f"auto_progression={auto_progression}, initial_stage={self.current_stage}"
        )
    
    def _initialize_stage_conditions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default stage progression conditions."""
        return {
            "forced_balance": {
                "min_steps": 100,
                "balance_threshold": 0.15,  # BUY-SELL diff < 15%
                "min_success_episodes": 10,  # At least 10 balanced episodes
                "success_rate": 0.8,  # 80% of recent episodes balanced
            },
            "balanced_transition": {
                "min_steps": 200,
                "balance_threshold": 0.20,  # Allow slightly more deviation
                "avg_reward_threshold": 0.0,  # Positive average reward
                "sharpe_threshold": 0.0,  # Non-negative Sharpe
            },
            "pnl_focused": {
                "min_steps": 500,
                "balance_threshold": 0.25,  # More flexibility
                "avg_reward_threshold": 2.0,  # Consistent profitability
                "sharpe_threshold": 0.5,
                "max_drawdown": 0.20,
            },
            "trading_focused": {
                "min_steps": 500,
                "avg_reward_threshold": 5.0,
                "sharpe_threshold": 0.8,
            },
        }
    
    def update(
        self,
        step: int,
        action_counts: List[int],
        recent_rewards: List[float],
        portfolio_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Update curriculum state and check for stage progression or emergency.
        
        Args:
            step: Current training step
            action_counts: [HOLD, BUY, SELL] action counts
            recent_rewards: Recent episode rewards
            portfolio_values: Portfolio value history (optional)
        
        Returns:
            Dictionary with curriculum status and any changes:
            {
                "stage": str,
                "changed": bool,
                "previous_stage": Optional[str],
                "emergency": bool,
                "steps_in_stage": int,
            }
        """
        if not self.enabled:
            return {"stage": self.current_stage, "changed": False, "emergency": False}
        
        self.total_steps = step
        previous_stage = self.current_stage
        stage_changed = False
        emergency_triggered = False
        
        # Update metrics
        if recent_rewards:
            for reward in recent_rewards:
                self.recent_rewards.append(reward)
                self.stage_rewards.append(reward)
        
        # 1. Check emergency conditions first
        if self.emergency_revert and self.emergency_count < self.max_emergency_reverts:
            emergency_triggered = self._check_emergency(action_counts)
            if emergency_triggered:
                self.emergency_count += 1
                stage_changed = True
                self.logger.warning(
                    f"🚨 Emergency {self.emergency_count}/{self.max_emergency_reverts}: "
                    f"Reverted to forced_balance"
                )
        
        # 2. Check stage progression (only if no emergency and auto_progression enabled)
        if (
            not emergency_triggered
            and self.auto_progression
            and self._should_progress(step, action_counts, portfolio_values)
        ):
            next_stage = self._get_next_stage()
            if next_stage and next_stage != self.current_stage:
                self._progress_to_stage(next_stage, step)
                stage_changed = True
        
        # 3. Compile status
        status = {
            "stage": self.current_stage,
            "changed": stage_changed,
            "previous_stage": previous_stage if stage_changed else None,
            "emergency": emergency_triggered,
            "steps_in_stage": step - self.stage_start_step,
            "total_steps": self.total_steps,
            "emergency_count": self.emergency_count,
        }
        
        if stage_changed:
            self.logger.info(
                f"Stage transition: {previous_stage} -> {self.current_stage} "
                f"(step {step}, emergency={emergency_triggered})"
            )
        
        return status
    
    def _check_emergency(self, action_counts: List[int]) -> bool:
        """
        Check if emergency conditions require reverting to forced_balance.
        
        Emergency triggers:
        - BUY-SELL deviation > 35%
        - Sustained negative rewards with bias
        
        Returns:
            True if emergency revert occurred
        """
        if self.current_stage == "forced_balance":
            return False  # Already at safest stage
        
        total_actions = sum(action_counts)
        if total_actions < 50:
            return False  # Not enough data
        
        buy_ratio = action_counts[1] / total_actions
        sell_ratio = action_counts[2] / total_actions
        buy_sell_diff = abs(buy_ratio - sell_ratio)
        
        # Emergency condition: extreme bias
        if buy_sell_diff > 0.35:
            self.logger.warning(
                f"🚨 BALANCE EMERGENCY: BUY-SELL diff={buy_sell_diff:.1%} "
                f"(BUY={buy_ratio:.1%}, SELL={sell_ratio:.1%}), reverting to forced_balance"
            )
            self._revert_to_forced_balance()
            return True
        
        # Secondary emergency: sustained negative rewards with moderate bias
        if (
            len(self.recent_rewards) >= 20
            and np.mean(list(self.recent_rewards)[-20:]) < -2.0
            and buy_sell_diff > 0.25
        ):
            self.logger.warning(
                f"🚨 PERFORMANCE EMERGENCY: Avg reward={np.mean(list(self.recent_rewards)[-20:]):.2f}, "
                f"BUY-SELL diff={buy_sell_diff:.1%}, reverting to forced_balance"
            )
            self._revert_to_forced_balance()
            return True
        
        return False
    
    def _should_progress(
        self,
        step: int,
        action_counts: List[int],
        portfolio_values: Optional[List[float]],
    ) -> bool:
        """
        Check if current stage conditions are met for progression.
        
        Returns:
            True if ready to progress to next stage
        """
        steps_in_stage = step - self.stage_start_step
        conditions = self.stage_conditions.get(self.current_stage)
        
        if not conditions:
            return False  # No progression rules for current stage
        
        # Minimum steps requirement
        if steps_in_stage < conditions.get("min_steps", 100):
            return False
        
        # Stage-specific checks
        if self.current_stage == "forced_balance":
            return self._check_forced_balance_completion(action_counts)
        
        elif self.current_stage == "balanced_transition":
            return self._check_balanced_transition_completion(action_counts)
        
        elif self.current_stage == "pnl_focused":
            return self._check_pnl_focused_completion(action_counts, portfolio_values)
        
        return False
    
    def _check_forced_balance_completion(self, action_counts: List[int]) -> bool:
        """Check if forced_balance stage is ready to progress."""
        conditions = self.stage_conditions["forced_balance"]
        total_actions = sum(action_counts)
        
        if total_actions < 100:
            return False
        
        buy_ratio = action_counts[1] / total_actions
        sell_ratio = action_counts[2] / total_actions
        buy_sell_diff = abs(buy_ratio - sell_ratio)
        
        # Check balance threshold
        balance_met = buy_sell_diff < conditions["balance_threshold"]
        
        # Check sustained balance (using recent stage rewards as proxy)
        recent_stage_rewards = list(self.stage_rewards)[-20:] if len(self.stage_rewards) >= 20 else []
        sustained_positive = (
            len(recent_stage_rewards) >= 10
            and np.mean(recent_stage_rewards) > 0.0
        )
        
        if balance_met and sustained_positive:
            self.logger.info(
                f"Forced balance conditions met: BUY-SELL diff={buy_sell_diff:.1%}, "
                f"avg_reward={np.mean(recent_stage_rewards):.2f}"
            )
            return True
        
        return False
    
    def _check_balanced_transition_completion(self, action_counts: List[int]) -> bool:
        """Check if balanced_transition stage is ready to progress."""
        conditions = self.stage_conditions["balanced_transition"]
        
        if len(self.stage_rewards) < 30:
            return False
        
        avg_reward = np.mean(list(self.stage_rewards)[-30:])
        
        # Check average reward threshold
        reward_met = avg_reward >= conditions.get("avg_reward_threshold", 0.0)
        
        # Check balance (more lenient than forced_balance)
        total_actions = sum(action_counts)
        if total_actions >= 100:
            buy_sell_diff = abs(action_counts[1] / total_actions - action_counts[2] / total_actions)
            balance_ok = buy_sell_diff < conditions.get("balance_threshold", 0.20)
        else:
            balance_ok = True  # Not enough data to judge
        
        if reward_met and balance_ok:
            self.logger.info(
                f"Balanced transition conditions met: avg_reward={avg_reward:.2f}"
            )
            return True
        
        return False
    
    def _check_pnl_focused_completion(
        self,
        action_counts: List[int],
        portfolio_values: Optional[List[float]],
    ) -> bool:
        """Check if pnl_focused stage is ready to progress."""
        conditions = self.stage_conditions["pnl_focused"]
        
        if len(self.stage_rewards) < 50:
            return False
        
        avg_reward = np.mean(list(self.stage_rewards)[-50:])
        reward_met = avg_reward >= conditions.get("avg_reward_threshold", 2.0)
        
        # Calculate Sharpe ratio if enough data
        sharpe_ok = True
        if len(self.stage_rewards) >= 30:
            rewards_array = np.array(list(self.stage_rewards)[-30:])
            if np.std(rewards_array) > 0:
                sharpe = np.mean(rewards_array) / np.std(rewards_array)
                sharpe_ok = sharpe >= conditions.get("sharpe_threshold", 0.5)
        
        if reward_met and sharpe_ok:
            self.logger.info(
                f"PnL focused conditions met: avg_reward={avg_reward:.2f}"
            )
            return True
        
        return False
    
    def _get_next_stage(self) -> Optional[str]:
        """Get the next stage in progression sequence."""
        try:
            current_idx = self.STAGE_SEQUENCE.index(self.current_stage)
            if current_idx < len(self.STAGE_SEQUENCE) - 1:
                return self.STAGE_SEQUENCE[current_idx + 1]
        except ValueError:
            # Current stage not in sequence
            pass
        return None
    
    def _progress_to_stage(self, next_stage: str, step: int):
        """Progress to the specified stage."""
        previous_stage = self.current_stage
        
        # Record stage history
        self.stage_history.append({
            "stage": previous_stage,
            "start_step": self.stage_start_step,
            "end_step": step,
            "duration": step - self.stage_start_step,
            "avg_reward": np.mean(list(self.stage_rewards)) if self.stage_rewards else 0.0,
        })
        
        # Update stage
        self.current_stage = next_stage
        self.stage_start_step = step
        self.stage_rewards.clear()
        
        self.logger.info(
            f"✨ Progressed from {previous_stage} to {next_stage} at step {step}"
        )
    
    def _revert_to_forced_balance(self):
        """Emergency revert to forced_balance stage."""
        if self.current_stage != "forced_balance":
            self.current_stage = "forced_balance"
            self.stage_start_step = self.total_steps
            self.stage_rewards.clear()
    
    def get_current_stage(self) -> str:
        """
        Get current curriculum stage for RewardCalculator.
        
        Returns:
            Current stage name (e.g., "forced_balance")
        """
        return self.current_stage
    
    def get_stage_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current curriculum state.
        
        Returns:
            Dictionary with stage info, metrics, and history
        """
        return {
            "current_stage": self.current_stage,
            "steps_in_stage": self.total_steps - self.stage_start_step,
            "total_steps": self.total_steps,
            "emergency_count": self.emergency_count,
            "stage_history": self.stage_history,
            "recent_avg_reward": (
                np.mean(list(self.recent_rewards)[-20:])
                if len(self.recent_rewards) >= 20
                else None
            ),
            "enabled": self.enabled,
            "auto_progression": self.auto_progression,
        }
    
    def reset(self):
        """Reset curriculum state for new episode/training run."""
        self.recent_rewards.clear()
        self.stage_rewards.clear()
        # Note: Don't reset current_stage to preserve learning progress
