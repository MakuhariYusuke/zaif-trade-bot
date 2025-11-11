"""
Reward Calculator - Handles reward calculation logic for trading environment.

This module separates the complex reward calculation logic from the main environment class.
Refactored to follow SOLID principles with component-based architecture.
"""

# mypy: disable-error-code=literal-required

import inspect
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_HOLD,
    MULTIPLIER_INDEX_SELL,
)
from ztb.trading.environment.constants import EPSILON
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.trading.strategies.action_signal_guide.components.market_regime import (
    MarketRegimeDetector,
)
from ztb.utils.logging_utils import StructuredLogger

from .asymmetric_reward_scaler import AsymmetricRewardScaler
from .behavioral_penalty_calculator import BehavioralPenaltyCalculator
from .dynamic_reward_shaper import DynamicRewardShaper
from .reward.opportunity_cost_penalty_calculator import OpportunityCostPenaltyCalculator
from .reward.unrealized_loss_penalty_calculator import UnrealizedLossPenaltyCalculator
from .signal_integrator import SignalIntegrator


class RewardCalculator:
    """
    Calculates rewards for trading actions with curriculum learning stages.

    This class orchestrates reward calculation using specialized components:
    - MarketRegimeDetector: Detects market regimes
    - DynamicRewardShaper: Applies market-aware reward shaping
    - SignalIntegrator: Integrates signal-based rewards
    - AsymmetricRewardScaler: Applies position-based scaling

    Follows SOLID principles for maintainability and testability.
    """

    ACTION_INDEX_NAMES = ["HOLD", "BUY", "SELL"]

    def __init__(
        self,
        config: EnvironmentConfig,
        reward_settings: RewardSettings,
        initial_portfolio_value: float,
    ):
        """
        Initialize RewardCalculator with component-based architecture.

        Args:
            config: Environment configuration
            reward_settings: Dictionary of reward settings
            initial_portfolio_value: Initial portfolio value
        """
        self.config = config
        self.reward_settings = reward_settings
        self.initial_portfolio_value = initial_portfolio_value
        self.logger = get_logger("ztb.trading.environment.reward")
        self.structured_logger = StructuredLogger("ztb.trading.environment.reward", json_format=True)

        # Internal state for tracking
        self._action_counts: List[int] = [0, 0, 0]  # [HOLD, BUY, SELL]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self.last_signal_strength: float = 0.0
        self.last_signal_reward: float = 0.0
        self._previous_portfolio_value = initial_portfolio_value
        self._last_reward_components: Dict[str, Union[str, float]] = {}
        self._recent_actions = []  # Reset this list as well

        # Initialize components
        self._initialize_components(config)

        self.logger.info(
            f"Dynamic reward shaping enabled: {self.dynamic_reward_shaper.enabled}"
        )
        if self.dynamic_reward_shaper.enabled:
            self.logger.info(
                f"Market regime awareness: {self.dynamic_reward_shaper.market_regime_awareness}"
            )
            self.logger.info(
                f"Volatility adjusted rewards: {self.dynamic_reward_shaper.volatility_adjusted_rewards}"
            )
            self.logger.info(
                f"Trend strength bonus: {self.dynamic_reward_shaper.trend_strength_bonus}"
            )
        self.logger.info(f"Signal guide enabled: {self.signal_integrator.enabled}")

        # Behavior optimization parameters
        self.action_balance_target = self.get_setting_float(
            "behavior_optimization.action_balance_target", 0.8
        )
        self.entropy_regularization = self.get_setting_float(
            "behavior_optimization.entropy_regularization", 0.01
        )
        self.action_smoothing = self.get_setting_float(
            "behavior_optimization.action_smoothing", 0.1
        )
        self.consistency_penalty = self.get_setting_float(
            "behavior_optimization.consistency_penalty", 0.05
        )
        self.balance_penalty = self.get_setting_float(
            "behavior_optimization.balance_penalty", 1.0
        )
        self.balance_penalty_tolerance = self.get_setting_float(
            "behavior_optimization.balance_penalty_tolerance", 0.05
        )

        # Validate and clamp action_balance_target
        original_target = self.action_balance_target
        self.action_balance_target = max(0.05, min(0.45, self.action_balance_target))
        if self.action_balance_target != original_target:
            self.logger.warning(
                f"action_balance_target {original_target} clamped to {self.action_balance_target}"
            )

        # Logging setup
        self._setup_logging()

    def _initialize_components(self, config: EnvironmentConfig):
        """Initialize all sub-components."""
        self.market_regime_detector = self._init_market_regime_detector()
        self.dynamic_reward_shaper = self._init_dynamic_reward_shaper()
        self.signal_integrator = self._init_signal_integrator(config)
        self.asymmetric_reward_scaler = AsymmetricRewardScaler(env_config=config)
        self.behavioral_penalty_calculator = BehavioralPenaltyCalculator(config=config)
        self.unrealized_loss_penalty_calculator = UnrealizedLossPenaltyCalculator(
            reward_settings=self.reward_settings
        )
        self.opportunity_cost_penalty_calculator = OpportunityCostPenaltyCalculator(
            reward_settings=self.reward_settings
        )

    def _init_market_regime_detector(self) -> MarketRegimeDetector:
        self.logger.debug("Initializing MarketRegimeDetector")
        return MarketRegimeDetector()

    def _init_dynamic_reward_shaper(self) -> DynamicRewardShaper:
        return DynamicRewardShaper(
            market_regime_detector=self.market_regime_detector,
            enabled=self.get_setting_bool("dynamic_reward_shaping.enabled", False),
            market_regime_awareness=self.get_setting_bool(
                "dynamic_reward_shaping.market_regime_awareness", False
            ),
            volatility_adjusted_rewards=self.get_setting_bool(
                "dynamic_reward_shaping.volatility_adjusted_rewards", False
            ),
            trend_strength_bonus=self.get_setting_bool(
                "dynamic_reward_shaping.trend_strength_bonus", False
            ),
            bull_market_bonus_coeff=self.get_setting_float(
                "dynamic_reward_shaping.regime_coefficients.bull_market_bonus_coeff",
                1.2,
            ),
            bear_market_penalty_coeff=self.get_setting_float(
                "dynamic_reward_shaping.regime_coefficients.bear_market_penalty_coeff",
                0.8,
            ),
            sideways_market_penalty_coeff=self.get_setting_float(
                "dynamic_reward_shaping.regime_coefficients.sideways_market_penalty_coeff",
                0.9,
            ),
            volatile_market_bonus_coeff=self.get_setting_float(
                "dynamic_reward_shaping.regime_coefficients.volatile_market_bonus_coeff",
                1.1,
            ),
            high_volatility_threshold=self.market_regime_detector.volatility_threshold,
            low_volatility_threshold=self.market_regime_detector.volatility_threshold
            * 0.5,
            high_volatility_bonus=self.get_setting_float(
                "dynamic_reward_shaping.volatility_coefficients.high_volatility_bonus",
                1.3,
            ),
            low_volatility_penalty=self.get_setting_float(
                "dynamic_reward_shaping.volatility_coefficients.low_volatility_penalty",
                0.7,
            ),
            trend_strength_threshold=self.market_regime_detector.trend_threshold,
            strong_trend_bonus=self.get_setting_float(
                "dynamic_reward_shaping.trend_coefficients.strong_trend_bonus", 1.2
            ),
            weak_trend_penalty=self.get_setting_float(
                "dynamic_reward_shaping.trend_coefficients.weak_trend_penalty", 0.9
            ),
        )

    def _init_signal_integrator(self, config: EnvironmentConfig) -> SignalIntegrator:
        signal_guidance_config = getattr(config, "signal_guidance", {})
        return SignalIntegrator(
            config=config,
            enabled=getattr(config, "signal_guidance_enabled", False),
            guidance_level=signal_guidance_config.get("guidance_level", "strong"),
            signal_bonus_weight=signal_guidance_config.get("signal_bonus_weight", 0.1),
            signal_penalty_weight=signal_guidance_config.get(
                "signal_penalty_weight", 0.05
            ),
            granville_weight=signal_guidance_config.get("granville_weight", 1.2),
            dow_theory_weight=signal_guidance_config.get("dow_theory_weight", 1.5),
            heikin_ashi_weight=signal_guidance_config.get("heikin_ashi_weight", 1.0),
            enable_advanced_integration=signal_guidance_config.get(
                "enable_advanced_integration", True
            ),
        )

    def _setup_logging(self):
        import logging

        reward_logger = logging.getLogger("ztb.trading.environment.reward")
        log_level_str = self.get_setting_str(
            "logging.reward_calculator_level", "WARNING"
        ).upper()
        level = getattr(logging, log_level_str, logging.WARNING)
        reward_logger.setLevel(level)

        # General logging counters for all curriculum stages
        self._curriculum_log_counter = 0
        self._curriculum_log_interval = 100  # Log every 100 steps for all stages

        self._forced_balance_log_counter = 0
        self._forced_balance_log_interval = 100
        self._forced_balance_last_state: Optional[str] = None
        self._forced_balance_last_summary_step = 0
        self._forced_balance_summary_interval = 500

        # Dynamic log level control
        self._dynamic_logging_enabled = self.get_setting_bool(
            "logging.dynamic_level_control", True
        )
        self._log_level_change_threshold = self.get_setting_int(
            "logging.level_change_threshold", 1000
        )
        self._current_log_level = level
        self._log_evaluation_counter = 0

    def set_log_level(self, level: str) -> None:
        """Dynamically set log level for reward calculator.

        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR)
        """
        import logging

        level_upper = level.upper()
        if hasattr(logging, level_upper):
            new_level = getattr(logging, level_upper)
            reward_logger = logging.getLogger("ztb.trading.environment.reward")
            reward_logger.setLevel(new_level)
            self._current_log_level = new_level
            self.structured_logger.info(
                "Log level changed",
                extra={"old_level": self._current_log_level, "new_level": new_level}
            )

    def _evaluate_dynamic_logging(self, step: int) -> None:
        """Evaluate and adjust log level based on training progress.

        Args:
            step: Current training step
        """
        if not self._dynamic_logging_enabled:
            return

        self._log_evaluation_counter += 1
        if self._log_evaluation_counter % self._log_level_change_threshold != 0:
            return

        # Reduce logging frequency as training progresses
        if step > 50000 and self._current_log_level < logging.WARNING:
            self.set_log_level("WARNING")
        elif step > 100000 and self._current_log_level < logging.ERROR:
            self.set_log_level("ERROR")

    def _map_action_to_index(self, action: int) -> int:
        """Normalize action identifiers to consistent indices [HOLD, BUY, SELL]."""
        action_int = int(action)
        if action_int in (ACTION_HOLD, 0):
            return 0
        if action_int in (ACTION_BUY, 1):
            return 1
        if action_int in (ACTION_SELL, -1, 2):
            return 2
        raise ValueError(
            f"Unsupported action value for forced balance tracking: {action}"
        )

    def _record_action(self, action: int) -> int:
        """Increment internal action counters and return normalized index."""
        action_index = self._map_action_to_index(action)
        self._action_counts[action_index] += 1
        return action_index

    def _map_forced_balance_penalty(self, deviation: float, severity: float) -> float:
        """Convert deviation above target into a scaled penalty value."""
        penalty_scale = self.get_setting_float("forced_balance.penalty.scale", 1.0)
        severity_multiplier = 1.0 + 0.5 * min(1.0, severity)

        # Get deviation thresholds and penalty values from settings with descriptive names
        thresh_small = self.get_setting_float(
            "forced_balance.penalty.threshold_small", 0.05
        )
        thresh_medium = self.get_setting_float(
            "forced_balance.penalty.threshold_medium", 0.1
        )
        thresh_large = self.get_setting_float(
            "forced_balance.penalty.threshold_large", 0.2
        )

        penalty_small = self.get_setting_float(
            "forced_balance.penalty.value_small_deviation", 10.0
        )
        penalty_medium = self.get_setting_float(
            "forced_balance.penalty.value_medium_deviation", 25.0
        )
        penalty_large = self.get_setting_float(
            "forced_balance.penalty.value_large_deviation", 50.0
        )
        penalty_very_large = self.get_setting_float(
            "forced_balance.penalty.value_very_large_deviation", 100.0
        )

        if deviation < thresh_small:
            base_penalty = penalty_small
        elif deviation < thresh_medium:
            base_penalty = penalty_medium
        elif deviation < thresh_large:
            base_penalty = penalty_large
        else:
            base_penalty = penalty_very_large
        return base_penalty * penalty_scale * severity_multiplier

    def _map_forced_balance_bonus(self, deviation: float, severity: float) -> float:
        """Convert deviation below target into a bonus encouraging corrective actions."""
        bonus_scale = self.get_setting_float("forced_balance.bonus.scale", 1.0)
        severity_multiplier = 1.0 + 0.5 * min(1.0, severity)

        # Get deviation thresholds and bonus values from settings with descriptive names
        thresh_small = self.get_setting_float(
            "forced_balance.bonus.threshold_small", 0.05
        )
        thresh_medium = self.get_setting_float(
            "forced_balance.bonus.threshold_medium", 0.1
        )

        bonus_small = self.get_setting_float(
            "forced_balance.bonus.value_small_deviation", 6.0
        )
        bonus_medium = self.get_setting_float(
            "forced_balance.bonus.value_medium_deviation", 12.0
        )
        bonus_large = self.get_setting_float(
            "forced_balance.bonus.value_large_deviation", 20.0
        )

        if deviation < thresh_small:
            base_bonus = bonus_small
        elif deviation < thresh_medium:
            base_bonus = bonus_medium
        else:
            base_bonus = bonus_large
        return base_bonus * bonus_scale * severity_multiplier

    def reset(self):
        """Resets the internal state of the calculator for a new episode."""
        self.logger.debug("Resetting RewardCalculator state.")
        self._action_counts = [0, 0, 0]
        self._consecutive_idle_steps = 0
        self._consecutive_position_hold_steps = 0
        self._win_count = 0
        self._loss_count = 0
        self.last_signal_strength = 0.0
        self.last_signal_reward = 0.0
        self._previous_portfolio_value = self.initial_portfolio_value
        self._last_reward_components = {}

        # Reset sub-components
        if hasattr(self, "behavioral_penalty_calculator") and hasattr(
            self.behavioral_penalty_calculator, "reset"
        ):
            self.behavioral_penalty_calculator.reset()

        if hasattr(self, "unrealized_loss_penalty_calculator") and hasattr(
            self.unrealized_loss_penalty_calculator, "reset"
        ):
            self.unrealized_loss_penalty_calculator.reset()

        if hasattr(self, "opportunity_cost_penalty_calculator") and hasattr(
            self.opportunity_cost_penalty_calculator, "reset"
        ):
            self.opportunity_cost_penalty_calculator.reset()

        self.logger.info("RewardCalculator has been reset.")

    def get_setting_float(self, key: str, default: float) -> float:
        """Get float reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, (int, float)):
                return float(value)
        return default

    def get_setting_int(self, key: str, default: int) -> int:
        """Get integer reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, (int, float)):
                return int(value)
        return default

    def get_setting_bool(self, key: str, default: bool) -> bool:
        """Get boolean reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
        return default

    def get_setting_str(self, key: str, default: str) -> str:
        """Get string reward setting with fallback. Supports nested keys with dot notation."""
        if self.reward_settings:
            value = self._get_nested_setting(key)
            if isinstance(value, str):
                return value
        return default

    def get_last_reward_components(self) -> Dict[str, Union[str, float]]:
        """Returns the components of the last calculated reward for debugging."""
        return self._last_reward_components

    def _get_nested_setting(
        self, key: str
    ) -> Optional[Union[int, float, bool, str, dict, list]]:
        """Get nested setting value using dot notation."""
        keys = key.split(".")
        value: Any = self.reward_settings

        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                elif hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return None
            return value
        except (KeyError, TypeError, AttributeError):
            return None

    def calculate_reward(
        self,
        action: int,
        current_price: float,
        position: float,
        portfolio_value: float,
        atr: float,
        transaction_cost: float,
        reward_scaling: float,
        pnl: float,
        old_position: float,
        step: int,
        observation: Optional[np.ndarray],
        reward_history: List[float],
        portfolio_value_history: List[float],
    ) -> float:
        """
        Calculate reward with curriculum learning stages.

        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            current_price: Current market price
            position: Current position
            portfolio_value: Current portfolio value
            atr: Average True Range
            transaction_cost: Transaction cost
            reward_scaling: Reward scaling factor
            pnl: Profit/Loss from action
            old_position: Position before action
            step: Current step number
            observation: Current observation (for feature access)
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values

        Returns:
            Calculated reward value
        """
        # Dynamic log level evaluation
        self._evaluate_dynamic_logging(step)

        self._last_reward_components = {}  # Reset at the beginning of each calculation

        # Debug logging for reward calculation inputs
        self.logger.debug(
            f"Reward calc inputs: action={action}, pnl={pnl:.2f}, position={position:.4f}, "
            f"portfolio_value={portfolio_value:.2f}, atr={atr:.2f}, current_price={current_price:.2f}, "
            f"old_position={old_position:.4f}, step={step}"
        )

        # Log curriculum stage info with throttling for all stages
        if self.config.curriculum_stage == "forced_balance":
            should_log_stage = (
                self._forced_balance_log_counter % self._forced_balance_log_interval
                == 0
            )
            if should_log_stage:
                self.logger.warning(
                    f"RewardCalculator: curriculum_stage={self.config.curriculum_stage}, total_actions={sum(self._action_counts)}"
                )
        else:
            # For other stages, use general throttling
            should_log_stage = (
                self._curriculum_log_counter % self._curriculum_log_interval == 0
            )
            if should_log_stage:
                self.logger.warning(
                    f"RewardCalculator: curriculum_stage={self.config.curriculum_stage}, total_actions={sum(self._action_counts)}"
                )

        # Increment general counter for all stages
        self._curriculum_log_counter += 1

        self.last_signal_strength = 0.0
        self.last_signal_reward = 0.0

        # Record the action for behavioral analysis BEFORE calculating penalties
        self.behavioral_penalty_calculator.record_action(action)

        # Update win/loss counts
        if pnl > 0:
            self._win_count += 1
        elif pnl < 0:
            self._loss_count += 1

        # Check if simple reward is enabled
        use_simple_reward = self.get_setting_bool("use_simple_reward", False)

        if use_simple_reward:
            return self.calculate_reward_simple(
                pnl,
                portfolio_value,
                position,
                old_position,
                action,
                reward_history,
                portfolio_value_history,
                current_price,
                step,
                transaction_cost,
            )

        # Original complex reward function
        curriculum_stage = self.config.curriculum_stage
        self.logger.info(
            "Curriculum stage: %s, position: %.2f, action: %d",
            curriculum_stage,
            position,
            action,
        )

        eps = self.get_setting_float("eps", 1e-8)
        atr = atr if atr > eps else 1.0
        max_position_size = max(eps, self.config.max_position_size)

        # Calculate effective max position considering capital constraints
        effective_max_position = min(
            max_position_size, self.initial_portfolio_value / max(current_price, eps)
        )
        self.logger.debug(
            f"Position calculations: max_position_size={max_position_size:.4f}, effective_max_position={effective_max_position:.4f}, initial_portfolio_value={self.initial_portfolio_value:.2f}, current_price={current_price:.2f}"
        )

        # Adapt reward scaling based on max position size to prevent clipping
        scale_adjustment_base = self.get_setting_float("scale_adjustment_base", 1.0)
        scale_adjustment = scale_adjustment_base / max(0.01, max_position_size)
        reward_scaling = reward_scaling * scale_adjustment

        atr_normalised = pnl / atr
        portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # Adjust pnl for transaction costs if a trade occurred
        # Note: transaction_cost is already deducted in position_manager, so we don't deduct again
        # if abs(position - old_position) > eps:  # Trade occurred
        #     adjusted_pnl = pnl - transaction_cost
        #     self.logger.debug(f"Trade occurred: adjusted pnl from {pnl:.4f} to {adjusted_pnl:.4f} (transaction_cost: {transaction_cost:.4f})")
        #     pnl = adjusted_pnl
        #     atr_normalised = pnl / atr
        #     portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # v444: Refactored reward calculation to be more modular and extensible.
        # 1. Select the appropriate reward calculation method based on the curriculum stage.
        # 2. Calculate the base reward using the selected method.
        # 3. Apply common post-processing steps (scaling, clipping, signal integration).

        # Add balance penalty for BUY/SELL actions
        balance_penalty = 0.0
        action_bonus = 0.0
        if action == ACTION_BUY:
            action_bonus = self.config.action_bonuses.get("buy_action_bonus", 0.0)
        elif action == ACTION_SELL:
            action_bonus = self.config.action_bonuses.get("sell_action_bonus", 0.0)
        elif action == ACTION_HOLD:
            action_bonus = self.config.action_bonuses.get("hold_action_bonus", 0.0)

        if action in [ACTION_BUY, ACTION_SELL]:
            balance_penalty = (
                self.behavioral_penalty_calculator.calculate_balance_penalty(
                    action, action_bonus
                )
            )
            self._last_reward_components["balance_penalty"] = balance_penalty
        self._last_reward_components["action_bonus"] = action_bonus

        # Create a mapping from curriculum stage to the corresponding reward calculation method
        stage_to_method_map = {
            "forced_balance": self._calculate_forced_balance_reward,
            "balanced_transition": self._calculate_balanced_transition_reward,
            "pnl_focused": self._calculate_pnl_focused_reward,
            "trading_focused": self._calculate_trading_focused_reward,
            "profit_optimized": self._calculate_profit_optimized_reward,
            "risk_management": self._calculate_risk_management_reward,
            "opportunity_cost": self._calculate_opportunity_cost_reward,
            "ultra_profit": self._calculate_ultra_profit_reward,
            "stability_optimized": self._calculate_stability_optimized_reward,
            "backtest_optimization": self._calculate_backtest_optimization_reward,
        }

        # Get the reward calculation method for the current stage, defaulting to _calculate_default_reward
        reward_method = (
            stage_to_method_map.get(curriculum_stage, self._calculate_default_reward)
            if curriculum_stage
            else self._calculate_default_reward
        )

        # Prepare arguments for the reward method
        method_args = {
            "action": action,
            "atr_normalised": atr_normalised,
            "portfolio_return": portfolio_return,
            "position": position,
            "effective_max_position": effective_max_position,
            "current_price": current_price,
            "atr": atr,
            "pnl": pnl,
            "reward_scaling": reward_scaling,
            "observation": observation,
            "step": step,
            "portfolio_value_delta": portfolio_value - self._previous_portfolio_value,
        }
        self._previous_portfolio_value = portfolio_value

        # Filter arguments for the specific method being called
        sig = inspect.signature(reward_method)
        valid_args = {k: v for k, v in method_args.items() if k in sig.parameters}

        # Calculate the base reward for the current stage
        base_reward = reward_method(**valid_args)

        # Apply action bonus directly to reward
        base_reward += action_bonus

        # Record action bonus in components
        self._last_reward_components["action_bonus"] = action_bonus

        # Apply the balance penalty calculated earlier
        base_reward += balance_penalty

        # Apply common post-processing to the base reward
        final_reward = self._post_process_reward(
            base_reward,
            position,
            pnl,
            observation,
            action,
            step,
            effective_max_position,
        )

        self._last_reward_components["final_reward"] = final_reward
        return final_reward

    def _post_process_reward(
        self,
        reward: float,
        position: float,
        pnl: float,
        observation: Optional[np.ndarray],
        action: int,
        step: int,
        effective_max_position: float,
    ) -> float:
        """
        Apply common post-processing steps to a calculated reward.
        - Asymmetric scaling based on position and PnL.
        - Reward value clipping.
        - Signal integration for guidance.
        """
        # Apply asymmetric scaling
        reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)
        self._last_reward_components["after_asymmetric_scaling"] = reward

        # Apply clipping
        reward_clip_min = self.get_setting_float("reward_clip_min", -80.0)
        reward_clip_max = self.get_setting_float("reward_clip_max", 80.0)
        reward = np.clip(reward, reward_clip_min, reward_clip_max)
        self._last_reward_components["after_clipping"] = reward

        # Apply signal integration
        reward = self.signal_integrator.integrate_signal(
            reward, observation, action, step
        )
        self._last_reward_components["after_signal_integration"] = reward

        return reward

    def calculate_reward_simple(
        self,
        pnl: float,
        portfolio_value: float,
        position: float,
        old_position: float,
        action: int,
        reward_history: List[float],
        portfolio_value_history: List[float],
        current_price: float = 0.0,
        step: int = 0,
        transaction_cost: float = 0.0,
    ) -> float:
        """
        Calculate simple reward based on PnL with v431 enhancements and v440.1 dynamic shaping.

        Enhanced with v431 successful elements:
        - HOLD penalty multiplier for encouraging trading activity
        - Trade frequency bonus for promoting active trading
        - Reward scaling and clipping for stable learning
        - Dynamic reward shaping based on market conditions (v440.1)

        Args:
            pnl: Profit/Loss from action
            portfolio_value: Current portfolio value
            position: Current position
            old_position: Position before action
            action: Action taken
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values
            current_price: Current market price (for regime detection)
            step: Current step number (for regime adaptation)
            transaction_cost: Transaction cost for the action

        Returns:
            Enhanced simple reward value
        """
        try:
            self.logger.debug(
                f"calculate_reward_simple called: pnl={pnl}, action={action}"
            )
            # Basic NaN/inf checks
            if np.isnan(pnl) or np.isinf(pnl):
                self.logger.error(
                    "RewardCalculator failed, using simple reward: math range error"
                )
                return 0.0

            if np.isnan(portfolio_value) or np.isinf(portfolio_value):
                self.logger.error(
                    "RewardCalculator failed, using simple reward: invalid portfolio_value"
                )
                return 0.0

            if np.isnan(position) or np.isinf(position):
                self.logger.error(
                    "RewardCalculator failed, using simple reward: invalid position"
                )
                return 0.0

            # Get v431 enhancement parameters
            hold_penalty_multiplier = self.get_setting_float(
                "hold_penalty_multiplier", 1.0
            )
            trade_frequency_bonus = self.get_setting_float("trade_frequency_bonus", 0.0)
            reward_scaling = self.get_setting_float("reward_scaling", 1.0)
            reward_clip_value = self.get_setting_float("reward_clip_value", 10.0)

            # Adjust PnL for transaction costs if trade occurred
            # Note: transaction_cost is already deducted in position_manager, so we don't deduct again
            adjusted_pnl = pnl

            # Base PnL-based reward (scaled)
            if adjusted_pnl > 0:
                reward = adjusted_pnl * reward_scaling
            elif adjusted_pnl < 0:
                reward = adjusted_pnl * reward_scaling
            else:
                reward = 0.0

            # HOLD penalty (v431 enhancement) - encourage trading activity
            if action == ACTION_HOLD:  # 0 = HOLD
                reward *= hold_penalty_multiplier

            # Trade frequency bonus (v431 enhancement) - promote active trading
            if action in [ACTION_BUY, ACTION_SELL]:  # 1 = BUY, 2 = SELL
                reward += trade_frequency_bonus

            # Small penalty for large position changes to encourage stability
            position_change = abs(position - old_position)
            if position_change > 0.1:  # Large position change
                reward -= 0.1

            # Apply dynamic reward shaping (v440.1 enhancement)
            reward = self.dynamic_reward_shaper.shape_reward(
                reward, current_price, step, pnl
            )

            # Apply signal-based reward integration
            if self.signal_integrator.enabled:
                # Create observation from current state (simplified for now)
                observation = np.array([current_price, position, pnl])
                signal_reward = self.signal_integrator.integrate_signal(
                    reward=reward, observation=observation, action=action, step=step
                )
                reward = signal_reward  # integrate_signal returns the modified reward

            # Apply asymmetric reward scaling based on position
            reward = self.asymmetric_reward_scaler.scale_reward(reward, position, pnl)

            # Apply reward clipping
            # reward = np.clip(reward, -reward_clip_value, reward_clip_value)

            # Ensure reward is finite
            if not np.isfinite(reward):
                self.logger.error(
                    "RewardCalculator failed, using simple reward: non-finite reward"
                )
                return 0.0

            return reward

        except Exception as e:
            self.logger.error(f"RewardCalculator failed, using simple reward: {e}")
            return 0.0

    def _calculate_forced_balance_reward(self, action: int, step: int) -> float:
        """Stage: Forced balance reward that encourages corrective actions toward configured targets."""
        action_index = self._record_action(action)
        total_actions = sum(self._action_counts)

        self._last_reward_components = {"stage": "forced_balance"}

        # Log control: only log every N steps or on state changes
        should_log_detailed = (
            self._forced_balance_log_counter % self._forced_balance_log_interval == 0
        )
        self._forced_balance_log_counter += 1

        min_actions = self.get_setting_int("forced_balance_min_actions", 10)
        exploration_reward = self.get_setting_float(
            "forced_balance_exploration_reward", 2.0
        )
        if total_actions < min_actions:
            if should_log_detailed:
                self.logger.warning(
                    f"Forced balance: early phase (total_actions={total_actions} < {min_actions}), using exploration reward"
                )
            self._last_reward_components["base_reward"] = exploration_reward
            return exploration_reward

        hold_target = self.get_setting_float(
            "balance_penalty_targets.hold_target", 1.0 / 3.0
        )
        buy_target = self.get_setting_float(
            "balance_penalty_targets.buy_target", 1.0 / 3.0
        )
        sell_target = self.get_setting_float(
            "balance_penalty_targets.sell_target", 1.0 / 3.0
        )
        target_ratios = [hold_target, buy_target, sell_target]

        action_ratios = [count / total_actions for count in self._action_counts]
        signed_deviations = [
            ratio - target for ratio, target in zip(action_ratios, target_ratios)
        ]
        abs_deviations = [abs(dev) for dev in signed_deviations]
        rms_deviation = (
            sum(dev**2 for dev in signed_deviations) / len(signed_deviations)
        ) ** 0.5
        max_abs_deviation = max(abs_deviations)
        max_over_deviation = max(
            (dev for dev in signed_deviations if dev > 0), default=0.0
        )
        max_under_deviation = min(
            (dev for dev in signed_deviations if dev < 0), default=0.0
        )

        balance_broken_threshold = self.get_setting_float(
            "forced_balance_threshold", 0.15
        )
        is_imbalanced = max_abs_deviation > balance_broken_threshold

        state_parts = []
        for idx, dev in enumerate(signed_deviations):
            if abs(dev) > balance_broken_threshold:
                direction = "over" if dev > 0 else "under"
                state_parts.append(
                    f"{self.ACTION_INDEX_NAMES[idx]}_{direction}:{abs(dev):.3f}"
                )
        current_state = "|".join(state_parts) if state_parts else "balanced"

        if should_log_detailed or current_state != self._forced_balance_last_state:
            deviations_str = ", ".join(
                f"{name}={ratio:.3f} ({dev:+.3f})"
                for name, ratio, dev in zip(
                    self.ACTION_INDEX_NAMES, action_ratios, signed_deviations
                )
            )
            self.logger.warning(
                f"Forced balance: total_actions={total_actions}, rms_dev={rms_deviation:.3f}, "
                f"max_dev={max_abs_deviation:.3f}, max_over={max_over_deviation:.3f}, "
                f"max_under={abs(max_under_deviation):.3f}, state={current_state}, deviations=[{deviations_str}]"
            )
            self._forced_balance_last_state = current_state

        if (
            step - self._forced_balance_last_summary_step
            >= self._forced_balance_summary_interval
        ):
            ratios_str = ", ".join(f"{ratio:.3f}" for ratio in action_ratios)
            deviation_summary = ", ".join(f"{dev:+.3f}" for dev in signed_deviations)
            self.logger.info(
                f"Forced balance SUMMARY [Step {step}]: total_actions={total_actions}, "
                f"ratios=[{ratios_str}], signed_dev=[{deviation_summary}], "
                f"rms_dev={rms_deviation:.3f}, max_dev={max_abs_deviation:.3f}, "
                f"state={current_state}, counts={self._action_counts}"
            )
            self._forced_balance_last_summary_step = step

        if not is_imbalanced:
            balanced_reward = self.get_setting_float(
                "forced_balance_balanced_reward", 2.0
            )
            self._last_reward_components["base_reward"] = balanced_reward
            return balanced_reward

        global_penalty_scale = self.get_setting_float(
            "forced_balance_global_penalty_scale", 0.0
        )
        global_pressure = -global_penalty_scale * max_abs_deviation

        current_deviation = signed_deviations[action_index]
        if current_deviation > 0:
            penalty = self._map_forced_balance_penalty(
                current_deviation, max_abs_deviation
            )
            reward = global_pressure - penalty
            self._last_reward_components["imbalance_penalty"] = -penalty
        elif current_deviation < 0:
            bonus = self._map_forced_balance_bonus(
                abs(current_deviation), max_abs_deviation
            )
            reward = global_pressure + bonus
            self._last_reward_components["corrective_bonus"] = bonus
        else:
            on_target_reward = self.get_setting_float(
                "forced_balance_on_target_reward", 2.0
            )
            reward = global_pressure + on_target_reward
            self._last_reward_components["on_target_bonus"] = on_target_reward

        self._last_reward_components["base_reward"] = reward

        if should_log_detailed:
            self.logger.debug(
                "Forced balance decision: action=%s, deviation=%.3f, global_pressure=%.3f, reward=%.3f",
                self.ACTION_INDEX_NAMES[action_index],
                current_deviation,
                global_pressure,
                reward,
            )

        # Apply scaling specific to this stage
        forced_balance_scaling = self.get_setting_float("forced_balance.scaling", 1.0)
        reward *= forced_balance_scaling
        self._last_reward_components["scaled_reward"] = reward

        return reward

    def _calculate_balanced_transition_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Stage 1: Normal reward with balance penalty."""
        self._record_action(action)
        total_actions = sum(self._action_counts)

        self._last_reward_components = {"stage": "balanced_transition"}

        tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)
        penalty = (
            self.balance_penalty
        )  # Use the initialized balance_penalty from behavior_optimization
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]
            # Get target ratios from config
            hold_target = self.get_setting_float(
                "balance_penalty_targets.hold_target", 0.4
            )
            buy_target = self.get_setting_float(
                "balance_penalty_targets.buy_target", 0.3
            )
            sell_target = self.get_setting_float(
                "balance_penalty_targets.sell_target", 0.3
            )
            target_ratios = [hold_target, buy_target, sell_target]  # [HOLD, BUY, SELL]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    # Penalty proportional to deviation beyond tolerance
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation
                    self.logger.info(
                        f"Balance penalty applied: {balance_penalty:.3f}, ratios: {action_ratios}, targets: {target_ratios}"
                    )

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )
        self._last_reward_components["base_reward"] = base_reward
        self._last_reward_components["balance_penalty"] = balance_penalty

        final_reward = base_reward - balance_penalty
        self.logger.info(
            f"Balanced transition: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, final_reward={final_reward:.3f}, action_counts={self._action_counts}"
        )
        return final_reward * reward_scaling

    def _calculate_trading_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Stage: Trading-focused reward that heavily penalizes HOLD and encourages trading."""
        self._record_action(action)
        total_actions = sum(self._action_counts)

        self._last_reward_components = {"stage": "trading_focused"}

        # Trading-focused balance penalty: HOLD 10%, BUY 45%, SELL 45%
        target_ratios = [0.1, 0.45, 0.45]  # [HOLD, BUY, SELL] - minimize HOLD
        tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)
        penalty = self.get_setting_float(
            "balance_penalty", 8.0
        )  # Higher penalty for trading focus
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > tolerance:
                    # Penalty proportional to deviation beyond tolerance
                    excess_deviation = deviation - tolerance
                    balance_penalty += penalty * excess_deviation

            self.logger.info(
                f"Trading focused penalty applied: {balance_penalty:.3f}, ratios: {action_ratios}, targets: {target_ratios}"
            )

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
        )

        # Add strong HOLD penalty
        hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.01)
        if action == ACTION_HOLD:
            # Strong penalty for HOLD action
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty
            self.logger.debug(f"HOLD penalty applied: {hold_penalty:.3f}")

        # Boost trading bonuses
        trading_bonus_multiplier = self.get_setting_float(
            "trading_bonus_multiplier", 2.0
        )
        if action in [ACTION_BUY, ACTION_SELL]:
            trading_bonus = (
                self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
            )
            base_reward += trading_bonus
            self.logger.debug(f"Trading bonus applied: {trading_bonus:.3f}")

        final_reward = base_reward - balance_penalty
        self.logger.info(
            f"Trading focused: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, final_reward={final_reward:.3f}, action_counts={self._action_counts}"
        )
        return final_reward * reward_scaling

    def _calculate_profit_optimized_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
        observation: np.ndarray = None,
    ) -> float:
        """Stage: Profit-optimized reward that maximizes profitable trading while minimizing losses."""
        self._record_action(action)
        self._last_reward_components = {"stage": "profit_optimized"}

        # 1. Calculate base reward from PnL
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
            observation,
        )

        # 2. Apply profit/loss modifiers and specific action bonuses/penalties
        profit_multiplier = self.get_setting_float("profit_multiplier", 2.0)
        loss_penalty_multiplier = self.get_setting_float("loss_penalty_multiplier", 1.5)
        profit_sell_penalty_rate = self.get_setting_float(
            "profit_sell_penalty_rate", 0.0
        )
        profit_hold_bonus_rate = self.get_setting_float("profit_hold_bonus_rate", 0.0)

        pnl_normalizer = atr * effective_max_position * current_price
        normalized_pnl = pnl / max(pnl_normalizer, 1e-8)

        if pnl > 0:
            profit_bonus = normalized_pnl * profit_multiplier
            base_reward += profit_bonus
            self._last_reward_components["profit_bonus"] = profit_bonus

            if action == ACTION_SELL and profit_sell_penalty_rate > 0:
                profit_sell_penalty = normalized_pnl * profit_sell_penalty_rate
                base_reward -= profit_sell_penalty
                self._last_reward_components[
                    "profit_sell_penalty"
                ] = -profit_sell_penalty

            if action == ACTION_HOLD and profit_hold_bonus_rate > 0:
                profit_hold_bonus = normalized_pnl * profit_hold_bonus_rate
                base_reward += profit_hold_bonus
                self._last_reward_components["profit_hold_bonus"] = profit_hold_bonus

        elif pnl < 0:
            loss_penalty = abs(normalized_pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty
            self._last_reward_components["loss_penalty"] = -loss_penalty

        # 3. Apply common rewards/penalties for trading actions
        if action in [ACTION_BUY, ACTION_SELL]:
            base_reward = self._calculate_base_trading_reward(
                base_reward, position, effective_max_position
            )
        elif action == ACTION_HOLD:
            hold_penalty_rate = self.get_setting_float("hold_penalty_rate", 0.1)
            hold_penalty = (
                hold_penalty_rate * abs(position) / max(effective_max_position, 0.01)
            )
            base_reward -= hold_penalty
            self._last_reward_components["hold_penalty"] = -hold_penalty

        # 4. Apply balance penalty at the end
        balance_penalty = 0.0
        total_actions = sum(self._action_counts)
        if total_actions >= 10:
            target_ratios = [0.15, 0.425, 0.425]  # [HOLD, BUY, SELL]
            tolerance = self.get_setting_float("balance_penalty_tolerance", 0.05)
            penalty_val = self.get_setting_float("balance_penalty", 6.0)
            action_ratios = [count / total_actions for count in self._action_counts]

            deviation = abs(
                action_ratios[self._map_action_to_index(action)]
                - target_ratios[self._map_action_to_index(action)]
            )
            if deviation > tolerance:
                excess_deviation = deviation - tolerance
                balance_penalty = penalty_val * excess_deviation

        final_reward = base_reward - balance_penalty
        self._last_reward_components["balance_penalty"] = -balance_penalty

        self.logger.info(
            f"Profit optimized: base_reward={base_reward:.3f}, balance_penalty={balance_penalty:.3f}, pnl={pnl:.3f}, final_reward={final_reward:.3f}"
        )
        return final_reward * reward_scaling

    def _calculate_base_trading_reward(
        self, base_reward: float, position: float, effective_max_position: float
    ) -> float:
        """Calculates common reward components for BUY and SELL actions."""

        # Trading bonus
        trading_bonus_multiplier = self.get_setting_float(
            "trading_bonus_multiplier", 3.0
        )
        trading_bonus = (
            self.get_setting_float("trading_bonus", 0.01) * trading_bonus_multiplier
        )
        base_reward += trading_bonus
        self._last_reward_components["trading_bonus"] = trading_bonus

        # Position size bonus
        position_size_bonus_rate = self.get_setting_float(
            "position_size_bonus_rate", 0.05
        )
        position_utilization = abs(position) / max(effective_max_position, 0.01)
        if 0.1 <= position_utilization <= 0.8:
            position_size_bonus = position_size_bonus_rate * position_utilization
            base_reward += position_size_bonus
            self._last_reward_components["position_size_bonus"] = position_size_bonus

        # Activity incentive bonus
        activity_bonus_rate = self.get_setting_float("activity_bonus_rate", 0.02)
        recent_trades = sum(1 for a in self._recent_actions[-5:] if a != ACTION_HOLD)
        if recent_trades >= 2:
            activity_bonus = activity_bonus_rate * (recent_trades / 5.0)
            base_reward += activity_bonus
            self._last_reward_components["activity_bonus"] = activity_bonus

        return base_reward

    def _calculate_ultra_profit_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
    ) -> float:
        """Simplified ultra-profit reward that focuses on basic trading principles."""

        # Basic reward components
        reward = 0.0

        # Profit/Loss component - normalized by ATR
        if pnl > 0:
            reward += atr_normalised * 2.0  # Increased reward for profits
        elif pnl < 0:
            reward -= abs(atr_normalised) * 1.0  # Penalty for losses

        # Position penalty - discourage excessive position sizes
        position_utilization = abs(position) / effective_max_position
        if position_utilization > 0.5:
            reward -= (position_utilization - 0.5) * 0.2

        # Action diversity encouragement - reduced penalty
        self._record_action(action)
        total_actions = sum(self._action_counts)

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]
            # Encourage balanced actions: roughly 20% HOLD, 40% BUY, 40% SELL
            target_ratios = [0.2, 0.4, 0.4]

            for i, ratio in enumerate(action_ratios):
                deviation = abs(ratio - target_ratios[i])
                if deviation > 0.15:  # More lenient deviation
                    reward -= deviation * 0.005

        # Strong trading bonus for BUY/SELL actions to encourage trading
        if action in [ACTION_BUY, ACTION_SELL]:
            reward += 0.1  # Increased trading bonus

        # Apply scaling and clipping
        reward *= reward_scaling
        reward_clip_min = self.get_setting_float("reward_clip_min", -1.0)
        reward_clip_max = self.get_setting_float("reward_clip_max", 1.0)
        reward = np.clip(reward, reward_clip_min, reward_clip_max)

        self.logger.debug(
            f"Simplified ultra profit reward: pnl={pnl:.4f}, atr_normalised={atr_normalised:.4f}, position={position:.4f}, action={action}, reward={reward:.6f}"
        )
        return reward

    def _calculate_pnl_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
        observation: Optional[np.ndarray],
        step: int,
    ) -> float:
        """Stage 2: PnL-focused reward with trend analysis."""
        base_profit_bonus = (
            max(
                0.0,
                self.get_setting_float("base_profit_bonus_atr_coeff", 1.5)
                * atr_normalised
                + self.get_setting_float("base_profit_bonus_portfolio_coeff", 1.2)
                * portfolio_return,
            )
            if pnl > 0
            else 0.0
        )

        # Trend analysis if observation available
        trend_multiplier = 1.0
        if observation is not None and hasattr(observation, "__getitem__"):
            try:
                # Assume RSI and MACD are in observation
                rsi_idx = -2 if len(observation) > 2 else None
                macd_idx = -1 if len(observation) > 1 else None

                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(observation[rsi_idx])
                    macd_hist = float(observation[macd_idx])

                    # Check for NaN/inf
                    if (
                        np.isnan(rsi)
                        or np.isinf(rsi)
                        or np.isnan(macd_hist)
                        or np.isinf(macd_hist)
                    ):
                        trend_multiplier = 1.0
                    else:
                        # Trend ratio
                        trend_ratio = (rsi / 50.0) * (1.0 + macd_hist / 100.0)

                        if trend_ratio > 1.0 and action == ACTION_BUY:
                            trend_multiplier = 1.2
                        elif trend_ratio < 1.0 and action == ACTION_SELL:
                            trend_multiplier = 1.2

                        # Oversold/Overbought signals
                        if rsi > 60.0 and trend_ratio > 1.0 and action == ACTION_SELL:
                            trend_multiplier *= 1.3
                        elif rsi < 40.0 and trend_ratio < 1.0 and action == ACTION_SELL:
                            trend_multiplier *= 1.3
            except (IndexError, TypeError, ValueError):
                pass

        # Balance BUY/SELL actions
        # profit_bonus_multipliers array order: [BUY, SELL, HOLD]
        multipliers_raw = self._get_nested_setting("profit_bonus_multipliers") or [
            1.0,
            1.0,
            0.8,
        ]
        if isinstance(multipliers_raw, list) and len(multipliers_raw) >= 3:
            multipliers = [float(x) for x in multipliers_raw[:3]]
        else:
            multipliers = [1.0, 1.0, 0.8]

        if action == ACTION_BUY:
            profit_bonus = (
                base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY] * trend_multiplier
            )
        elif action == ACTION_SELL:
            profit_bonus = (
                base_profit_bonus
                * multipliers[MULTIPLIER_INDEX_SELL]
                * trend_multiplier
            )
        else:  # HOLD
            profit_bonus = (
                base_profit_bonus
                * multipliers[MULTIPLIER_INDEX_HOLD]
                * trend_multiplier
            )

        # Action penalties with configurable per-action penalties
        # Get action-specific penalties from settings (default: 0.0)
        hold_penalty = self.get_setting_float("hold_action_penalty", 0.0)
        buy_penalty = self.get_setting_float("buy_action_penalty", 0.0)
        sell_penalty = self.get_setting_float("sell_action_penalty", 0.0)

        # Base action penalties (legacy behavior)
        base_action_penalty = (
            self.get_setting_float("base_action_penalty", 0.015)
            if action in [ACTION_BUY, ACTION_SELL]
            else 0.0
        )

        if action == ACTION_HOLD:
            position_size_factor = abs(position) / effective_max_position
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = self.get_setting_float("hold_penalty_base", 0.01) + (
                self.get_setting_float("hold_penalty_position_factor", 0.04)
                * position_size_factor
                * volatility_factor
            )
            base_action_penalty *= self.get_setting_float(
                "hold_penalty_multiplier", 1.0
            )
            # Add configured HOLD penalty
            action_penalty = base_action_penalty + hold_penalty
        elif action == ACTION_BUY:
            # Add configured BUY penalty (negative value = reward)
            action_penalty = base_action_penalty + buy_penalty
        else:  # ACTION_SELL
            # Add configured SELL penalty (negative value = reward)
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = (
            self.get_setting_float("loss_penalty_coeff", -0.2) * abs(atr_normalised)
            if pnl < 0
            else 0.0
        )

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty
        self.logger.debug(
            f"PnL focused reward components: profit_bonus={profit_bonus:.4f}, action_penalty={action_penalty:.4f}, loss_penalty={loss_penalty:.4f}, position_penalty={position_penalty:.4f}, final={reward:.4f}"
        )
        return reward

    def _calculate_default_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[np.ndarray] = None,
        step: int = 0,
        portfolio_value_delta: float = 0.0,
    ) -> float:
        """Default reward calculation."""
        self._last_reward_components = {"stage": "default"}

        # PnL reward
        pnl_reward = self._calculate_pnl_reward(pnl, 1.0)  # Default scaling
        self._last_reward_components["pnl_reward"] = pnl_reward

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )
        self._last_reward_components["position_penalty"] = position_penalty

        # Hold penalty
        hold_penalty = self._calculate_hold_penalty(action)
        self._last_reward_components["hold_penalty"] = hold_penalty

        # Consistency penalty
        consistency_penalty = (
            self.behavioral_penalty_calculator.calculate_consistency_penalty()
        )
        self._last_reward_components["consistency_penalty"] = consistency_penalty

        # Combine rewards
        reward = pnl_reward + position_penalty + hold_penalty + consistency_penalty
        self._last_reward_components["total_reward"] = reward
        return reward

    def _calculate_pnl_reward(self, pnl: float, reward_scaling: float) -> float:
        """Calculates the basic profit and loss reward component."""
        pnl_reward_multiplier = self.get_setting_float("pnl_reward_multiplier", 1.0)
        return pnl * reward_scaling * pnl_reward_multiplier

    def _calculate_position_penalty(
        self, position: float, effective_max_position: float
    ) -> float:
        """Calculates a penalty for holding a large position."""
        position_penalty_weight = self.get_setting_float(
            "position_penalty_weight", 0.01
        )
        # Normalize position size and apply a quadratic penalty
        normalized_position = position / max(effective_max_position, EPSILON)
        return -position_penalty_weight * (normalized_position**2)

    def _calculate_hold_penalty(self, action: int) -> float:
        """Calculates a penalty for holding the position."""
        if action == ACTION_HOLD:
            hold_penalty_weight = self.get_setting_float("hold_penalty_weight", 0.01)
            return -hold_penalty_weight
        return 0.0

    def _calculate_portfolio_correlation_bonus(
        self, pnl: float, portfolio_value_delta: float
    ) -> float:
        """
        Calculates a bonus if the PnL of the last action is positively correlated
        with the change in total portfolio value.
        """
        correlation_bonus_weight = self.get_setting_float(
            "portfolio_correlation_bonus_weight", 0.1
        )
        if np.sign(pnl) == np.sign(portfolio_value_delta):
            return correlation_bonus_weight * abs(pnl)
        return 0.0

    def test_reward_calculation(self) -> None:
        """Test reward calculation with sample inputs."""
        # Sample inputs
        test_cases = [
            {
                "action": ACTION_HOLD,
                "pnl": 100.0,
                "position": 0.01,
                "portfolio_value": 200000.0,
                "atr": 500.0,
                "current_price": 5000000.0,
                "old_position": 0.0,
                "step": 1,
                "description": "HOLD with small profit",
            },
            {
                "action": ACTION_BUY,
                "pnl": 200.0,
                "position": 0.02,
                "portfolio_value": 200000.0,
                "atr": 500.0,
                "current_price": 5000000.0,
                "old_position": 0.01,
                "step": 1,
                "description": "BUY with profit",
            },
            {
                "action": ACTION_SELL,
                "pnl": -100.0,
                "position": 0.0,
                "portfolio_value": 200000.0,
                "atr": 500.0,
                "current_price": 5000000.0,
                "old_position": 0.02,
                "step": 1,
                "description": "SELL with loss",
            },
        ]

        self.logger.debug("Testing reward calculation...")
        for i, case in enumerate(test_cases):
            reward = self.calculate_reward(
                action=case["action"],
                current_price=case["current_price"],
                position=case["position"],
                portfolio_value=case["portfolio_value"],
                atr=case["atr"],
                transaction_cost=10.0,
                reward_scaling=10.0,  # Scale rewards for testing
                pnl=case["pnl"],
                old_position=case["old_position"],
                step=case["step"],
                observation=None,
                reward_history=[],
                portfolio_value_history=[case["portfolio_value"]] * 30,
            )
            self.logger.debug(
                f"Test {i+1}: {case['description']} -> Reward: {reward:.4f}"
            )
            self.logger.debug(
                f"  Components: pnl={case['pnl']:.2f}, position={case['position']:.4f}, atr={case['atr']:.2f}"
            )

    def _calculate_stability_optimized_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
        observation: Optional[np.ndarray],
        step: int,
    ) -> float:
        """Stage: Stability optimized reward."""
        self._last_reward_components = {"stage": "stability_optimized"}

        # PnL reward
        pnl_reward = self._calculate_pnl_reward(pnl, reward_scaling)
        self._last_reward_components["pnl_reward"] = pnl_reward

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )
        self._last_reward_components["position_penalty"] = position_penalty

        # Hold penalty
        hold_penalty = self._calculate_hold_penalty(action)
        self._last_reward_components["hold_penalty"] = hold_penalty

        # Consistency penalty
        consistency_penalty = (
            self.behavioral_penalty_calculator.calculate_consistency_penalty()
        )
        self._last_reward_components["consistency_penalty"] = consistency_penalty

        # Dynamic shaping
        dynamic_shaping_reward = self.dynamic_reward_shaper.shape_reward(
            pnl_reward, current_price, step, pnl
        )
        self._last_reward_components["dynamic_shaping_reward"] = dynamic_shaping_reward

        # Combine rewards
        reward = (
            pnl_reward
            + position_penalty
            + hold_penalty
            + consistency_penalty
            + dynamic_shaping_reward
        )
        self._last_reward_components["total_reward"] = reward
        return reward

    def _calculate_backtest_optimization_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
        observation: Optional[np.ndarray],
        step: int,
        portfolio_value_delta: float,
    ) -> float:
        """Stage: Backtest optimization reward."""
        self._last_reward_components = {"stage": "backtest_optimization"}

        # PnL reward
        pnl_reward = self._calculate_pnl_reward(pnl, reward_scaling)
        self._last_reward_components["pnl_reward"] = pnl_reward

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )
        self._last_reward_components["position_penalty"] = position_penalty

        # Hold penalty
        hold_penalty = self._calculate_hold_penalty(action)
        self._last_reward_components["hold_penalty"] = hold_penalty

        # Consistency penalty
        consistency_penalty = (
            self.behavioral_penalty_calculator.calculate_consistency_penalty()
        )
        self._last_reward_components["consistency_penalty"] = consistency_penalty

        # Dynamic shaping
        dynamic_shaping_reward = self.dynamic_reward_shaper.shape_reward(
            pnl_reward, current_price, step, pnl
        )
        self._last_reward_components["dynamic_shaping_reward"] = dynamic_shaping_reward

        # Portfolio correlation bonus
        portfolio_correlation_bonus = self._calculate_portfolio_correlation_bonus(
            pnl, portfolio_value_delta
        )
        self._last_reward_components[
            "portfolio_correlation_bonus"
        ] = portfolio_correlation_bonus

        # Combine rewards
        reward = (
            pnl_reward
            + position_penalty
            + hold_penalty
            + consistency_penalty
            + dynamic_shaping_reward
            + portfolio_correlation_bonus
        )
        self._last_reward_components["total_reward"] = reward
        return reward

    def _calculate_base_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculates a base reward value before stage-specific adjustments.
        This method can be a placeholder or a simple PnL calculation.
        """
        # For now, let's use a simple PnL-based reward as the base
        return self._calculate_pnl_reward(pnl, 1.0)

    def _calculate_risk_management_reward(
        self,
        action: int,
        pnl: float,
        position: float,
        atr_normalised: float,
        portfolio_return: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        observation: Optional[np.ndarray],
    ) -> float:
        """Stage: Risk management reward with unrealized loss penalty."""
        self._record_action(action)
        self._last_reward_components = {"stage": "risk_management"}

        # 1. Calculate base reward from PnL and other factors
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
            observation,
        )
        self._last_reward_components["base_reward"] = base_reward

        # 2. Apply common trading bonuses for BUY/SELL actions
        if action in [ACTION_BUY, ACTION_SELL]:
            base_reward = self._calculate_base_trading_reward(
                base_reward, position, effective_max_position
            )
        self._last_reward_components["base_trading_reward"] = base_reward

        # 3. Calculate and apply unrealized loss penalty
        unrealized_loss_penalty = self.unrealized_loss_penalty_calculator.calculate(
            pnl, position
        )
        if unrealized_loss_penalty < 0:
            self._last_reward_components[
                "unrealized_loss_penalty"
            ] = unrealized_loss_penalty

        total_reward = base_reward + unrealized_loss_penalty
        self._last_reward_components["total_reward"] = total_reward

        self.logger.debug(
            f"Risk management reward: base={base_reward:.4f}, "
            f"unrealized_loss_penalty={unrealized_loss_penalty:.4f}, "
            f"total={total_reward:.4f}"
        )

        return total_reward

    def _calculate_opportunity_cost_reward(
        self,
        action: int,
        position: float,
        pnl: float,
        atr_normalised: float,
        portfolio_return: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        observation: Optional[np.ndarray],
    ) -> float:
        """Stage: Opportunity cost reward to penalize inaction when flat."""
        self._record_action(action)
        self._last_reward_components = {"stage": "opportunity_cost"}

        # 1. Calculate base reward
        base_reward = self._calculate_base_reward(
            action,
            atr_normalised,
            portfolio_return,
            position,
            effective_max_position,
            current_price,
            atr,
            pnl,
            observation,
        )
        self._last_reward_components["base_reward"] = base_reward

        # 2. Calculate and apply opportunity cost penalty
        opportunity_cost_penalty = self.opportunity_cost_penalty_calculator.calculate(
            action, position
        )
        if opportunity_cost_penalty < 0:
            self._last_reward_components[
                "opportunity_cost_penalty"
            ] = opportunity_cost_penalty

        total_reward = base_reward + opportunity_cost_penalty
        self._last_reward_components["total_reward"] = total_reward

        self.logger.debug(
            f"Opportunity cost reward: base={base_reward:.4f}, "
            f"penalty={opportunity_cost_penalty:.4f}, "
            f"total={total_reward:.4f}"
        )

        return total_reward
