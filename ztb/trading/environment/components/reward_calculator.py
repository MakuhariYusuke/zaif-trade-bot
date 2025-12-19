"""
Reward Calculator - Handles reward calculation logic for trading environment.

This module separates the complex reward calculation logic from the main environment class.
Refactored to follow SOLID principles with component-based architecture.
"""

# mypy: disable-error-code=literal-required

import inspect
import logging
from typing import Dict, List, Optional, Union

import numpy as np

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
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
from .reward.balance_curriculum import BalanceCurriculumManager
from .reward.opportunity_cost_penalty_calculator import OpportunityCostPenaltyCalculator
from .reward.trend_detector import TrendDetector
from .reward.unrealized_loss_penalty_calculator import UnrealizedLossPenaltyCalculator
from .rewards.base import RewardContext
from .rewards.confidence_penalty import ConfidencePenaltyReward
from .rewards.forced_balance import ForcedBalanceReward
from .rewards.pnl_focused import PnlFocusedReward
from .rewards.profit_optimized import ProfitOptimizedReward
from .rewards.smart_incentive import SmartIncentiveReward
from .rewards.trading_focused import TradingFocusedReward
from .rewards.ultra_profit import UltraProfitReward
from .signal_integrator import SignalIntegrator


# Add get_logger function for compatibility
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    import logging

    return logging.getLogger(name)


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
        self.structured_logger = StructuredLogger(
            "ztb.trading.environment.reward", json_format=True
        )

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
        self._recent_actions: List[int] = []  # Reset this list as well
        self.mtf_scheduler = None

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
            "behavior_optimization.action_balance_target", 0.45
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
        # Prefer explicit setting in config.behavior_optimization if present (dict),
        # fall back to reward_settings otherwise.
        behavior_opt = getattr(self.config, "behavior_optimization", {}) or {}
        if isinstance(behavior_opt, dict) and "balance_penalty" in behavior_opt:
            try:
                self.balance_penalty = float(behavior_opt["balance_penalty"])
            except (TypeError, ValueError):
                self.balance_penalty = self.get_setting_float(
                    "behavior_optimization.balance_penalty", 1.0
                )
        else:
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

    def _initialize_components(self, config: EnvironmentConfig) -> None:
        """Initialize all sub-components."""
        self.market_regime_detector = self._init_market_regime_detector()
        self.dynamic_reward_shaper = self._init_dynamic_reward_shaper()
        self.signal_integrator = self._init_signal_integrator(config)
        self.asymmetric_reward_scaler = AsymmetricRewardScaler(env_config=config)
        # Create TrendDetector for behavioral integration
        self.trend_detector = TrendDetector(
            min_samples=self.get_setting_int("trend_detector.min_samples", 20)
        )
        # Behavioral penalty calculator (integrates TrendDetector and MTF manager)
        self.behavioral_penalty_calculator = BehavioralPenaltyCalculator(
            config=config, trend_detector=self.trend_detector
        )
        # Lightweight MTF weight manager (Layer 5 foundation)
        from .reward.mtf_weight_manager import MTFWeightManager

        self.mtf_weight_manager = MTFWeightManager(config)
        self.unrealized_loss_penalty_calculator = UnrealizedLossPenaltyCalculator(
            reward_settings=self.reward_settings
        )
        self.opportunity_cost_penalty_calculator = OpportunityCostPenaltyCalculator(
            reward_settings=self.reward_settings
        )

        # Initialize optional Balance Curriculum Manager
        curriculum_learning = getattr(config, "curriculum_learning", {}) or {}
        curriculum_enabled = bool(curriculum_learning.get("enabled", False))
        curriculum_auto = bool(curriculum_learning.get("auto_progression", True))
        curriculum_emergency = bool(curriculum_learning.get("emergency_revert", True))
        self.curriculum_manager = BalanceCurriculumManager(
            config=config,
            enabled=curriculum_enabled,
            auto_progression=curriculum_auto,
            emergency_revert=curriculum_emergency,
        )

        # v449: Strategy Pattern Components
        self.forced_balance_reward = ForcedBalanceReward()
        self.smart_incentive_reward = SmartIncentiveReward()
        self.pnl_focused_reward = PnlFocusedReward()
        self.ultra_profit_reward = UltraProfitReward()
        self.trading_focused_reward = TradingFocusedReward()
        self.profit_optimized_reward = ProfitOptimizedReward()
        self.confidence_penalty_reward = ConfidencePenaltyReward()

        self._setup_mtf_scheduler_integration(config)

    def _setup_mtf_scheduler_integration(self, config: EnvironmentConfig) -> None:
        """Attach MTFScheduler if behavior.mtf.weight_optimizer.enabled is configured."""
        behavior = getattr(config, "behavior", None)
        if not isinstance(behavior, dict):
            return
        mtf_section = behavior.get("mtf")
        if not isinstance(mtf_section, dict):
            return
        optimizer_cfg = mtf_section.get("weight_optimizer")
        if not isinstance(optimizer_cfg, dict):
            return
        if not optimizer_cfg.get("enabled"):
            return
        base_config = optimizer_cfg.get("base_config")
        if not base_config:
            self.logger.warning(
                "MTF weight optimizer enabled but base_config missing; skipping scheduler setup"
            )
            return

        def _get_int(key: str, default: int) -> int:
            value = optimizer_cfg.get(key, default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _get_float_optional(key: str):
            value = optimizer_cfg.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _get_int_optional(key: str):
            value = optimizer_cfg.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        out_dir = str(optimizer_cfg.get("out_dir", "config/v448/mtf_candidates"))
        candidates = _get_int("candidates", 10)
        per_seed = _get_int("per_seed", 3)
        timesteps = _get_int("timesteps", 2000)
        seed = _get_int("seed", 42)
        strategy = str(optimizer_cfg.get("strategy", "random"))
        gate_composite_score = _get_float_optional("gate_composite_score")
        gate_min_reports = _get_int_optional("gate_min_reports")

        try:
            from ztb.training.reward_function_optimizer.mtf_scheduler import (
                MTFScheduler,
                MTFSchedulerConfig,
            )
        except Exception as exc:  # pragma: no cover - import guard for environments without trainer deps
            self.logger.warning(
                "Unable to import MTFScheduler; skipping integration: %s", exc
            )
            return

        try:
            scheduler_config = MTFSchedulerConfig(
                base_config=str(base_config),
                out_dir=out_dir,
                candidates=candidates,
                per_seed=per_seed,
                timesteps=timesteps,
                strategy=strategy,
                seed=seed,
                gate_composite_score=gate_composite_score,
                gate_min_reports=gate_min_reports,
            )
            scheduler = MTFScheduler(self.mtf_weight_manager, scheduler_config)
        except Exception as exc:
            self.logger.warning("Failed to initialize MTFScheduler: %s", exc)
            return

        stage_filter = optimizer_cfg.get("stage_filter")
        if isinstance(stage_filter, str):
            stage_filter = [stage_filter]
        elif isinstance(stage_filter, list):
            stage_filter = [str(item) for item in stage_filter]
        else:
            stage_filter = None
        dry_run = bool(optimizer_cfg.get("dry_run", False))

        try:
            callback = scheduler.create_stage_change_callback(
                stage_filter=stage_filter, dry_run=dry_run
            )
            self.curriculum_manager.add_stage_change_listener(callback)
        except Exception as exc:
            self.logger.warning(
                "Failed to attach MTFScheduler stage-change listener: %s", exc
            )
            return

        self.mtf_scheduler = scheduler
        self.logger.info(
            "MTF scheduler enabled (dry_run=%s, stage_filter=%s, gates comp=%s reports=%s)",
            dry_run,
            stage_filter or "*",
            gate_composite_score,
            gate_min_reports,
        )

    def _init_market_regime_detector(self) -> MarketRegimeDetector:
        self.logger.debug("Initializing MarketRegimeDetector")
        cfg = getattr(self.config, "regime_detection_config", {}) or {}
        use_relative = bool(cfg.get("use_relative", False))
        reference_window = int(cfg.get("reference_window", 1000))
        percentile_threshold = float(cfg.get("percentile_threshold", 0.8))
        return MarketRegimeDetector(
            use_relative=use_relative,
            reference_window=reference_window,
            percentile_threshold=percentile_threshold,
        )

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

    def _setup_logging(self) -> None:
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
                extra={"old_level": self._current_log_level, "new_level": new_level},
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

    def _record_action(self, action: int) -> int:
        """
        Record action in behavioral penalty calculator and sync action counts.

        Args:
            action: Action taken (ACTION_HOLD=0, ACTION_BUY=1, ACTION_SELL=-1; legacy 2=SELL)

        Returns:
            Normalized action value (one of ACTION_* constants)
        """
        # If the behavior calculator has no recent actions but this RewardCalculator
        # has a non-empty _action_counts (e.g., tests set it directly), populate the
        # behavioral deque so that subsequent calculations use expected counts.
        try:
            if (
                hasattr(self.behavioral_penalty_calculator, "recent_actions")
                and getattr(self.behavioral_penalty_calculator, "recent_actions")
                is not None
                and len(self.behavioral_penalty_calculator.recent_actions) == 0
                and sum(self._action_counts) > 0
            ):
                from collections import deque

                maxlen = getattr(
                    self.behavioral_penalty_calculator.recent_actions, "maxlen", None
                )
                # Build a representative deque from current counts
                # NOTE: _action_counts is [HOLD, BUY, SELL] counts. Use ACTION_* values.
                arr = []
                arr.extend([ACTION_HOLD] * max(0, self._action_counts[0]))
                arr.extend([ACTION_BUY] * max(0, self._action_counts[1]))
                arr.extend([ACTION_SELL] * max(0, self._action_counts[2]))
                if maxlen:
                    arr = arr[-maxlen:]
                self.behavioral_penalty_calculator.recent_actions = deque(
                    arr, maxlen=maxlen
                )
        except Exception:
            pass

        self.behavioral_penalty_calculator.record_action(action)
        # Sync action counts with behavioral calculator's recent counts
        self._action_counts = self.behavioral_penalty_calculator._get_recent_counts()
        # Return the action as-is (HeavyTradingEnv normalizes actions via ActionExecutor).
        return action

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
            "forced_balance.penalty.value_small_deviation", 1.0
        )
        penalty_medium = self.get_setting_float(
            "forced_balance.penalty.value_medium_deviation", 2.5
        )
        penalty_large = self.get_setting_float(
            "forced_balance.penalty.value_large_deviation", 5.0
        )
        penalty_very_large = self.get_setting_float(
            "forced_balance.penalty.value_very_large_deviation", 10.0
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

    def reset(self) -> None:
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

        # Reset logging counters
        self._curriculum_log_counter = 0
        self._forced_balance_log_counter = 0

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
        # Reset recent action buffer as well
        self._recent_actions = []

    def reset_episode_state(self) -> None:
        """Alias for resetting episode-level state (compatibility with tests)."""
        self.reset()

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
    ) -> Optional[Union[int, float, bool, str, dict, list, RewardSettings]]:
        """Get nested setting value using dot notation."""
        keys = key.split(".")
        value: Union[
            int, float, bool, str, dict, list, RewardSettings, None
        ] = self.reward_settings

        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                elif hasattr(value, k):
                    value = getattr(value, k)
                else:
                    value = None
                    break
        except (KeyError, TypeError, AttributeError):
            value = None

        # If not found, check custom_reward_params dict in reward_settings for direct key
        if value is None:
            try:
                if hasattr(self.reward_settings, "custom_reward_params") and isinstance(
                    self.reward_settings.custom_reward_params, dict
                ):
                    return self.reward_settings.custom_reward_params.get(key)
            except Exception:
                pass

        # If still not found in reward_settings, try to read from config dicts (e.g., behavior_optimization)
        if value is None:
            parts = key.split('.')
            cfg_val = getattr(self.config, parts[0], None)
            if isinstance(cfg_val, dict):
                try:
                    for p in parts[1:]:
                        if cfg_val is None:
                            break
                        cfg_val = cfg_val.get(p)
                    if cfg_val is not None:
                        return cfg_val
                except Exception:
                    pass

        return value

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
        continuous_action_value: Optional[float] = None,
        trade_pnl: float = 0.0,
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
        continuous_action_value: Optional[float] = None,
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values
            trade_pnl: Realized PnL from executed trades at this step (0.0 if no trade).

        Returns:
            Calculated reward value
        """
        # Dynamic log level evaluation
        self._evaluate_dynamic_logging(step)

        self._last_reward_components = {}  # Reset at the beginning of each calculation

        # Record the action for behavioral analysis BEFORE calculating penalties
        self._record_action(action)

        # PnL source selection: step-based (mark-to-market) vs trade-based (realized).
        # This is primarily for training alignment (the backtest portfolio value is unaffected).
        pnl_mode = self.get_setting_str("pnl_mode", "step").strip().lower()
        step_pnl = float(pnl)
        trade_pnl_value = float(trade_pnl)
        step_pnl_weight = self.get_setting_float("step_pnl_weight", 1.0)
        trade_pnl_weight = self.get_setting_float("trade_pnl_weight", 1.0)
        trade_pnl_apply = self.get_setting_str("trade_pnl_apply", "always").strip().lower()

        eps_for_mode = self.get_setting_float("eps", 1e-8)
        close_event = (
            abs(float(old_position)) > eps_for_mode
            and (
                abs(float(position)) <= eps_for_mode
                or (float(old_position) * float(position) < 0.0)
            )
        )

        trade_component = trade_pnl_value * trade_pnl_weight
        if trade_pnl_apply in {"close", "on_close", "close_only"} and not close_event:
            trade_component = 0.0

        if pnl_mode in {"trade", "trade_only", "realized", "realized_only"}:
            pnl = trade_component
        elif pnl_mode in {"hybrid", "hybrid_close"}:
            pnl = (step_pnl * step_pnl_weight) + trade_component
        else:
            pnl = step_pnl

        self._last_reward_components["pnl_mode"] = pnl_mode
        self._last_reward_components["pnl_step"] = step_pnl
        self._last_reward_components["pnl_trade"] = trade_pnl_value
        self._last_reward_components["pnl_effective"] = float(pnl)
        self._last_reward_components["pnl_close_event"] = float(bool(close_event))

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
        # Prefer curriculum manager if enabled, otherwise fallback to config
        curriculum_stage = getattr(self.config, "curriculum_stage", None)
        if hasattr(self, "curriculum_manager") and getattr(
            self.curriculum_manager, "enabled", False
        ):
            # Update manager with latest metrics so it may progress or revert stages
            try:
                self.curriculum_manager.update(
                    step=step,
                    action_counts=self._action_counts,
                    recent_rewards=reward_history,
                    portfolio_values=portfolio_value_history,
                )
            except Exception:
                # Avoid breaking reward flow due to curriculum errors
                self.logger.exception("Failed to update curriculum manager")
            curriculum_stage = self.curriculum_manager.get_current_stage()
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
            "action_discovery": self._calculate_action_discovery_reward,
            "forced_balance": self._calculate_forced_balance_reward,
            "smart_incentive": self._calculate_smart_incentive_reward,
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
            # v449: Added for RewardContext
            "portfolio_value": portfolio_value,
            "transaction_cost": transaction_cost,
            "old_position": old_position,
            "reward_history": reward_history,
            "portfolio_value_history": portfolio_value_history,
        }
        self._previous_portfolio_value = portfolio_value

        # Filter arguments for the specific method being called
        sig = inspect.signature(reward_method)  # type: ignore
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_kwargs:
            valid_args = method_args
        else:
            valid_args = {k: v for k, v in method_args.items() if k in sig.parameters}

        # Calculate the base reward for the current stage
        base_reward = reward_method(**valid_args)  # type: ignore

        # v454: Confidence Penalty (Inverse Confidence Paradox)
        # Penalize high confidence actions that result in loss
        # Use component-based implementation
        confidence_penalty_context = RewardContext(
            action=action,
            atr_normalised=atr_normalised,
            portfolio_return=portfolio_return,
            position=position,
            effective_max_position=effective_max_position,
            current_price=current_price,
            atr=atr,
            pnl=pnl,
            reward_scaling=reward_scaling,
            observation=observation,
            step=step,
            portfolio_value=portfolio_value,
            transaction_cost=transaction_cost,
            old_position=old_position,
            reward_history=reward_history,
            portfolio_value_history=portfolio_value_history,
            config=self.config,
            reward_settings=self.reward_settings,
            initial_portfolio_value=self.initial_portfolio_value,
            continuous_action_value=continuous_action_value,
        )

        confidence_penalty = self.confidence_penalty_reward.calculate(
            confidence_penalty_context
        )

        self._last_reward_components["confidence_penalty"] = confidence_penalty
        base_reward += confidence_penalty

        # Apply action bonus directly to reward
        base_reward += action_bonus

        # Record action bonus in components
        self._last_reward_components["action_bonus"] = action_bonus

        # Apply the balance penalty calculated earlier
        base_reward += balance_penalty
        # Apply skewness penalty if available (penalize strong BUY/SELL skews)
        try:
            skew_penalty = (
                self.behavioral_penalty_calculator.calculate_skewness_penalty()
            )
        except Exception:
            skew_penalty = 0.0
        base_reward += skew_penalty
        self._last_reward_components["skew_penalty"] = skew_penalty

        # Balance shaping reward: positive when this action moves distribution towards targets
        try:
            balance_shaping = (
                self.behavioral_penalty_calculator.calculate_balance_shaping(action)
            )
        except Exception:
            balance_shaping = 0.0
        base_reward += balance_shaping
        self._last_reward_components["balance_shaping"] = balance_shaping

        # Action entropy shaping: encouraging diversity in actions
        try:
            entropy_shaping = (
                self.behavioral_penalty_calculator.calculate_action_entropy_shaping()
            )
        except Exception:
            entropy_shaping = 0.0
        base_reward += entropy_shaping
        self._last_reward_components["entropy_shaping"] = entropy_shaping

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
        # Add current MTF weights to telemetry if manager present
        try:
            mtf_w = (
                self.mtf_weight_manager.get_weights()
                if hasattr(self, "mtf_weight_manager")
                else None
            )
            self._last_reward_components["mtf_weights"] = mtf_w
        except Exception:
            self._last_reward_components["mtf_weights"] = None

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
        continuous_action_value: Optional[float] = None,
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

            continuous_action_value: Optional[float] = (None,)
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

            # Store reward components for analysis
            self._last_reward_components = {
                "stage": "simple_reward",
                "pnl": float(pnl),
                "adjusted_pnl": float(adjusted_pnl),
                "base_reward": float(adjusted_pnl * reward_scaling),
                "hold_penalty_applied": action == ACTION_HOLD,
                "trade_bonus_applied": action in [ACTION_BUY, ACTION_SELL],
                "position_change": float(position_change),
                "final_reward": float(reward),
            }

            return reward

        except Exception as e:
            self.logger.error(f"RewardCalculator failed, using simple reward: {e}")
            self._last_reward_components = {
                "stage": "simple_reward_error",
                "error": str(e),
            }
            return 0.0

    def _build_reward_context(self, **kwargs) -> RewardContext:
        """Helper to build RewardContext from kwargs."""
        action = kwargs.get("action")
        return RewardContext(
            action=action,
            current_price=kwargs.get("current_price", 0.0),
            position=kwargs.get("position", 0.0),
            portfolio_value=kwargs.get("portfolio_value", 0.0),
            atr=kwargs.get("atr", 1.0),
            transaction_cost=kwargs.get("transaction_cost", 0.0),
            reward_scaling=kwargs.get("reward_scaling", 1.0),
            pnl=kwargs.get("pnl", 0.0),
            old_position=kwargs.get("old_position", 0.0),
            step=kwargs.get("step", 0),
            observation=kwargs.get("observation"),
            reward_history=kwargs.get("reward_history", []),
            portfolio_value_history=kwargs.get("portfolio_value_history", []),
            config=self.config,
            reward_settings=self.reward_settings,
            action_counts=self._action_counts,
            target_ratios=self.behavioral_penalty_calculator.get_target_ratios()
            if hasattr(self, "behavioral_penalty_calculator")
            else {},
            behavioral_penalty_calculator=getattr(
                self, "behavioral_penalty_calculator", None
            ),
            market_regime_detector=getattr(self, "market_regime_detector", None),
            dynamic_reward_shaper=getattr(self, "dynamic_reward_shaper", None),
        )

    def _calculate_forced_balance_reward(self, **kwargs) -> float:
        """
        Stage: Forced balance reward that encourages corrective actions toward configured targets.
        Delegates to ForcedBalanceReward component.
        """
        # Sync RewardCalculator's counts with BehavioralPenaltyCalculator sliding-window counts
        try:
            counts = self.behavioral_penalty_calculator._get_recent_counts()
            # Only override if recent counts are present; preserve manually-set counts for tests
            if sum(counts) > 0:
                self._action_counts = counts
        except Exception:
            pass

        action = kwargs.get("action")
        self._record_action(action)

        # Build Context
        context = self._build_reward_context(**kwargs)

        # If global balance penalty is disabled, the forced_balance stage should be neutral.
        if getattr(self, "balance_penalty", 1.0) == 0.0:
            self._last_reward_components["stage"] = "forced_balance"
            self._last_reward_components["base_reward"] = 0.0
            return 0.0

        reward = self.forced_balance_reward.calculate(context)

        # Update _last_reward_components from component details
        if hasattr(self.forced_balance_reward, "last_reward_details"):
            self._last_reward_components.update(
                self.forced_balance_reward.last_reward_details
            )

        # Apply scaling specific to this stage
        forced_balance_scaling = self.get_setting_float("forced_balance.scaling", 1.0)
        reward *= forced_balance_scaling
        self._last_reward_components["scaled_reward"] = reward

        return reward

    def _calculate_action_discovery_reward(
        self,
        action: int,
        pnl: float,
        reward_scaling: float,
        continuous_action_value: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Stage: Action Discovery - encourage exploration and taking actions.

        This stage intentionally minimizes transaction costs and rewards
        the *act* of taking an action and direction correctness rather
        than requiring immediate profit. Used in early curriculum stages
        to overcome conservative HOLD bias.
        """
        self._record_action(action)
        self._last_reward_components = {"stage": "action_discovery"}

        # Use continuous magnitude if available to reward stronger signals
        magnitude = 1.0
        if continuous_action_value is not None:
            magnitude = abs(float(continuous_action_value))

        # Direction correctness: use sign of pnl (proxy for correctness)
        direction_multiplier = 0.0
        if pnl > 0:
            direction_multiplier = 1.0
        elif pnl < 0:
            direction_multiplier = -0.5

        base_reward = magnitude * direction_multiplier
        # Do not subtract transaction cost in discovery stage (encourage actions)
        discovery_scale = self.get_setting_float("action_discovery.scale", 1.0)
        final_reward = base_reward * discovery_scale * reward_scaling
        self._last_reward_components["base_reward"] = base_reward
        self._last_reward_components["final_reward"] = final_reward
        return final_reward

    def _calculate_smart_incentive_reward(self, **kwargs) -> float:
        """
        Stage: Smart Incentive reward.
        Delegates to SmartIncentiveReward component.
        """
        action = kwargs.get("action")
        self._record_action(action)

        context = self._build_reward_context(**kwargs)
        reward = self.smart_incentive_reward.calculate(context)

        self._last_reward_components = {
            "stage": "smart_incentive",
            "base_reward": reward,
        }
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
        portfolio_value: float,
        transaction_cost: float,
        old_position: float,
        **kwargs,
    ) -> float:
        """Stage: Trading-focused reward that heavily penalizes HOLD and encourages trading."""
        self._record_action(action)
        self._last_reward_components = {"stage": "trading_focused"}

        # Delegate to TradingFocusedReward component
        context = RewardContext(
            action=action,
            atr_normalised=atr_normalised,
            portfolio_return=portfolio_return,
            position=position,
            effective_max_position=effective_max_position,
            current_price=current_price,
            atr=atr,
            pnl=pnl,
            reward_scaling=reward_scaling,
            portfolio_value=portfolio_value,
            transaction_cost=transaction_cost,
            old_position=old_position,
            observation=kwargs.get("observation"),
            step=kwargs.get("step", 0),
            reward_history=kwargs.get("reward_history", []),
            portfolio_value_history=kwargs.get("portfolio_value_history", []),
            config=self.config,
            reward_settings=self.reward_settings,
            action_counts=self._action_counts,
            initial_portfolio_value=self.initial_portfolio_value,
        )
        return self.trading_focused_reward.calculate(context)

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
        portfolio_value: float,
        transaction_cost: float,
        old_position: float,
        **kwargs,
    ) -> float:
        """Stage: Profit-optimized reward that maximizes profitable trading while minimizing losses."""
        self._record_action(action)
        self._last_reward_components = {"stage": "profit_optimized"}

        # Delegate to ProfitOptimizedReward component
        context = RewardContext(
            action=action,
            atr_normalised=atr_normalised,
            portfolio_return=portfolio_return,
            position=position,
            effective_max_position=effective_max_position,
            current_price=current_price,
            atr=atr,
            pnl=pnl,
            reward_scaling=reward_scaling,
            portfolio_value=portfolio_value,
            transaction_cost=transaction_cost,
            old_position=old_position,
            observation=kwargs.get("observation"),
            step=kwargs.get("step", 0),
            reward_history=kwargs.get("reward_history", []),
            portfolio_value_history=kwargs.get("portfolio_value_history", []),
            config=self.config,
            reward_settings=self.reward_settings,
            action_counts=self._action_counts,
            recent_actions=self._recent_actions,
        )
        return self.profit_optimized_reward.calculate(context)

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
        portfolio_value: float,
        transaction_cost: float,
        old_position: float,
        **kwargs,
    ) -> float:
        """Simplified ultra-profit reward that focuses on basic trading principles."""
        # Delegate to UltraProfitReward component
        context = RewardContext(
            action=action,
            atr_normalised=atr_normalised,
            portfolio_return=portfolio_return,
            position=position,
            effective_max_position=effective_max_position,
            current_price=current_price,
            atr=atr,
            pnl=pnl,
            reward_scaling=reward_scaling,
            portfolio_value=portfolio_value,
            transaction_cost=transaction_cost,
            old_position=old_position,
            observation=kwargs.get("observation"),
            step=kwargs.get("step", 0),
            reward_history=kwargs.get("reward_history", []),
            portfolio_value_history=kwargs.get("portfolio_value_history", []),
            config=self.config,
            reward_settings=self.reward_settings,
            action_counts=self._action_counts,
        )
        return self.ultra_profit_reward.calculate(context)

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
        portfolio_value: float,
        transaction_cost: float,
        old_position: float,
        reward_history: List[float],
        portfolio_value_history: List[float],
    ) -> float:
        """Stage 2: PnL-focused reward with trend analysis."""
        # Delegate to PnlFocusedReward component
        context = RewardContext(
            action=action,
            atr_normalised=atr_normalised,
            portfolio_return=portfolio_return,
            position=position,
            effective_max_position=effective_max_position,
            current_price=current_price,
            atr=atr,
            pnl=pnl,
            reward_scaling=reward_scaling,
            observation=observation,
            step=step,
            portfolio_value=portfolio_value,
            transaction_cost=transaction_cost,
            old_position=old_position,
            reward_history=reward_history,
            portfolio_value_history=portfolio_value_history,
            config=self.config,
            reward_settings=self.reward_settings,
            action_counts=self._action_counts,
        )
        return self.pnl_focused_reward.calculate(context)

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
                action=case["action"],  # type: ignore
                current_price=case["current_price"],  # type: ignore
                position=case["position"],  # type: ignore
                portfolio_value=case["portfolio_value"],  # type: ignore
                atr=case["atr"],  # type: ignore
                transaction_cost=10.0,
                reward_scaling=10.0,  # Scale rewards for testing
                pnl=case["pnl"],  # type: ignore
                old_position=case["old_position"],  # type: ignore
                step=case["step"],  # type: ignore
                observation=None,
                reward_history=[],
                portfolio_value_history=[case["portfolio_value"]] * 30,  # type: ignore  # type: ignore
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
