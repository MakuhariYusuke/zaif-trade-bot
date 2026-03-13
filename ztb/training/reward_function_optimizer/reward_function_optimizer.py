#!/usr/bin/env python3
"""
Reward Function Structure Optimizer

This module provides comprehensive optimization of reward function structures
including parameter tuning, multi-objective optimization, and automated reward design.
"""

import json
import hashlib
import math
import time
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import importlib.util

from ztb.io.json_io import read_json_object, write_json
from ztb.metrics.metrics import calculate_autocorrelation, detect_outliers_iqr
from ztb.training.hyperparameter_optimizer import ParameterSpace
from ztb.utils.logging_utils import get_logger
from ztb.utils.safety import ensure_dict, safe_to_float, safe_to_int

from .components.evaluation_engine import EvaluationEngine

# Import extracted components
from .components.optimization_engine import OptimizationEngine
from .parameter_space import RewardFunctionParameterSpace

logger = get_logger(__name__)

ConfigMap = dict[str, object]
ScoreMap = dict[str, float]
HistoryRecord = dict[str, object]

SAC_HYPERPARAMETER_KEYS: tuple[str, ...] = (
    "learning_rate",
    "batch_size",
    "buffer_size",
    "gamma",
    "tau",
    "ent_coef",
    "reward_scale",
)
SAC_INTEGER_HYPERPARAMETER_KEYS: tuple[str, ...] = ("batch_size", "buffer_size")
REWARD_SCALAR_SETTING_KEYS: tuple[str, ...] = (
    "trading_bonus",
    "trading_bonus_multiplier",
    "base_profit_bonus_atr_coeff",
    "base_profit_bonus_portfolio_coeff",
    "momentum_weight",
    "volatility_weight",
    "time_decay_weight",
)
DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS: dict[str, float | int] = {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 50_000,
    "ent_coef": 0.01,
    "tau": 0.005,
    "gamma": 0.99,
    "reward_scale": 1.0,
}
RISK_MODERATE_DRAWDOWN_THRESHOLD = 0.05
RISK_HIGH_DRAWDOWN_THRESHOLD = 0.15
WIN_RATE_BULL_THRESHOLD = 0.6
WIN_RATE_BEAR_THRESHOLD = 0.4

OPTUNA_AVAILABLE = importlib.util.find_spec("optuna") is not None
if not OPTUNA_AVAILABLE:
    logger.warning("Optuna not available. Reward function optimization will be limited.")
else:
    import optuna

TQDM_AVAILABLE = importlib.util.find_spec("tqdm") is not None
if not TQDM_AVAILABLE:
    logger.warning("tqdm not available. Progress bars will be disabled.")

NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
if not NUMPY_AVAILABLE:
    logger.warning("NumPy not available. Some calculations will be limited.")

@dataclass
class RewardFunctionConfig:
    """Configuration for reward function optimization."""

    stage: str
    parameters: ConfigMap = field(default_factory=dict)
    objectives: list[str] = field(
        default_factory=lambda: ["profit", "sharpe", "win_rate"]
    )
    constraints: ConfigMap = field(default_factory=dict)

@dataclass
class RewardOptimizationResult:
    """Result of reward function optimization."""

    best_config: RewardFunctionConfig
    best_scores: ScoreMap
    optimization_history: list[HistoryRecord] = field(default_factory=list)
    optimization_time: float = 0.0
    convergence_info: ConfigMap = field(default_factory=dict)

class RewardFunctionOptimizer:
    """
    Optimizer for reward function structures and parameters.

    This class provides:
    - Parameter optimization for existing reward functions
    - Multi-objective optimization (profit, risk, consistency)
    - Automated reward function design
    - Cross-validation across market conditions
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize reward function optimizer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "configs/reward_optimization.json"
        self.logger = get_logger("ztb.optimization.reward")

        # Load configuration if exists
        self.config = self._load_config()

        # Shared parameter-space manager (single source of truth)
        self.parameter_space_manager = RewardFunctionParameterSpace()

        # Default parameter spaces for different reward stages
        self.parameter_spaces = self._define_parameter_spaces()

        # Optimization history
        self.optimization_history: deque[HistoryRecord] = deque(maxlen=1000)

        # Evaluation cache for robustness (bounded to prevent memory growth)
        self.max_evaluation_cache_size = max(
            32, safe_to_int(self.config.get("evaluation_cache_max_size", 512), 512)
        )
        self.evaluation_cache: dict[str, ScoreMap] = {}
        self._evaluation_cache_order: deque[str] = deque()

        # Dynamic weighting system
        self.dynamic_weights = {
            "market_regime": "neutral",  # neutral, bull, bear, volatile, sideways
            "performance_trend": "stable",  # improving, declining, stable
            "risk_level": "moderate",  # low, moderate, high
        }

        # Console output settings
        self.verbose = True
        self.show_progress = TQDM_AVAILABLE
        self.show_detailed_scores = True

        # Evaluation settings for robustness
        self.max_retries = 3
        self.retry_delay = 1.0

        # Initialize extracted components
        self.optimization_engine = OptimizationEngine()
        self.evaluation_engine = EvaluationEngine()

    def _build_evaluation_cache_key(
        self, params: ConfigMap, objectives: list[str]
    ) -> str:
        payload = {
            "params": {str(k): params[k] for k in sorted(params.keys())},
            "objectives": sorted(str(obj) for obj in objectives),
        }
        payload_json = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        payload_bytes = payload_json.encode("utf-8")
        return hashlib.sha1(payload_bytes).hexdigest()

    def _normalize_scores(self, raw_scores: object, objectives: list[str]) -> ScoreMap:
        score_map = ensure_dict(raw_scores)
        normalized: ScoreMap = {}
        for objective in objectives:
            normalized[objective] = safe_to_float(score_map.get(objective, 0.0), 0.0)
        # Keep commonly referenced diagnostics if present.
        for key in ("max_drawdown", "total_trades", "avg_trade_return"):
            if key in score_map:
                normalized[key] = safe_to_float(score_map.get(key, 0.0), 0.0)
        return normalized

    def _evaluate_reward_params(self, params: ConfigMap) -> ScoreMap:
        """Evaluate reward parameters through the backtest config bridge."""
        return self.run_backtest_evaluation(self.create_backtest_config(params))

    @staticmethod
    def _extract_profit_bonus_multipliers(params: ConfigMap) -> list[float]:
        """Extract BUY/SELL/HOLD multipliers with safe defaults."""
        return [
            safe_to_float(params.get("profit_bonus_multiplier_buy", 1.0), 1.0),
            safe_to_float(params.get("profit_bonus_multiplier_sell", 1.0), 1.0),
            safe_to_float(params.get("profit_bonus_multiplier_hold", 1.0), 1.0),
        ]

    @staticmethod
    def _extract_reward_inputs_from_settings(
        reward_settings: ConfigMap,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """Normalize reward settings used by synthetic evaluation."""
        profit_weight = safe_to_float(reward_settings.get("profit_weight", 1.0), 1.0)
        risk_weight = safe_to_float(reward_settings.get("risk_weight", 1.0), 1.0)
        consistency_weight = safe_to_float(
            reward_settings.get("consistency_weight", 1.0), 1.0
        )

        multipliers_obj = reward_settings.get("profit_bonus_multipliers", [1.0, 1.0, 1.0])
        multipliers = multipliers_obj if isinstance(multipliers_obj, list) else [1.0, 1.0, 1.0]
        padded = [1.0, 1.0, 1.0]
        for i in range(min(3, len(multipliers))):
            padded[i] = safe_to_float(multipliers[i], 1.0)
        profit_mult_buy, profit_mult_sell, profit_mult_hold = padded

        momentum_weight = safe_to_float(reward_settings.get("momentum_weight", 0.0), 0.0)
        volatility_weight = safe_to_float(
            reward_settings.get("volatility_weight", 0.0), 0.0
        )

        return (
            profit_weight,
            risk_weight,
            consistency_weight,
            profit_mult_buy,
            profit_mult_sell,
            profit_mult_hold,
            momentum_weight,
            volatility_weight,
        )

    @staticmethod
    def _split_sac_and_reward_params(
        params: ConfigMap,
    ) -> tuple[ConfigMap, ConfigMap]:
        sac_params: ConfigMap = {}
        reward_params: ConfigMap = {}
        for key, value in params.items():
            if key in SAC_HYPERPARAMETER_KEYS:
                sac_params[key] = value
            else:
                reward_params[key] = value
        return sac_params, reward_params

    def _extract_reward_settings_from_params(self, params: ConfigMap) -> ConfigMap:
        reward_settings: ConfigMap = {}
        if any(k.startswith("profit_bonus_multiplier_") for k in params):
            reward_settings["profit_bonus_multipliers"] = (
                self._extract_profit_bonus_multipliers(params)
            )
        for key in REWARD_SCALAR_SETTING_KEYS:
            if key in params:
                reward_settings[key] = params[key]
        reward_settings.update(
            {
                k: v
                for k, v in params.items()
                if k.endswith("_weight") and k not in SAC_HYPERPARAMETER_KEYS
            }
        )
        return reward_settings

    def _apply_sac_hyperparameters(self, config: ConfigMap, params: ConfigMap) -> None:
        sac_hyperparameters = ensure_dict(config.get("sac_hyperparameters"))
        if not sac_hyperparameters:
            sac_hyperparameters = {}

        for key, default_value in DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS.items():
            sac_hyperparameters.setdefault(key, default_value)

        for key in SAC_HYPERPARAMETER_KEYS:
            if key not in params:
                continue
            raw_value = params[key]
            if key in SAC_INTEGER_HYPERPARAMETER_KEYS:
                fallback_int = safe_to_int(
                    sac_hyperparameters.get(key, DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS[key]),
                    safe_to_int(DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS[key], 1),
                )
                normalized_int = safe_to_int(raw_value, fallback_int)
                if normalized_int > 0:
                    sac_hyperparameters[key] = normalized_int
                continue

            fallback_float = safe_to_float(
                sac_hyperparameters.get(key, DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS[key]),
                safe_to_float(DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS[key], 0.0),
            )
            normalized_float = safe_to_float(raw_value, fallback_float)
            if normalized_float > 0:
                sac_hyperparameters[key] = normalized_float

        config["sac_hyperparameters"] = sac_hyperparameters

    @staticmethod
    def _extract_sac_inputs_from_config(
        config: ConfigMap,
    ) -> tuple[float, int, int, float, float, float, float]:
        sac_hyperparameters = ensure_dict(config.get("sac_hyperparameters"))
        learning_rate = safe_to_float(
            sac_hyperparameters.get("learning_rate", 3e-4),
            3e-4,
        )
        batch_size = max(
            1,
            safe_to_int(
                sac_hyperparameters.get("batch_size", 256),
                256,
            ),
        )
        buffer_size = max(
            1,
            safe_to_int(
                sac_hyperparameters.get("buffer_size", 50_000),
                50_000,
            ),
        )
        gamma = safe_to_float(sac_hyperparameters.get("gamma", 0.99), 0.99)
        tau = safe_to_float(sac_hyperparameters.get("tau", 0.005), 0.005)
        ent_coef = safe_to_float(sac_hyperparameters.get("ent_coef", 0.01), 0.01)
        reward_scale = safe_to_float(sac_hyperparameters.get("reward_scale", 1.0), 1.0)

        return learning_rate, batch_size, buffer_size, gamma, tau, ent_coef, reward_scale

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _extract_numeric_metric(
        score_maps: list[ConfigMap], key: str
    ) -> list[float]:
        values: list[float] = []
        for score_map in score_maps:
            if key in score_map:
                values.append(safe_to_float(score_map.get(key, 0.0), 0.0))
        return values

    @staticmethod
    def _classify_risk_level(avg_drawdown: float) -> str:
        if avg_drawdown >= RISK_HIGH_DRAWDOWN_THRESHOLD:
            return "high"
        if avg_drawdown >= RISK_MODERATE_DRAWDOWN_THRESHOLD:
            return "moderate"
        return "low"

    @staticmethod
    def _classify_market_regime_from_win_rate(avg_win_rate: float) -> str:
        if avg_win_rate > WIN_RATE_BULL_THRESHOLD:
            return "bull"
        if avg_win_rate < WIN_RATE_BEAR_THRESHOLD:
            return "bear"
        return "neutral"

    @classmethod
    def _compute_sac_adjustment_factors(
        cls,
        learning_rate: float,
        batch_size: int,
        buffer_size: int,
        gamma: float,
        tau: float,
        ent_coef: float,
        reward_scale: float,
    ) -> tuple[float, float, float]:
        log_lr_distance = abs(math.log10(max(learning_rate, 1e-8)) - math.log10(3e-4))
        lr_factor = 1.0 - min(log_lr_distance, 2.0) * 0.08

        batch_distance = abs(math.log2(max(float(batch_size), 1.0)) - math.log2(256.0))
        batch_factor = 1.0 - min(batch_distance, 4.0) * 0.015

        buffer_distance = abs(
            math.log10(max(float(buffer_size), 1.0)) - math.log10(50_000.0)
        )
        buffer_factor = 1.0 - min(buffer_distance, 2.0) * 0.02

        gamma_factor = 1.0 - min(abs(gamma - 0.99), 0.1) * 1.0
        tau_factor = 1.0 - min(abs(tau - 0.005), 0.02) * 10.0
        entropy_factor = 1.0 - min(abs(ent_coef - 0.01), 0.09) * 2.0
        reward_scale_factor = 1.0 + cls._clamp((reward_scale - 1.0) * 0.04, -0.08, 0.08)

        profit_factor = cls._clamp(
            lr_factor
            * batch_factor
            * buffer_factor
            * gamma_factor
            * tau_factor
            * entropy_factor
            * reward_scale_factor,
            0.75,
            1.2,
        )
        risk_factor = cls._clamp(
            1.0
            + (1.0 - lr_factor) * 0.5
            + (1.0 - gamma_factor) * 0.4
            + (reward_scale_factor - 1.0) * 0.3,
            0.8,
            1.35,
        )
        consistency_factor = cls._clamp(
            (batch_factor + buffer_factor + gamma_factor + tau_factor) / 4.0,
            0.8,
            1.1,
        )
        return profit_factor, risk_factor, consistency_factor

    def _store_evaluation_cache(self, cache_key: str, scores: ScoreMap) -> None:
        if cache_key not in self.evaluation_cache:
            self._evaluation_cache_order.append(cache_key)
        self.evaluation_cache[cache_key] = scores
        while len(self._evaluation_cache_order) > self.max_evaluation_cache_size:
            oldest = self._evaluation_cache_order.popleft()
            self.evaluation_cache.pop(oldest, None)

    def _update_dynamic_weights_from_history(self):
        """
        Update dynamic weights based on optimization history and performance trends.
        This enables adaptive weighting during the optimization process.
        """
        if len(self.optimization_history) < 5:
            return  # Not enough history for meaningful updates

        recent_trials = list(self.optimization_history)[-10:]  # Last 10 trials
        recent_scores = [ensure_dict(trial.get("scores")) for trial in recent_trials]

        # Calculate performance trends
        profit_scores = self._extract_numeric_metric(recent_scores, "profit")
        if len(profit_scores) >= 3:
            # Simple trend analysis
            recent_avg = sum(profit_scores[-3:]) / 3
            older_avg = sum(profit_scores[:-3]) / max(1, len(profit_scores) - 3)

            if recent_avg > older_avg * 1.05:  # 5% improvement
                self.dynamic_weights["performance_trend"] = "improving"
            elif recent_avg < older_avg * 0.95:  # 5% decline
                self.dynamic_weights["performance_trend"] = "declining"
            else:
                self.dynamic_weights["performance_trend"] = "stable"

        # Estimate risk level from drawdown patterns
        drawdown_scores = self._extract_numeric_metric(recent_scores, "max_drawdown")
        if drawdown_scores:
            avg_drawdown = sum(drawdown_scores) / len(drawdown_scores)
            self.dynamic_weights["risk_level"] = self._classify_risk_level(avg_drawdown)

        # Market regime estimation (simplified - could be enhanced with actual market data)
        win_rates = self._extract_numeric_metric(recent_scores, "win_rate")
        if win_rates:
            avg_win_rate = sum(win_rates) / len(win_rates)
            self.dynamic_weights["market_regime"] = (
                self._classify_market_regime_from_win_rate(avg_win_rate)
            )

        self.logger.debug(
            f"Updated dynamic weights from history: {self.dynamic_weights}"
        )

    def _robust_evaluate_parameters(
        self,
        params: ConfigMap,
        objectives: list[str],
        trial_number: int,
        evaluation_function: Callable[[ConfigMap], ScoreMap] | None = None,
    ) -> ScoreMap:
        """
        Robust parameter evaluation with retry logic and caching.

        Args:
            params: Parameters to evaluate
            objectives: list of objectives
            trial_number: Current trial number for logging
            evaluation_function: Optional custom evaluation function

        Returns:
            Evaluation scores
        """
        cache_key = self._build_evaluation_cache_key(params, objectives)
        cached = self.evaluation_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if evaluation_function is None:
            evaluation_function = self._evaluate_reward_params

        last_error: Exception | None = None
        for retry in range(self.max_retries):
            try:
                raw_scores = evaluation_function(params)
                normalized = self._normalize_scores(raw_scores, objectives)
                self._store_evaluation_cache(cache_key, normalized)
                return dict(normalized)
            except Exception as exc:  # pragma: no cover - defensive retry path
                last_error = exc
                self.logger.warning(
                    "Evaluation retry %d/%d failed for trial %d: %s",
                    retry + 1,
                    self.max_retries,
                    trial_number,
                    exc,
                )
                if retry + 1 < self.max_retries:
                    time.sleep(self.retry_delay)

        if last_error is not None:
            self.logger.error(
                "Evaluation failed after retries for trial %d: %s",
                trial_number,
                last_error,
            )
        fallback = {objective: 0.0 for objective in objectives}
        return fallback

    def set_console_output(
        self,
        verbose: bool = True,
        show_progress: bool = True,
        show_detailed_scores: bool = True,
    ):
        """
        Configure console output settings.

        Args:
            verbose: Enable verbose logging
            show_progress: Show progress bars during optimization
            show_detailed_scores: Show detailed performance scores
        """
        self.verbose = verbose
        self.show_progress = show_progress and TQDM_AVAILABLE
        self.show_detailed_scores = show_detailed_scores

    def _print_header(self, title: str, subtitle: str = None):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"🎯 {title}")
        if subtitle:
            print(f"   {subtitle}")
        print(f"{'='*60}")

    def _print_progress(self, current: int, total: int, message: str = ""):
        """Print progress information."""
        if self.show_progress and TQDM_AVAILABLE:
            return  # Let tqdm handle progress
        else:
            percentage = (current / total) * 100 if total > 0 else 0
            bar_length = 40
            filled_length = int(bar_length * current // total) if total > 0 else 0
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"\r[{bar}] {percentage:.1f}% {message}", end="", flush=True)
            if current == total:
                print()  # New line at completion

    def _print_scores(self, scores: ScoreMap, title: str = "Performance Scores"):
        """Print formatted performance scores."""
        if not self.show_detailed_scores:
            return

        print(f"\n📊 {title}:")
        print("-" * 40)

        # Define score categories and their display properties
        score_categories = {
            "profit": {"icon": "💰", "format": ".4f", "higher_better": True},
            "sharpe": {"icon": "📈", "format": ".4f", "higher_better": True},
            "win_rate": {"icon": "🎯", "format": ".3f", "higher_better": True},
            "max_drawdown": {"icon": "📉", "format": ".4f", "higher_better": False},
            "consistency": {"icon": "⚖️", "format": ".4f", "higher_better": True},
            "total_trades": {"icon": "🔄", "format": "d", "higher_better": None},
            "avg_trade_return": {"icon": "📊", "format": ".4f", "higher_better": True},
        }

        for key, value in scores.items():
            if key in score_categories:
                cat = score_categories[key]
                icon = cat["icon"]
                fmt = cat["format"]
                higher_better = cat["higher_better"]
                numeric_value = safe_to_float(value, 0.0)

                # Color coding based on performance
                if higher_better is True and numeric_value > 0:
                    color = "🟢"
                elif higher_better is False and numeric_value < 0.1:  # Low drawdown is good
                    color = "🟢"
                elif higher_better is False and numeric_value > 0.2:  # High drawdown is bad
                    color = "🔴"
                else:
                    color = "🟡"

                if fmt == "d":
                    formatted_value = str(int(round(numeric_value)))
                else:
                    formatted_value = f"{numeric_value:{fmt}}"

                print(
                    f"  {icon} {key.replace('_', ' ').title()}: {color}{formatted_value}"
                )
            else:
                print(f"  📋 {key}: {value}")

    def _print_optimization_summary(self, result: RewardOptimizationResult):
        """Print comprehensive optimization summary."""
        print("\n🎉 Optimization Completed!")
        print(f"⏱️  Total Time: {result.optimization_time:.2f} seconds")
        print(f"🎯 Best Stage: {result.best_config.stage}")
        print(f"📈 Trials Completed: {len(result.optimization_history)}")

        if result.convergence_info:
            conv = result.convergence_info
            print(f"🏆 Best Trial: #{conv.get('best_trial_number', 'N/A')}")
            best_value = safe_to_float(conv.get("study_best_value", 0.0), 0.0)
            print(f"📊 Study Best Value: {best_value:.4f}")

        self._print_scores(result.best_scores, "Final Best Scores")

        # Show parameter improvements if available
        if len(result.optimization_history) > 1:
            print("\n📈 Optimization Progress:")
            first_entry = ensure_dict(result.optimization_history[0])
            first_scores = ensure_dict(first_entry.get("scores"))
            first_score = safe_to_float(first_scores.get("profit", 0.0), 0.0)
            best_score = safe_to_float(result.best_scores.get("profit", 0.0), 0.0)
            improvement = (
                ((best_score - first_score) / abs(first_score)) * 100
                if first_score != 0
                else 0
            )
            print(f"  Profit Improvement: {improvement:+.2f}%")

    def _handle_error(self, error: Exception, context: str):
        """Handle and display errors gracefully."""
        print(f"\n❌ Error in {context}:")
        print(f"   {str(error)}")
        print(f"   Type: {type(error).__name__}")

        # Provide helpful suggestions
        if "Optuna" in str(error):
            print("   💡 Suggestion: Ensure Optuna is properly installed")
        elif "ParameterSpace" in str(error):
            print("   💡 Suggestion: Check parameter definitions in config")
        elif "FileNotFound" in str(error):
            print("   💡 Suggestion: Verify config file path exists")

        logger.error(f"Error in {context}: {error}", exc_info=True)

    def _define_parameter_spaces(self) -> dict[str, dict[str, ParameterSpace]]:
        """Define parameter spaces for different reward-function stages."""
        return self.parameter_space_manager.get_parameter_spaces()

    def create_parameter_space_from_config(
        self, config: ConfigMap, exploration_range: float = 0.1
    ) -> dict[str, ParameterSpace]:
        """Create an exploration parameter-space around a base config."""
        return self.parameter_space_manager.create_parameter_space_from_config(
            config, exploration_range
        )

    def update_dynamic_weights(
        self,
        market_data: dict[str, object] | None = None,
        performance_history: list[dict[str, float]] | None = None,
    ) -> None:
        """
        Update dynamic weights based on market conditions and performance history.

        Args:
            market_data: Current market data (volatility, trend, etc.)
            performance_history: Recent performance metrics
        """
        # Update market regime
        if market_data:
            volatility = safe_to_float(market_data.get("volatility", 0.5), 0.5)
            trend_strength = safe_to_float(market_data.get("trend_strength", 0.0), 0.0)

            if volatility > 0.8:
                self.dynamic_weights["market_regime"] = "volatile"
            elif trend_strength > 0.7:
                self.dynamic_weights["market_regime"] = "bull"
            elif trend_strength < -0.7:
                self.dynamic_weights["market_regime"] = "bear"
            elif volatility < 0.3:
                self.dynamic_weights["market_regime"] = "sideways"
            else:
                self.dynamic_weights["market_regime"] = "neutral"

        # Update performance trend
        if performance_history and len(performance_history) >= 3:
            recent_scores = [
                safe_to_float(h.get("composite_score", 0.0), 0.0)
                for h in performance_history[-3:]
            ]
            if all(
                recent_scores[i] < recent_scores[i + 1]
                for i in range(len(recent_scores) - 1)
            ):
                self.dynamic_weights["performance_trend"] = "improving"
            elif all(
                recent_scores[i] > recent_scores[i + 1]
                for i in range(len(recent_scores) - 1)
            ):
                self.dynamic_weights["performance_trend"] = "declining"
            else:
                self.dynamic_weights["performance_trend"] = "stable"

        # Update risk level based on recent drawdowns
        if performance_history:
            recent_drawdowns = [
                safe_to_float(h.get("max_drawdown", 0.0), 0.0)
                for h in performance_history[-5:]
            ]
            avg_drawdown = (
                sum(recent_drawdowns) / len(recent_drawdowns) if recent_drawdowns else 0
            )
            self.dynamic_weights["risk_level"] = self._classify_risk_level(avg_drawdown)

        self.logger.info(f"Updated dynamic weights: {self.dynamic_weights}")

    def get_dynamic_objective_weights(self, objectives: list[str]) -> ScoreMap:
        """
        Get dynamic objective weights based on current market and performance conditions.

        Args:
            objectives: list of objectives to weight

        Returns:
            Dictionary of objective weights
        """
        base_weights = {
            "profit": 0.4,
            "sharpe": 0.3,
            "win_rate": 0.2,
            "consistency": 0.1,
        }

        # Adjust weights based on market regime
        regime = self.dynamic_weights["market_regime"]
        if regime == "volatile":
            base_weights["profit"] *= 0.8  # Reduce profit weight in volatile markets
            base_weights["sharpe"] *= 1.3  # Increase risk-adjusted return weight
        elif regime == "bull":
            base_weights["profit"] *= 1.2  # Increase profit weight in bull markets
            base_weights["consistency"] *= 0.9
        elif regime == "bear":
            base_weights["profit"] *= 0.9
            base_weights["consistency"] *= 1.2  # Increase consistency in bear markets

        # Adjust weights based on performance trend
        trend = self.dynamic_weights["performance_trend"]
        if trend == "improving":
            base_weights["profit"] *= 1.1  # Reinforce successful strategies
        elif trend == "declining":
            base_weights["consistency"] *= 1.2  # Focus on stability when declining

        # Adjust weights based on risk level
        risk = self.dynamic_weights["risk_level"]
        if risk == "high":
            base_weights["sharpe"] *= 1.4  # Strongly emphasize risk-adjusted returns
            base_weights["profit"] *= 0.8
        elif risk == "low":
            base_weights["profit"] *= 1.1  # Can afford to take more risk for profit

        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            base_weights = {k: v / total_weight for k, v in base_weights.items()}

        # Return only weights for requested objectives
        return {obj: base_weights.get(obj, 0.0) for obj in objectives}

    def _load_config(self) -> ConfigMap:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                config = ensure_dict(read_json_object(config_file))
                self.logger.info(f"Loaded configuration from: {config_file}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {config_file}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}

    def load_base_config_from_file(self, config_file_path: str) -> ConfigMap:
        """
        Load base configuration from JSON file.

        Args:
            config_file_path: Path to JSON config file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_file_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        config_data = ensure_dict(read_json_object(config_path))

        # Extract parameters from different sections
        parameters = {}

        # Extract SAC hyperparameters
        if "sac_hyperparameters" in config_data:
            parameters.update(ensure_dict(config_data["sac_hyperparameters"]))

        # Extract reward settings
        if "reward_settings" in config_data:
            reward_settings = ensure_dict(config_data["reward_settings"])
            # Only include numeric reward parameters
            for key, value in reward_settings.items():
                if isinstance(value, (int, float)):
                    parameters[key] = value

        # Extract environment parameters if they are numeric
        if "environment" in config_data:
            for key, value in ensure_dict(config_data["environment"]).items():
                if isinstance(value, (int, float)):
                    parameters[key] = value

        # If no structured parameters found, assume the whole file is parameters
        if not parameters:
            if "parameters" in config_data:
                parameters = ensure_dict(config_data["parameters"])
            else:
                # Assume the whole file is parameters
                parameters = dict(config_data)

        stage = str(config_data.get("stage", "profit_optimized"))

        return {"parameters": parameters, "stage": stage, "file_path": str(config_path)}

    def optimize_from_config_file(
        self,
        config_file_path: str,
        exploration_range: float = 0.1,
        n_trials: int = 100,
        objectives: list[str] | None = None,
    ) -> RewardOptimizationResult:
        """
        Optimize reward function parameters starting from existing config file.

        Args:
            config_file_path: Path to JSON config file
            exploration_range: Fraction of current value to explore (±range)
            n_trials: Number of optimization trials
            objectives: list of objectives to optimize

        Returns:
            Optimization result
        """
        # Load base configuration
        try:
            base_config = self.load_base_config_from_file(config_file_path)
        except Exception as e:
            self._handle_error(e, f"loading config file {config_file_path}")
            raise

        # Create parameter space from config
        param_space = self.create_parameter_space_from_config(
            base_config["parameters"], exploration_range
        )

        # Use the stage from config or default
        stage = str(base_config.get("stage", "profit_optimized"))
        objectives = objectives or ["profit", "sharpe", "win_rate"]

        # Print header
        self._print_header(
            "Config-Based Reward Function Optimization",
            f"Starting from {Path(config_file_path).name} | Exploring ±{exploration_range*100:.0f}% range",
        )

        print(f"📁 Config File: {config_file_path}")
        print(f"🎯 Parameters to optimize: {len(param_space)}")
        print(f"📊 Exploration range: ±{exploration_range*100:.0f}%")
        print(f"🎲 Optimization trials: {n_trials}")

        # Temporarily set the parameter space for this optimization
        original_spaces = self.parameter_spaces.copy()
        self.parameter_spaces[stage] = param_space

        try:
            # Run optimization
            result = self.optimize_reward_function(
                stage=stage,
                evaluation_function=self._evaluate_reward_params,
                n_trials=n_trials,
                objectives=objectives,
            )

            return result

        finally:
            # Restore original parameter spaces
            self.parameter_spaces = original_spaces

    def optimize_hyperparameters_from_config(
        self, config_file_path: str, exploration_range: float = 0.1, n_trials: int = 50
    ) -> ConfigMap:
        """
        Optimize SAC hyperparameters starting from existing config file.

        Args:
            config_file_path: Path to JSON config file
            exploration_range: Fraction of current value to explore (±range)
            n_trials: Number of optimization trials

        Returns:
            Dictionary with optimized parameters and performance
        """
        # Load base configuration
        try:
            base_config = self.load_base_config_from_file(config_file_path)
        except Exception as e:
            self._handle_error(e, f"loading config file {config_file_path}")
            raise

        # Split parameters by concern to avoid accidental cross-contamination.
        sac_params, reward_params = self._split_sac_and_reward_params(
            ensure_dict(base_config.get("parameters"))
        )
        if not sac_params:
            sac_params = dict(DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS)
            self.logger.warning(
                "No numeric SAC hyperparameters found in %s; falling back to defaults.",
                config_file_path,
            )

        # Print header
        self._print_header(
            "SAC Hyperparameter Optimization",
            f"Starting from {Path(config_file_path).name} | {len(sac_params)} parameters to optimize",
        )

        print(f"🧠 SAC Parameters: {list(sac_params.keys())}")
        print(
            f"🎯 Reward Parameters: {list(reward_params.keys()) if reward_params else 'None (using defaults)'}"
        )
        print(f"📊 Exploration range: ±{exploration_range*100:.0f}%")
        print(f"🎲 Optimization trials: {n_trials}")

        # Create parameter space for SAC hyperparameters
        sac_param_space = self.create_parameter_space_from_config(
            sac_params, exploration_range
        )
        base_backtest_config = self.create_backtest_config(reward_params)

        # Define objective function for SAC hyperparameter optimization
        def sac_objective(trial):
            """Objective function for SAC hyperparameter optimization."""
            params = self.optimization_engine.sample_parameters_for_trial(
                trial, sac_param_space
            )

            # Reuse fixed reward settings and only apply trial-specific SAC updates.
            try:
                backtest_config = self.create_backtest_config(
                    params,
                    base_config=base_backtest_config,
                )
                scores = self.run_backtest_evaluation(backtest_config)
            except Exception as e:
                self._handle_error(e, f"SAC evaluation (trial {trial.number})")
                return -999.0  # Poor score for failed evaluations

            # Return composite score
            return safe_to_float(scores.get("profit", 0.0), 0.0)

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Run optimization
        optimization_started_at = time.time()
        try:
            study.optimize(sac_objective, n_trials=n_trials)
        except Exception as e:
            self._handle_error(e, "SAC hyperparameter optimization")
            raise

        # Get best parameters and evaluate final performance
        best_params = study.best_params
        best_score = study.best_value

        # Create final backtest config and get full scores
        try:
            final_config = self.create_backtest_config(
                best_params,
                base_config=base_backtest_config,
            )
            final_scores = self.run_backtest_evaluation(final_config)
        except Exception as e:
            self._handle_error(e, "final SAC evaluation")
            final_scores = {"profit": best_score}

        result = {
            "optimized_parameters": best_params,
            "base_parameters": sac_params,
            "optimization_score": best_score,
            "final_scores": final_scores,
            "config_file": config_file_path,
            "exploration_range": exploration_range,
            "n_trials": n_trials,
        }

        # Print results
        print("\n🎉 SAC Hyperparameter Optimization Completed!")
        print(f"⏱️  Optimization Time: {(time.time() - optimization_started_at):.2f} seconds")
        print(f"🏆 Best Score: {best_score:.4f}")
        print("📊 Best Parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")

        self._print_scores(final_scores, "Final Performance Scores")

        self.logger.info("SAC hyperparameter optimization completed")
        self.logger.info(f"Best score: {best_score:.4f}")
        self.logger.info(f"Optimized parameters: {best_params}")

        return result

    def create_backtest_config(
        self,
        reward_params: ConfigMap,
        base_config: ConfigMap | None = None,
    ) -> ConfigMap:
        """
        Create backtest configuration with reward function parameters.

        Args:
            reward_params: Reward function parameters
            base_config: Base configuration to extend

        Returns:
            Complete backtest configuration
        """
        config = deepcopy(base_config) if base_config is not None else {
            "total_timesteps": 50000,
            "eval_freq": 10000,
            "n_eval_episodes": 10,
            "sac_hyperparameters": dict(DEFAULT_SYNTHETIC_SAC_HYPERPARAMETERS),
        }

        normalized_params = ensure_dict(reward_params)
        self._apply_sac_hyperparameters(config, normalized_params)

        reward_settings = ensure_dict(config.get("reward_settings"))
        reward_settings.update(self._extract_reward_settings_from_params(normalized_params))
        config["reward_settings"] = reward_settings
        return config

    def run_backtest_evaluation(self, config: ConfigMap) -> ScoreMap:
        """
        Run actual backtest evaluation with given configuration.

        Args:
            config: Backtest configuration

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # For now, use a simplified evaluation based on reward parameters
            # In production, this would integrate with actual backtesting
            reward_settings = ensure_dict(config.get("reward_settings"))

            (
                profit_weight,
                risk_weight,
                consistency_weight,
                profit_mult_buy,
                profit_mult_sell,
                profit_mult_hold,
                momentum_weight,
                volatility_weight,
            ) = self._extract_reward_inputs_from_settings(reward_settings)
            (
                learning_rate,
                batch_size,
                buffer_size,
                gamma,
                tau,
                ent_coef,
                reward_scale,
            ) = self._extract_sac_inputs_from_config(config)
            (
                sac_profit_factor,
                sac_risk_factor,
                sac_consistency_factor,
            ) = self._compute_sac_adjustment_factors(
                learning_rate=learning_rate,
                batch_size=batch_size,
                buffer_size=buffer_size,
                gamma=gamma,
                tau=tau,
                ent_coef=ent_coef,
                reward_scale=reward_scale,
            )

            # Simulate performance based on parameter combinations
            # Higher profit multipliers generally lead to higher returns but higher risk
            base_profit = (
                profit_mult_buy + profit_mult_sell
            ) * 0.3 + profit_mult_hold * 0.1
            base_profit *= profit_weight

            # Risk increases with high profit multipliers
            base_risk = (profit_mult_buy + profit_mult_sell) * 0.2
            base_risk *= risk_weight

            # Win rate influenced by consistency and balanced multipliers
            win_rate = 0.5 + (consistency_weight - 1.0) * 0.1
            win_rate += (
                1.0 - abs(profit_mult_buy - profit_mult_sell)
            ) * 0.1  # Balance bonus

            # Consistency score
            consistency = (
                consistency_weight * 0.5
                + (1.0 - abs(profit_mult_buy - profit_mult_sell)) * 0.3
            )

            # Advanced components add small bonuses
            advanced_bonus = (momentum_weight + volatility_weight) * 0.05
            base_profit += advanced_bonus

            # SAC hyperparameters influence synthetic score surfaces as secondary factors.
            base_profit *= sac_profit_factor
            base_risk *= sac_risk_factor
            consistency *= sac_consistency_factor
            win_rate += (sac_consistency_factor - 1.0) * 0.1
            win_rate = min(max(win_rate, 0.1), 0.9)

            # Sharpe ratio considers both return and risk
            sharpe = base_profit / max(base_risk, 0.1)

            # Max drawdown increases with risk
            max_drawdown = base_risk * 0.5

            return {
                "profit": float(base_profit),
                "sharpe": float(sharpe),
                "win_rate": float(win_rate),
                "max_drawdown": float(max_drawdown),
                "consistency": float(consistency),
                "total_trades": max(0, int(round(base_profit * 100))),  # Simulated trade count
                "avg_trade_return": float(base_profit / max(base_risk, 0.1)),
            }

        except Exception as e:
            self.logger.warning(f"Backtest evaluation failed: {e}")
            # Return default metrics if evaluation fails
            return {
                "profit": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.5,
                "max_drawdown": 0.1,
                "consistency": 0.5,
                "total_trades": 0,
                "avg_trade_return": 0.0,
            }

    def optimize_reward_function(
        self,
        stage: str,
        evaluation_function: Callable[[ConfigMap], ScoreMap],
        n_trials: int = 100,
        objectives: list[str] | None = None,
        constraints: ConfigMap | None = None,
    ) -> RewardOptimizationResult:
        """
        Optimize reward function parameters for a specific stage.

        Args:
            stage: Reward function stage to optimize
            evaluation_function: Function that evaluates parameter performance
            n_trials: Number of optimization trials
            objectives: list of objectives to optimize
            constraints: Optimization constraints

        Returns:
            Optimization result with best parameters and scores
        """

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for reward function optimization")

        if stage not in self.parameter_spaces:
            raise ValueError(f"Unknown reward stage: {stage}")

        objectives = objectives or ["profit", "sharpe", "win_rate"]
        constraints = constraints or {}

        trial_counter = {"value": 0}

        def robust_eval(params: ConfigMap) -> ScoreMap:
            trial_counter["value"] += 1
            return self._robust_evaluate_parameters(
                params=params,
                objectives=objectives,
                trial_number=trial_counter["value"],
                evaluation_function=evaluation_function,
            )

        # Delegate optimization to OptimizationEngine
        result_dict = self.optimization_engine.optimize(
            stage=stage,
            evaluation_function=robust_eval,
            n_trials=n_trials,
            objectives=objectives,
            constraints=constraints,
            parameter_spaces=self.parameter_spaces,
        )

        history = result_dict.get("optimization_history")
        if isinstance(history, list):
            for record in history:
                self.optimization_history.append(ensure_dict(record))
            self._update_dynamic_weights_from_history()

        # Convert to RewardOptimizationResult
        return RewardOptimizationResult(
            best_config=RewardFunctionConfig(
                stage=result_dict["best_config"]["stage"],
                parameters=result_dict["best_config"]["parameters"],
                objectives=result_dict["best_config"]["objectives"],
                constraints=result_dict["best_config"]["constraints"],
            ),
            best_scores=result_dict["best_scores"],
            optimization_history=result_dict["optimization_history"],
            optimization_time=result_dict["optimization_time"],
            convergence_info=result_dict["convergence_info"],
        )

    def optimize_pareto_front(
        self,
        stage: str,
        n_trials: int = 200,
        objectives: list[str] | None = None,
        constraints: ConfigMap | None = None,
    ) -> list[RewardOptimizationResult]:
        """
        Optimize Pareto front for multi-objective reward function design.

        Args:
            stage: Reward function stage to optimize
            n_trials: Number of optimization trials
            objectives: list of objectives to optimize
            constraints: Optimization constraints

        Returns:
            list of Pareto optimal solutions
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Pareto optimization")

        if stage not in self.parameter_spaces:
            raise ValueError(f"Unknown reward stage: {stage}")

        objectives = objectives or ["profit", "sharpe", "win_rate", "consistency"]
        constraints = constraints or {}

        # Print header
        self._print_header(
            "Pareto Front Optimization",
            f"Optimizing {len(objectives)} objectives for {stage.title()} stage",
        )

        print(f"🎯 Objectives: {objectives}")
        print(f"🎲 Total trials: {n_trials}")
        print(f"📊 Stage: {stage}")

        start_time = time.time()

        def objective(trial):
            """Multi-objective function for Pareto optimization."""
            params = self.optimization_engine.sample_parameters_for_trial(
                trial, self.parameter_spaces[stage]
            )

            # Evaluate parameters using actual backtest
            try:
                backtest_config = self.create_backtest_config(params)
                scores = self.run_backtest_evaluation(backtest_config)
            except Exception as e:
                self._handle_error(e, f"Pareto evaluation (trial {trial.number})")
                # Return poor scores for all objectives
                scores = {objective_name: -999.0 for objective_name in objectives}

            # Store trial information
            trial_info = {
                "trial_number": trial.number,
                "parameters": params.copy(),
                "scores": scores.copy(),
                "timestamp": datetime.now().isoformat(),
            }
            self.optimization_history.append(trial_info)

            # Return multiple objectives for Pareto optimization
            return [
                scores.get(obj, 0.0) if obj != "max_drawdown" else -scores.get(obj, 0.0)
                for obj in objectives
            ]

        # Create Optuna study for multi-objective optimization
        study = optuna.create_study(
            directions=["maximize"] * len(objectives),  # All objectives to maximize
            sampler=optuna.samplers.NSGAIISampler(seed=42),
        )

        # Run optimization with progress tracking
        try:
            if self.show_progress and TQDM_AVAILABLE:
                from tqdm import tqdm

                with tqdm(
                    total=n_trials, desc="🎯 Pareto Optimization", unit="trial"
                ) as pbar:

                    def callback(study, trial):
                        pbar.update(1)
                        if study.best_trials:
                            best_values = study.best_trials[0].values
                            pbar.set_postfix(
                                {
                                    "solutions": len(study.best_trials),
                                    "best_profit": f"{best_values[0]:.4f}"
                                    if len(best_values) > 0
                                    else "N/A",
                                }
                            )

                    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            else:
                study.optimize(objective, n_trials=n_trials)

        except Exception as e:
            self._handle_error(e, "Pareto front optimization")
            raise

        # Extract Pareto optimal solutions
        pareto_solutions = []
        for trial in study.best_trials:
            params = trial.params
            scores = {}
            for i, obj in enumerate(objectives):
                score_value = trial.values[i]
                if obj == "max_drawdown":
                    score_value = -score_value  # Convert back
                scores[obj] = score_value

            solution = RewardOptimizationResult(
                best_config=RewardFunctionConfig(
                    stage=stage,
                    parameters=params,
                    objectives=objectives,
                    constraints=constraints,
                ),
                best_scores=scores,
                optimization_history=[],
                optimization_time=time.time() - start_time,
                convergence_info={
                    "trial_number": trial.number,
                    "pareto_rank": 0,  # All are Pareto optimal
                    "n_objectives": len(objectives),
                },
            )
            pareto_solutions.append(solution)

        # Print results
        print("\n🎉 Pareto Front Optimization Completed!")
        print(f"⏱️  Optimization Time: {time.time() - start_time:.2f} seconds")
        print(f"⭐ Pareto Solutions Found: {len(pareto_solutions)}")
        print(f"🎯 Objectives Optimized: {objectives}")

        if pareto_solutions:
            print("\n🏆 Top Pareto Solution:")
            top_solution = pareto_solutions[0]
            for obj, value in top_solution.best_scores.items():
                print(f"   {obj}: {value:.4f}")
            print(f"📊 Parameters: {top_solution.best_config.parameters}")

        self.logger.info(f"Pareto optimization completed for stage '{stage}'")
        self.logger.info(f"Found {len(pareto_solutions)} Pareto optimal solutions")

        return pareto_solutions

    def auto_select_stage(self, market_data: dict[str, object] | None = None) -> str:
        """
        Automatically select the best optimization stage based on market conditions.

        Args:
            market_data: Current market data (volatility, trend, etc.)

        Returns:
            Recommended stage name
        """
        if not market_data:
            return "balanced_transition"  # Default stage

        volatility = safe_to_float(market_data.get("volatility", 0.5), 0.5)
        trend_strength = safe_to_float(market_data.get("trend_strength", 0.0), 0.0)
        market_phase = str(market_data.get("phase", "neutral"))

        # Decision logic based on market conditions
        if volatility > 0.8:
            return "high_volatility"
        elif trend_strength > 0.7:
            return "bull_market"
        elif trend_strength < -0.7:
            return "bear_market"
        elif volatility < 0.3:
            return "sideways_market"
        elif market_phase == "profit_focused":
            return "profit_optimized"
        elif market_phase == "ultra_profit":
            return "ultra_profit"
        elif market_phase == "trading_intensive":
            return "trading_focused"
        else:
            return "balanced_transition"

    def optimize_adaptive(
        self,
        market_data: dict[str, object] | None = None,
        n_trials: int = 100,
        objectives: list[str] | None = None,
    ) -> RewardOptimizationResult:
        """
        Adaptive optimization that selects the best stage based on market conditions.

        Args:
            market_data: Current market data for stage selection
            n_trials: Number of optimization trials
            objectives: list of objectives to optimize

        Returns:
            Optimization result with automatically selected stage
        """
        # Update dynamic weights based on market data
        self.update_dynamic_weights(market_data)

        # Auto-select the best stage
        selected_stage = self.auto_select_stage(market_data)

        # Print header
        self._print_header(
            "Adaptive Reward Function Optimization",
            f"Auto-selected stage: {selected_stage.title()} based on market conditions",
        )

        print(f"📊 Market Data: {market_data or 'None provided'}")
        print(f"🎯 Selected Stage: {selected_stage}")
        print(f"⚖️  Dynamic Weights: {self.dynamic_weights}")
        print(f"🎲 Optimization Trials: {n_trials}")

        self.logger.info(f"Adaptive optimization selected stage: {selected_stage}")
        self.logger.info(f"Market conditions: {market_data}")
        self.logger.info(f"Dynamic weights: {self.dynamic_weights}")

        # Run optimization with selected stage
        result = self.optimize_reward_function(
            stage=selected_stage,
            evaluation_function=self._evaluate_reward_params,
            n_trials=n_trials,
            objectives=objectives,
        )

        # Add adaptive selection info to result
        result.convergence_info["adaptive_selection"] = {
            "selected_stage": selected_stage,
            "market_data": market_data,
            "dynamic_weights": self.dynamic_weights.copy(),
        }

        print("\n🎉 Adaptive Optimization Completed!")
        print(f"📊 Selected Stage: {selected_stage}")
        best_profit = safe_to_float(result.best_scores.get("profit", 0.0), 0.0)
        print(f"🏆 Best Score: {best_profit:.4f}")

        return result

    def save_optimization_result(
        self,
        result: RewardOptimizationResult,
        output_path: str = "optimization_results/reward_optimization_result.json",
    ) -> None:
        """
        Save optimization result to file.

        Args:
            result: Optimization result to save
            output_path: Path to save result
        """

        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        result_dict = {
            "stage": result.best_config.stage,
            "best_parameters": result.best_config.parameters,
            "best_scores": result.best_scores,
            "objectives": result.best_config.objectives,
            "constraints": result.best_config.constraints,
            "optimization_time": result.optimization_time,
            "convergence_info": result.convergence_info,
            "optimization_history": result.optimization_history,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to file
        write_json(output_file, result_dict, indent=2, ensure_ascii=False)

        self.logger.info(f"Optimization result saved to: {output_file}")

    def load_optimization_result(
        self, input_path: str = "optimization_results/reward_optimization_result.json"
    ) -> RewardOptimizationResult:
        """
        Load optimization result from file.

        Args:
            input_path: Path to load result from

        Returns:
            Loaded optimization result
        """

        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Optimization result file not found: {input_file}")

        result_dict = read_json_object(input_file)

        # Convert back to RewardOptimizationResult
        result = RewardOptimizationResult(
            best_config=RewardFunctionConfig(
                stage=result_dict["stage"],
                parameters=result_dict["best_parameters"],
                objectives=result_dict["objectives"],
                constraints=result_dict["constraints"],
            ),
            best_scores=result_dict["best_scores"],
            optimization_history=result_dict["optimization_history"],
            optimization_time=result_dict["optimization_time"],
            convergence_info=result_dict["convergence_info"],
        )

        self.logger.info(f"Optimization result loaded from: {input_file}")
        return result

    def generate_optimization_report(
        self,
        result: RewardOptimizationResult,
        output_path: str = "optimization_results/reward_optimization_report.md",
    ) -> None:
        """
        Generate comprehensive optimization report.

        Args:
            result: Optimization result to report
            output_path: Path to save report
        """

        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        report = f"""# Reward Function Optimization Report

## Overview
- **Stage**: {result.best_config.stage}
- **Optimization Time**: {result.optimization_time:.2f} seconds
- **Objectives**: {', '.join(result.best_config.objectives)}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Best Parameters
```json
{json.dumps(result.best_config.parameters, indent=2)}
```

## Performance Scores
"""

        for objective, score in result.best_scores.items():
            report += f"- **{objective}**: {safe_to_float(score, 0.0):.4f}\n"

        study_best_value = safe_to_float(
            result.convergence_info.get("study_best_value", 0.0), 0.0
        )

        report += f"""
## Convergence Information
- **Best Trial**: {result.convergence_info.get('best_trial_number', 'N/A')}
- **Total Trials**: {result.convergence_info.get('n_trials', 'N/A')}
- **Study Best Value**: {study_best_value:.4f}

## Optimization History Summary
- **Total Trials**: {len(result.optimization_history)}
- **Parameter Evolution**: Available in optimization result file

## Recommendations
1. Implement the optimized parameters in the reward function
2. Validate performance across different market conditions
3. Consider fine-tuning based on additional objectives
4. Monitor for overfitting to optimization dataset

---
*Generated by RewardFunctionOptimizer*
"""

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        self.logger.info(f"Optimization report generated: {report_file}")

    def analyze_optimization_statistics(
        self, results: list[dict[str, object]]
    ) -> ConfigMap:
        """最適化結果の統計分析を実行"""
        if not results:
            return {}

        # 報酬値の時系列を抽出
        rewards = [safe_to_float(ensure_dict(r).get("reward", 0.0), 0.0) for r in results]

        if len(rewards) < 3:
            return {"insufficient_data": True}

        # 外れ値検出
        outliers = detect_outliers_iqr(rewards)
        outlier_count = sum(outliers)
        outlier_rate = outlier_count / len(outliers)

        # 自己相関係数（最適化の安定性を評価）
        autocorr_1 = (
            calculate_autocorrelation(rewards, lag=1) if len(rewards) > 1 else 0.0
        )

        # 最適化の収束傾向を評価
        first_half = rewards[: len(rewards) // 2]
        second_half = rewards[len(rewards) // 2 :]

        first_half_mean = sum(first_half) / len(first_half)
        second_half_mean = sum(second_half) / len(second_half)
        improvement_trend = second_half_mean - first_half_mean

        return {
            "total_trials": len(results),
            "outlier_count": outlier_count,
            "outlier_rate": outlier_rate,
            "autocorrelation_lag1": autocorr_1,
            "first_half_mean_reward": first_half_mean,
            "second_half_mean_reward": second_half_mean,
            "improvement_trend": improvement_trend,
            "optimization_stability": 1.0
            - abs(autocorr_1),  # 低い相関 = 安定した最適化
        }
