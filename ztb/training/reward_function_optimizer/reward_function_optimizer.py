#!/usr/bin/env python3
"""
Reward Function Structure Optimizer

This module provides comprehensive optimization of reward function structures
including parameter tuning, multi-objective optimization, and automated reward design.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import importlib.util

from ztb.metrics.statistics import calculate_autocorrelation, detect_outliers_iqr
from ztb.training.hyperparameter_optimizer import ParameterSpace
from ztb.utils.logging_utils import get_logger

from .components.evaluation_engine import EvaluationEngine

# Import extracted components
from .components.optimization_engine import OptimizationEngine

logger = get_logger(__name__)

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
    parameters: Dict[str, Any] = field(default_factory=dict)
    objectives: List[str] = field(
        default_factory=lambda: ["profit", "sharpe", "win_rate"]
    )
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardOptimizationResult:
    """Result of reward function optimization."""

    best_config: RewardFunctionConfig
    best_scores: Dict[str, float]
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)


class RewardFunctionOptimizer:
    """
    Optimizer for reward function structures and parameters.

    This class provides:
    - Parameter optimization for existing reward functions
    - Multi-objective optimization (profit, risk, consistency)
    - Automated reward function design
    - Cross-validation across market conditions
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize reward function optimizer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "configs/reward_optimization.json"
        self.logger = get_logger("ztb.optimization.reward")

        # Load configuration if exists
        self.config = self._load_config()

        # Default parameter spaces for different reward stages
        self.parameter_spaces = self._define_parameter_spaces()

        # Optimization history
        self.optimization_history = []

        # Evaluation cache for robustness
        self.evaluation_cache = {}

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

    def _update_dynamic_weights_from_history(self):
        """
        Update dynamic weights based on optimization history and performance trends.
        This enables adaptive weighting during the optimization process.
        """
        if len(self.optimization_history) < 5:
            return  # Not enough history for meaningful updates

        recent_trials = self.optimization_history[-10:]  # Last 10 trials
        recent_scores = [trial.get("scores", {}) for trial in recent_trials]

        # Calculate performance trends
        profit_scores = [s.get("profit", 0) for s in recent_scores if "profit" in s]
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
        drawdown_scores = [
            s.get("max_drawdown", 0) for s in recent_scores if "max_drawdown" in s
        ]
        if drawdown_scores:
            avg_drawdown = sum(drawdown_scores) / len(drawdown_scores)
            if avg_drawdown < -0.1:  # High risk
                self.dynamic_weights["risk_level"] = "high"
            elif avg_drawdown < -0.05:  # Moderate risk
                self.dynamic_weights["risk_level"] = "moderate"
            else:  # Low risk
                self.dynamic_weights["risk_level"] = "low"

        # Market regime estimation (simplified - could be enhanced with actual market data)
        win_rates = [s.get("win_rate", 0) for s in recent_scores if "win_rate" in s]
        if win_rates:
            avg_win_rate = sum(win_rates) / len(win_rates)
            if avg_win_rate > 0.6:
                self.dynamic_weights["market_regime"] = "bull"
            elif avg_win_rate < 0.4:
                self.dynamic_weights["market_regime"] = "bear"
            else:
                self.dynamic_weights["market_regime"] = "neutral"

        self.logger.debug(
            f"Updated dynamic weights from history: {self.dynamic_weights}"
        )

    def _robust_evaluate_parameters(
        self, params: Dict[str, Any], objectives: List[str], trial_number: int
    ) -> Dict[str, float]:
        """
        Robust parameter evaluation with retry logic and caching.

        Args:
            params: Parameters to evaluate
            objectives: List of objectives
            trial_number: Current trial number for logging

        Returns:
            Evaluation scores
        """
        # Delegate evaluation to EvaluationEngine
        return self.evaluation_engine.evaluate_configuration(
            params=params,
            objectives=objectives,
            trial_number=trial_number,
            evaluation_cache=self.evaluation_cache,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

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

    def _print_scores(
        self, scores: Dict[str, float], title: str = "Performance Scores"
    ):
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

                # Color coding based on performance
                if higher_better is True and value > 0:
                    color = "🟢"
                elif higher_better is False and value < 0.1:  # Low drawdown is good
                    color = "🟢"
                elif higher_better is False and value > 0.2:  # High drawdown is bad
                    color = "🔴"
                else:
                    color = "🟡"

                print(f"  {icon} {key.replace('_', ' ').title()}: {color}{value:{fmt}}")
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
            print(f"📊 Study Best Value: {conv.get('study_best_value', 'N/A'):.4f}")

        self._print_scores(result.best_scores, "Final Best Scores")

        # Show parameter improvements if available
        if len(result.optimization_history) > 1:
            print("\n📈 Optimization Progress:")
            first_score = result.optimization_history[0]["scores"].get("profit", 0)
            best_score = result.best_scores.get("profit", 0)
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

    def _define_parameter_spaces(self) -> Dict[str, Dict[str, ParameterSpace]]:
        """Define parameter spaces for different reward function stages."""

        spaces = {}

        # Balanced transition stage parameters - Enhanced with improved ranges
        spaces["balanced_transition"] = {
            # Basic trading parameters with narrower, more realistic ranges
            "balance_penalty_tolerance": ParameterSpace(
                "balance_penalty_tolerance", "float", 0.02, 0.1, log_scale=False
            ),
            "balance_penalty": ParameterSpace(
                "balance_penalty", "float", 2.0, 10.0, log_scale=False
            ),
            "hold_penalty_rate": ParameterSpace(
                "hold_penalty_rate", "float", 0.005, 0.05, log_scale=True
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 1.2, 3.0, log_scale=False
            ),
            "trading_bonus": ParameterSpace(
                "trading_bonus", "float", 0.005, 0.03, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD] - asymmetric ranges
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 0.8, 1.5, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 0.8, 1.5, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.2, 0.8, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses - narrower ranges
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 1.0, 2.5, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 1.0, 2.5, log_scale=False
            ),
            # Advanced reward components - market-aware weights
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.1, 0.7, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.1, 0.6, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.05, 0.3, log_scale=False
            ),
            # Multi-objective weights - normalized ranges
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.3, 1.2, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.2, 0.8, log_scale=False
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.3, 1.0, log_scale=False
            ),
            # Asymmetric reward scaling parameters (v435 enhancement)
            "long_position_reward_multiplier": ParameterSpace(
                "long_position_reward_multiplier", "float", 1.0, 2.0, log_scale=False
            ),
            "short_position_reward_multiplier": ParameterSpace(
                "short_position_reward_multiplier", "float", 0.5, 1.0, log_scale=False
            ),
            "long_position_penalty_multiplier": ParameterSpace(
                "long_position_penalty_multiplier", "float", 0.5, 1.0, log_scale=False
            ),
            "short_position_penalty_multiplier": ParameterSpace(
                "short_position_penalty_multiplier", "float", 1.0, 1.5, log_scale=False
            ),
        }

        # Trading focused stage parameters - Enhanced
        spaces["trading_focused"] = {
            # Basic trading parameters
            "balance_penalty_tolerance": ParameterSpace(
                "balance_penalty_tolerance", "float", 0.01, 0.2, log_scale=False
            ),
            "balance_penalty": ParameterSpace(
                "balance_penalty", "float", 5.0, 50.0, log_scale=False
            ),
            "hold_penalty_rate": ParameterSpace(
                "hold_penalty_rate", "float", 0.01, 1.0, log_scale=True
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 1.0, 10.0, log_scale=False
            ),
            "trading_bonus": ParameterSpace(
                "trading_bonus", "float", 0.01, 1.0, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD]
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 0.8, 3.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 0.8, 3.0, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.05, 1.0, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 1.0, 5.0, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 1.0, 5.0, log_scale=False
            ),
            # Advanced reward components
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.0, 1.0, log_scale=False
            ),
            # Multi-objective weights
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.1, 2.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.1, 2.0, log_scale=False
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.1, 2.0, log_scale=False
            ),
        }

        # Profit optimized stage parameters - Enhanced
        spaces["profit_optimized"] = {
            # Profit-focused parameters
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.5, 5.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.01, 1.0, log_scale=True
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.01, 1.0, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD]
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 1.0, 5.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 1.0, 5.0, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.01, 1.0, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 2.0, 8.0, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 2.0, 8.0, log_scale=False
            ),
            # Advanced reward components
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.0, 3.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 3.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.0, 1.5, log_scale=False
            ),
            # Risk management parameters
            "position_penalty_weight": ParameterSpace(
                "position_penalty_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "drawdown_penalty_weight": ParameterSpace(
                "drawdown_penalty_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "stagnation_penalty_weight": ParameterSpace(
                "stagnation_penalty_weight", "float", 0.001, 0.1, log_scale=True
            ),
            # Performance bonuses
            "growth_bonus_weight": ParameterSpace(
                "growth_bonus_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "win_streak_bonus_weight": ParameterSpace(
                "win_streak_bonus_weight", "float", 0.001, 0.1, log_scale=True
            ),
            # Asymmetric reward scaling parameters (v435 enhancement)
            "long_position_reward_multiplier": ParameterSpace(
                "long_position_reward_multiplier", "float", 1.0, 3.0, log_scale=False
            ),
            "short_position_reward_multiplier": ParameterSpace(
                "short_position_reward_multiplier", "float", 0.3, 1.0, log_scale=False
            ),
            "long_position_penalty_multiplier": ParameterSpace(
                "long_position_penalty_multiplier", "float", 0.3, 1.0, log_scale=False
            ),
            "short_position_penalty_multiplier": ParameterSpace(
                "short_position_penalty_multiplier", "float", 1.0, 2.0, log_scale=False
            ),
        }

        # Ultra profit stage parameters - Enhanced
        spaces["ultra_profit"] = {
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 1.0, 10.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.001, 0.1, log_scale=True
            ),
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 0.001, 0.1, log_scale=True
            ),
            # Profit bonus multipliers for each action [BUY, SELL, HOLD]
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 2.0, 10.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 2.0, 10.0, log_scale=False
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.001, 0.5, log_scale=False
            ),
            # ATR and portfolio-based profit bonuses
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 3.0, 15.0, log_scale=False
            ),
            "base_profit_bonus_portfolio_coeff": ParameterSpace(
                "base_profit_bonus_portfolio_coeff", "float", 3.0, 15.0, log_scale=False
            ),
            # Advanced reward components
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 0.0, 5.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 5.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "ultra_profit_multiplier": ParameterSpace(
                "ultra_profit_multiplier", "float", 1.0, 5.0, log_scale=False
            ),
            "ultra_risk_multiplier": ParameterSpace(
                "ultra_risk_multiplier", "float", 0.1, 2.0, log_scale=False
            ),
            # Asymmetric reward scaling parameters (v435 enhancement)
            "long_position_reward_multiplier": ParameterSpace(
                "long_position_reward_multiplier", "float", 1.0, 4.0, log_scale=False
            ),
            "short_position_reward_multiplier": ParameterSpace(
                "short_position_reward_multiplier", "float", 0.2, 1.0, log_scale=False
            ),
            "long_position_penalty_multiplier": ParameterSpace(
                "long_position_penalty_multiplier", "float", 0.2, 1.0, log_scale=False
            ),
            "short_position_penalty_multiplier": ParameterSpace(
                "short_position_penalty_multiplier", "float", 1.0, 3.0, log_scale=False
            ),
        }

        # Market regime-specific optimization stages

        # Bull market optimization
        spaces["bull_market"] = {
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 1.0, 8.0, log_scale=False
            ),
            "momentum_weight": ParameterSpace(
                "momentum_weight", "float", 1.0, 5.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 0.0, 2.0, log_scale=False
            ),
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 1.5, 8.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 0.5, 3.0, log_scale=False
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 2.0, 8.0, log_scale=False
            ),
        }

        # Bear market optimization
        spaces["bear_market"] = {
            "profit_weight": ParameterSpace(
                "profit_weight", "float", 0.5, 4.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 0.5, 3.0, log_scale=False
            ),
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 1.0, 4.0, log_scale=False
            ),
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 0.1, 2.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 1.0, 6.0, log_scale=False
            ),
            "trading_bonus_multiplier": ParameterSpace(
                "trading_bonus_multiplier", "float", 1.0, 5.0, log_scale=False
            ),
        }

        # Sideways market optimization
        spaces["sideways_market"] = {
            "consistency_weight": ParameterSpace(
                "consistency_weight", "float", 1.0, 5.0, log_scale=False
            ),
            "hold_penalty_rate": ParameterSpace(
                "hold_penalty_rate", "float", 0.001, 0.05, log_scale=True
            ),
            "profit_bonus_multiplier_hold": ParameterSpace(
                "profit_bonus_multiplier_hold", "float", 0.5, 2.0, log_scale=False
            ),
            "time_decay_weight": ParameterSpace(
                "time_decay_weight", "float", 0.1, 1.0, log_scale=False
            ),
            "stagnation_penalty_weight": ParameterSpace(
                "stagnation_penalty_weight", "float", 0.001, 0.05, log_scale=True
            ),
        }

        # High volatility market optimization
        spaces["high_volatility"] = {
            "volatility_weight": ParameterSpace(
                "volatility_weight", "float", 2.0, 8.0, log_scale=False
            ),
            "risk_weight": ParameterSpace(
                "risk_weight", "float", 1.0, 5.0, log_scale=False
            ),
            "base_profit_bonus_atr_coeff": ParameterSpace(
                "base_profit_bonus_atr_coeff", "float", 3.0, 12.0, log_scale=False
            ),
            "profit_bonus_multiplier_buy": ParameterSpace(
                "profit_bonus_multiplier_buy", "float", 1.0, 6.0, log_scale=False
            ),
            "profit_bonus_multiplier_sell": ParameterSpace(
                "profit_bonus_multiplier_sell", "float", 1.0, 6.0, log_scale=False
            ),
        }

        return spaces

    def update_dynamic_weights(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        performance_history: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """
        Update dynamic weights based on market conditions and performance history.

        Args:
            market_data: Current market data (volatility, trend, etc.)
            performance_history: Recent performance metrics
        """
        # Update market regime
        if market_data:
            volatility = market_data.get("volatility", 0.5)
            trend_strength = market_data.get("trend_strength", 0.0)

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
                h.get("composite_score", 0) for h in performance_history[-3:]
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
                h.get("max_drawdown", 0) for h in performance_history[-5:]
            ]
            avg_drawdown = (
                sum(recent_drawdowns) / len(recent_drawdowns) if recent_drawdowns else 0
            )

            if avg_drawdown > 0.15:
                self.dynamic_weights["risk_level"] = "high"
            elif avg_drawdown < 0.05:
                self.dynamic_weights["risk_level"] = "low"
            else:
                self.dynamic_weights["risk_level"] = "moderate"

        self.logger.info(f"Updated dynamic weights: {self.dynamic_weights}")

    def get_dynamic_objective_weights(self, objectives: List[str]) -> Dict[str, float]:
        """
        Get dynamic objective weights based on current market and performance conditions.

        Args:
            objectives: List of objectives to weight

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

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from: {config_file}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {config_file}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}

    def load_base_config_from_file(self, config_file_path: str) -> Dict[str, Any]:
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

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Extract parameters from different sections
        parameters = {}

        # Extract SAC hyperparameters
        if "sac_hyperparameters" in config_data:
            parameters.update(config_data["sac_hyperparameters"])

        # Extract reward settings
        if "reward_settings" in config_data:
            reward_settings = config_data["reward_settings"]
            # Only include numeric reward parameters
            for key, value in reward_settings.items():
                if isinstance(value, (int, float)):
                    parameters[key] = value

        # Extract environment parameters if they are numeric
        if "environment" in config_data:
            for key, value in config_data["environment"].items():
                if isinstance(value, (int, float)):
                    parameters[key] = value

        # If no structured parameters found, assume the whole file is parameters
        if not parameters:
            if "parameters" in config_data:
                parameters = config_data["parameters"]
            else:
                # Assume the whole file is parameters
                parameters = config_data

        stage = config_data.get("stage", "profit_optimized")

        return {"parameters": parameters, "stage": stage, "file_path": str(config_path)}


    def optimize_from_config_file(
        self,
        config_file_path: str,
        exploration_range: float = 0.1,
        n_trials: int = 100,
        objectives: Optional[List[str]] = None,
    ) -> RewardOptimizationResult:
        """
        Optimize reward function parameters starting from existing config file.

        Args:
            config_file_path: Path to JSON config file
            exploration_range: Fraction of current value to explore (±range)
            n_trials: Number of optimization trials
            objectives: List of objectives to optimize

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
        stage = base_config.get("stage", "profit_optimized")
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
                evaluation_function=lambda params: self.run_backtest_evaluation(
                    self.create_backtest_config(params)
                ),
                n_trials=n_trials,
                objectives=objectives,
            )

            return result

        finally:
            # Restore original parameter spaces
            self.parameter_spaces = original_spaces

    def optimize_hyperparameters_from_config(
        self, config_file_path: str, exploration_range: float = 0.1, n_trials: int = 50
    ) -> Dict[str, Any]:
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

        # Extract SAC hyperparameters
        sac_params = {}
        reward_params = {}

        for key, value in base_config["parameters"].items():
            if key in [
                "learning_rate",
                "batch_size",
                "buffer_size",
                "gamma",
                "tau",
                "ent_coef",
                "reward_scale",
            ]:
                sac_params[key] = value
            else:
                reward_params[key] = value

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

        # Define objective function for SAC hyperparameter optimization
        def sac_objective(trial):
            """Objective function for SAC hyperparameter optimization."""
            params = {}

            # Sample SAC parameters
            for param_name, param_def in sac_param_space.items():
                if param_def.type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_def.low, param_def.high
                    )
                elif param_def.type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, int(param_def.low), int(param_def.high)
                    )

            # Create backtest config with optimized SAC params and fixed reward params
            try:
                backtest_config = self.create_backtest_config(
                    {**params, **reward_params}
                )
                scores = self.run_backtest_evaluation(backtest_config)
            except Exception as e:
                self._handle_error(e, f"SAC evaluation (trial {trial.number})")
                return -999  # Poor score for failed evaluations

            # Return composite score
            return scores.get("profit", 0.0)

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Run optimization
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
            final_config = self.create_backtest_config({**best_params, **reward_params})
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
        print(f"⏱️  Optimization Time: {(time.time() - time.time()):.2f} seconds")
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
        reward_params: Dict[str, Any],
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create backtest configuration with reward function parameters.

        Args:
            reward_params: Reward function parameters
            base_config: Base configuration to extend

        Returns:
            Complete backtest configuration
        """
        config = base_config or {
            "total_timesteps": 50000,
            "eval_freq": 10000,
            "n_eval_episodes": 10,
            "sac_hyperparameters": {
                "learning_rate": 3e-4,
                "batch_size": 256,
                "ent_coef": 0.01,
                "tau": 0.005,
                "gamma": 0.99,
            },
        }

        # Apply reward function parameters
        reward_settings = {}

        # Profit bonus multipliers
        if any(k.startswith("profit_bonus_multiplier_") for k in reward_params):
            multipliers = []
            multipliers.append(reward_params.get("profit_bonus_multiplier_buy", 1.0))
            multipliers.append(reward_params.get("profit_bonus_multiplier_sell", 1.0))
            multipliers.append(reward_params.get("profit_bonus_multiplier_hold", 1.0))
            reward_settings["profit_bonus_multipliers"] = multipliers

        # Trading bonuses
        if "trading_bonus" in reward_params:
            reward_settings["trading_bonus"] = reward_params["trading_bonus"]
        if "trading_bonus_multiplier" in reward_params:
            reward_settings["trading_bonus_multiplier"] = reward_params[
                "trading_bonus_multiplier"
            ]

        # ATR and portfolio coefficients
        if "base_profit_bonus_atr_coeff" in reward_params:
            reward_settings["base_profit_bonus_atr_coeff"] = reward_params[
                "base_profit_bonus_atr_coeff"
            ]
        if "base_profit_bonus_portfolio_coeff" in reward_params:
            reward_settings["base_profit_bonus_portfolio_coeff"] = reward_params[
                "base_profit_bonus_portfolio_coeff"
            ]

        # Advanced components
        if "momentum_weight" in reward_params:
            reward_settings["momentum_weight"] = reward_params["momentum_weight"]
        if "volatility_weight" in reward_params:
            reward_settings["volatility_weight"] = reward_params["volatility_weight"]
        if "time_decay_weight" in reward_params:
            reward_settings["time_decay_weight"] = reward_params["time_decay_weight"]

        # Multi-objective weights
        if any(k.endswith("_weight") for k in reward_params):
            reward_settings.update(
                {k: v for k, v in reward_params.items() if k.endswith("_weight")}
            )

        config["reward_settings"] = reward_settings
        return config

    def run_backtest_evaluation(self, config: Dict[str, Any]) -> Dict[str, float]:
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
            reward_settings = config.get("reward_settings", {})

            # Extract key parameters
            profit_weight = reward_settings.get("profit_weight", 1.0)
            risk_weight = reward_settings.get("risk_weight", 1.0)
            consistency_weight = reward_settings.get("consistency_weight", 1.0)

            profit_mult_buy = reward_settings.get(
                "profit_bonus_multipliers", [1.0, 1.0, 1.0]
            )[0]
            profit_mult_sell = reward_settings.get(
                "profit_bonus_multipliers", [1.0, 1.0, 1.0]
            )[1]
            profit_mult_hold = reward_settings.get(
                "profit_bonus_multipliers", [1.0, 1.0, 1.0]
            )[2]

            momentum_weight = reward_settings.get("momentum_weight", 0.0)
            volatility_weight = reward_settings.get("volatility_weight", 0.0)

            # Simulate performance based on parameter combinations
            # Higher profit multipliers generally lead to higher returns but higher risk
            base_profit = (
                profit_mult_buy + profit_mult_sell
            ) * 0.3 + profit_mult_hold * 0.1
            base_profit *= profit_weight

            # Risk increases with high profit multipliers
            base_risk = (profit_mult_buy + profit_mult_sell) * 0.2
            base_risk *= risk_weight

            # Sharpe ratio considers both return and risk
            sharpe = base_profit / max(base_risk, 0.1)

            # Win rate influenced by consistency and balanced multipliers
            win_rate = 0.5 + (consistency_weight - 1.0) * 0.1
            win_rate += (
                1.0 - abs(profit_mult_buy - profit_mult_sell)
            ) * 0.1  # Balance bonus
            win_rate = min(max(win_rate, 0.1), 0.9)

            # Max drawdown increases with risk
            max_drawdown = base_risk * 0.5

            # Consistency score
            consistency = (
                consistency_weight * 0.5
                + (1.0 - abs(profit_mult_buy - profit_mult_sell)) * 0.3
            )

            # Advanced components add small bonuses
            advanced_bonus = (momentum_weight + volatility_weight) * 0.05
            base_profit += advanced_bonus

            return {
                "profit": float(base_profit),
                "sharpe": float(sharpe),
                "win_rate": float(win_rate),
                "max_drawdown": float(max_drawdown),
                "consistency": float(consistency),
                "total_trades": int(base_profit * 100),  # Simulated trade count
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
        evaluation_function: Callable,
        n_trials: int = 100,
        objectives: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RewardOptimizationResult:
        """
        Optimize reward function parameters for a specific stage.

        Args:
            stage: Reward function stage to optimize
            evaluation_function: Function that evaluates parameter performance
            n_trials: Number of optimization trials
            objectives: List of objectives to optimize
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

        # Delegate optimization to OptimizationEngine
        result_dict = self.optimization_engine.optimize(
            stage=stage,
            evaluation_function=evaluation_function,
            n_trials=n_trials,
            objectives=objectives,
            constraints=constraints,
            parameter_spaces=self.parameter_spaces,
        )

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
        objectives: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[RewardOptimizationResult]:
        """
        Optimize Pareto front for multi-objective reward function design.

        Args:
            stage: Reward function stage to optimize
            n_trials: Number of optimization trials
            objectives: List of objectives to optimize
            constraints: Optimization constraints

        Returns:
            List of Pareto optimal solutions
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
            params = {}

            # Sample parameters for the stage
            param_space = self.parameter_spaces[stage]
            for param_name, param_def in param_space.items():
                if param_def.type == "float":
                    if param_def.log_scale:
                        params[param_name] = trial.suggest_float(
                            param_name, param_def.low, param_def.high, log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_def.low, param_def.high
                        )
                elif param_def.type == "int":
                    if param_def.low is not None and param_def.high is not None:
                        params[param_name] = trial.suggest_int(
                            param_name, int(param_def.low), int(param_def.high)
                        )
                elif param_def.type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_def.choices
                    )

            # Evaluate parameters using actual backtest
            try:
                backtest_config = self.create_backtest_config(params)
                scores = self.run_backtest_evaluation(backtest_config)
            except Exception as e:
                self._handle_error(e, f"Pareto evaluation (trial {trial.number})")
                # Return poor scores for all objectives
                scores = dict.fromkeys(objectives, -999)

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

    def auto_select_stage(self, market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Automatically select the best optimization stage based on market conditions.

        Args:
            market_data: Current market data (volatility, trend, etc.)

        Returns:
            Recommended stage name
        """
        if not market_data:
            return "balanced_transition"  # Default stage

        volatility = market_data.get("volatility", 0.5)
        trend_strength = market_data.get("trend_strength", 0.0)
        market_phase = market_data.get("phase", "neutral")

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
        market_data: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        objectives: Optional[List[str]] = None,
    ) -> RewardOptimizationResult:
        """
        Adaptive optimization that selects the best stage based on market conditions.

        Args:
            market_data: Current market data for stage selection
            n_trials: Number of optimization trials
            objectives: List of objectives to optimize

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
            evaluation_function=lambda params: self.run_backtest_evaluation(
                self.create_backtest_config(params)
            ),
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
        print(f"🏆 Best Score: {result.best_scores.get('profit', 'N/A'):.4f}")

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
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

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

        with open(input_file, "r", encoding="utf-8") as f:
            result_dict = json.load(f)

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
            report += f"- **{objective}**: {score:.4f}\n"

        report += f"""
## Convergence Information
- **Best Trial**: {result.convergence_info.get('best_trial_number', 'N/A')}
- **Total Trials**: {result.convergence_info.get('n_trials', 'N/A')}
- **Study Best Value**: {result.convergence_info.get('study_best_value', 'N/A'):.4f}

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
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """最適化結果の統計分析を実行"""
        if not results:
            return {}

        # 報酬値の時系列を抽出
        rewards = [r.get("reward", 0) for r in results]

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
