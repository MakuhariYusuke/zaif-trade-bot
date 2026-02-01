#!/usr/bin/env python3
"""Modernized base classes for binary-search hyperparameter optimization."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ztb.trading.environment.components.rewards.utils import RewardUtils

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv
else:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise ImportError(
            "Failed to import required RL libraries. Please ensure stable-baselines3 and sb3-contrib are installed. "
            f"Error: {e}"
        ) from e

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL  # noqa: E402
from ztb.trading.environment.environment import HeavyTradingEnv  # noqa: E402
from ztb.trading.environment.utils.config import EnvironmentConfig  # noqa: E402
from ztb.training.config.ppo_config import PPOConfig, get_ppo_config  # noqa: E402
from ztb.utils.cli_common import CLIFormatter  # noqa: E402
from ztb.io.data_loader import DataLoader


class TrainingCallback(BaseCallback):
    """Capture episode statistics and action distribution for quick experiments."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.actions_taken: list[int] = []
        self.current_episode_actions: list[int] = []

    def _on_step(self) -> bool:  # noqa: D401 (inherit docstring)
        if "actions" in self.locals:
            actions = self.locals["actions"]
            if isinstance(actions, (list, tuple)) and actions:
                action = int(actions[0])
            elif isinstance(actions, np.ndarray):
                action = int(actions.reshape(-1)[0])
            else:
                action_array = np.asarray(actions).reshape(-1)
                action = int(action_array[0]) if action_array.size else ACTION_HOLD
            self.current_episode_actions.append(action)

        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            if "episode" in info:
                episode_info = info["episode"]
                reward = float(episode_info.get("r", 0.0))
                length = int(episode_info.get("l", 0))

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.actions_taken.extend(self.current_episode_actions)
                self.current_episode_actions = []

                if len(self.episode_rewards) % 10 == 0 and self.verbose > 0:
                    recent_avg = float(np.mean(self.episode_rewards[-10:]))
                    logger.info(
                        "Episode %d: reward=%.4f length=%d (avg10=%.4f)",
                        len(self.episode_rewards),
                        reward,
                        length,
                        recent_avg,
                    )

        return True

    def get_training_stats(self) -> Dict[str, Union[float, int]]:
        if not self.episode_rewards:
            return {
                "avg_reward": 0.0,
                "reward_std": 0.0,
                "best_reward": 0.0,
                "worst_reward": 0.0,
                "episode_count": 0,
            }

        rewards = np.asarray(self.episode_rewards, dtype=np.float32)
        return {
            "avg_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "best_reward": float(np.max(rewards)),
            "worst_reward": float(np.min(rewards)),
            "episode_count": len(self.episode_rewards),
        }

    def get_action_distribution(
        self,
    ) -> Dict[str, Union[int, float, Dict[str, float], Dict[str, int]]]:
        if not self.actions_taken:
            return {
                "hold_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "total_actions": 0,
                "hold_pct": 0.0,
                "buy_pct": 0.0,
                "sell_pct": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "action_counts": {},
                "action_percentages": {},
            }

        counts = Counter(int(action) for action in self.actions_taken)
        total_actions = sum(counts.values())

        probabilities = np.array(list(counts.values()), dtype=np.float64)
        probabilities /= probabilities.sum() if probabilities.sum() else 1.0
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        max_entropy = math.log(float(len(probabilities))) if total_actions else 1.0
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        percentages = {
            str(action): (count / total_actions) * 100.0 if total_actions else 0.0
            for action, count in counts.items()
        }

        hold_count = counts.get(ACTION_HOLD, 0)
        buy_count = counts.get(ACTION_BUY, 0)
        sell_count = counts.get(ACTION_SELL, 0)

        return {
            "hold_count": hold_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_actions": total_actions,
            "hold_pct": percentages.get(str(ACTION_HOLD), 0.0),
            "buy_pct": percentages.get(str(ACTION_BUY), 0.0),
            "sell_pct": percentages.get(str(ACTION_SELL), 0.0),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "action_counts": {str(action): count for action, count in counts.items()},
            "action_percentages": percentages,
        }


@dataclass
class TrainingRunResult:
    """Aggregate outcome for a single hyperparameter evaluation."""

    parameter_value: Union[int, float]
    score: float
    stats: Dict[str, Union[int, float]]
    action_distribution: Dict[str, Any]
    total_timesteps: int
    model_path: Optional[str]
    elapsed_seconds: float
    iteration: int
    timestamp: str
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class HyperparameterOptimizer(ABC):
    """Shared utilities for the binary-search hyperparameter scripts."""

    enable_warmup_grid: bool
    enable_integer_neighbor_warmup: bool
    enable_deviation_probes: bool
    enable_refinement: bool

    warmup_quantiles: Tuple[float, ...]
    deviation_probe_fraction: float
    deviation_score_margin: float
    refine_span_fraction: float
    refine_tolerance_multiplier: float

    cache_precision: int = 12
    search_tolerance: float = 1e-3
    min_score_improvement: float = 1e-3
    entropy_weight: float = 0.5
    stability_weight: float = 0.5
    target_action_pct: float = 33.3

    def __init__(self, project_root: Optional[Path] = None) -> None:
        super().__init__()
        from ztb.utils.path_utils import get_project_root

        self.project_root = project_root or get_project_root()
        self.data_path = self.project_root / "ml-dataset-enhanced.csv"

        self.random_seed = 42
        self.training_row_limit: Optional[int] = 250_000
        self.stream_batch_size = 512
        self.env_max_features = 128

        self.tensorboard_root = self.project_root / "tensorboard" / "binary_search"
        self.models_dir = self.project_root / "models" / "binary_search"
        self.results_dir = self.project_root / "binary_search_results"
        self.tensorboard_root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.use_progress_bar = True

        self.env_config = EnvironmentConfig.from_dict(
            {
                "reward_scaling": 6.0,
                "transaction_cost": 0.001,
                "max_position_size": 1.0,
                "risk_free_rate": 0.02,
                "feature_set": "full",
                "initial_portfolio_value": 1_000_000.0,
                "curriculum_stage": "simple_portfolio",
                "allow_reverse": False,
                "enable_forced_diversity": False,
                "reward_profit_bonus_multipliers": [1.0, 1.0, 1.0],
            }
        )

        base_overrides: PPOConfig = get_ppo_config(
            {
                "learning_rate": 5e-4,
                "gamma": 0.95,
                "gae_lambda": 0.8,
                "clip_range": 0.3,
                "max_grad_norm": 1.0,
                "target_kl": 0.005,
                "ent_coef": 0.05,
                "normalize_advantage": False,
                "verbose": 1,
            }
        )
        self.ppo_params: Dict[str, Any] = dict(base_overrides)

        self._data_cache: Optional[pd.DataFrame] = None
        self._result_cache: Dict[Tuple[str, int], TrainingRunResult] = {}
        self.history: List[TrainingRunResult] = []

        # Custom search range (overrides get_parameter_range if set)
        self.custom_search_range: Optional[List[Union[int, float]]] = None

        # Sequencing heuristics
        self.enable_warmup_grid = True
        self.enable_integer_neighbor_warmup = True
        self.enable_deviation_probes = True
        self.enable_refinement = True

        self.warmup_quantiles = (0.25, 0.75)
        self.deviation_probe_fraction = 0.25
        self.deviation_score_margin = 0.1
        self.refine_span_fraction = 0.1
        self.refine_tolerance_multiplier = 0.5

    @property
    @abstractmethod
    def parameter_name(self) -> str:
        """Name of the parameter being optimized."""

    @abstractmethod
    def get_parameter_range(self) -> Tuple[float, float]:
        """Get the range (min, max) for binary search."""

    @abstractmethod
    def update_ppo_params(self, value: Union[int, float]) -> None:
        """Update PPO parameters with the test value."""
        # Default implementation using parameter_name
        param_name = self.parameter_name()
        if hasattr(self, "ppo_params") and param_name in self.ppo_params:
            if isinstance(self.ppo_params[param_name], int):
                self.ppo_params[param_name] = int(value)
            else:
                self.ppo_params[param_name] = float(value)
        else:
            raise NotImplementedError(
                f"update_ppo_params not implemented for {param_name}"
            )

    def create_environment(self, **overrides: Dict[str, Any]) -> HeavyTradingEnv:
        config_dict = self.env_config.as_dict()
        if overrides:
            config_dict.update(overrides)
        df = self._load_dataset().copy()
        return HeavyTradingEnv(
            df=df,
            config=config_dict,
            streaming_pipeline=None,
            stream_batch_size=self.stream_batch_size,
            max_features=self.env_max_features,
        )

    def _load_dataset(self) -> pd.DataFrame:
        if self._data_cache is not None:
            return self._data_cache

        df = DataLoader.load_csv_optimized(self.data_path)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if self.training_row_limit is not None:
            df = df.tail(self.training_row_limit)
        self._data_cache = df
        return df

    def _build_model_kwargs(self) -> Dict[str, Any]:
        allowed_keys = {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "clip_range_vf",
            "normalize_advantage",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "verbose",
        }
        return {k: v for k, v in self.ppo_params.items() if k in allowed_keys}

    def create_model(self, env: DummyVecEnv) -> MaskablePPO:
        tensorboard_path = self.tensorboard_root / self.parameter_name
        tensorboard_path.mkdir(parents=True, exist_ok=True)
        return MaskablePPO(
            "MlpPolicy",
            env,
            seed=self.random_seed,
            tensorboard_log=str(tensorboard_path),
            **self._build_model_kwargs(),
        )

    def train_model(
        self, total_timesteps: int = 100_000
    ) -> Tuple[MaskablePPO, TrainingCallback, float]:
        def make_env() -> Any:
            env = self.create_environment()

            def _mask_fn(environment: Any) -> NDArray[np.bool_]:
                return cast(NDArray[np.bool_], environment.get_action_masks())

            masked_env = ActionMasker(env, _mask_fn)
            return cast(Any, Monitor(masked_env))

        vec_env = DummyVecEnv([make_env])

        set_random_seed(self.random_seed)
        model = self.create_model(vec_env)
        callback = TrainingCallback(verbose=int(self.ppo_params.get("verbose", 0)))

        start = time.perf_counter()
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=self.use_progress_bar,
            )
        finally:
            vec_env.close()
            model.env = None
        elapsed = time.perf_counter() - start

        return model, callback, elapsed

    def evaluate_result(
        self, callback: TrainingCallback
    ) -> Tuple[float, Dict[str, Union[int, float]], Dict[str, Any]]:
        stats = callback.get_training_stats()
        action_dist = callback.get_action_distribution()

        if stats["avg_reward"] != 0.0:
            base_score = float(stats["avg_reward"])
        else:
            total_reward = float(sum(callback.episode_rewards))
            episode_count = max(1, int(stats.get("episode_count", 1)))
            base_score = total_reward / episode_count

        def _extract_pct(key: str) -> float:
            value = action_dist.get(key, 0.0)
            if isinstance(value, (int, float)):
                return float(value)
            return 0.0

        hold_pct = _extract_pct("hold_pct")
        buy_pct = _extract_pct("buy_pct")
        sell_pct = _extract_pct("sell_pct")

        balance_penalty = RewardUtils.calculate_balance_deviation_from_percentages(
            [hold_pct, buy_pct, sell_pct], self.target_action_pct
        )
        balance_bonus = max(0.0, 100.0 - balance_penalty) / 100.0

        reward_std = float(stats.get("reward_std", 0.0))
        stability_bonus = max(0.0, 1.0 - reward_std / (abs(base_score) + 1e-6))

        entropy_value = action_dist.get("normalized_entropy", 0.0)
        entropy_bonus = (
            float(entropy_value) if isinstance(entropy_value, (int, float)) else 0.0
        )

        score = base_score
        score += self.entropy_weight * entropy_bonus
        score += self.stability_weight * stability_bonus
        score += balance_bonus

        return score, stats, action_dist

    def _format_parameter_value(self, value: Union[int, float]) -> str:
        if isinstance(value, int):
            return f"{value:d}"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    def save_model(
        self,
        model: MaskablePPO,
        value: Union[int, float],
        iteration: Optional[int] = None,
        note: Optional[str] = None,
    ) -> str:
        param_str = self._format_parameter_value(value)
        parts = [self.parameter_name, param_str]
        if iteration is not None:
            parts.insert(0, f"iter{iteration:02d}")
        if note:
            parts.append(note)
        filename = "_".join(parts) + ".zip"
        model_path = self.models_dir / filename
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info("Model saved to: %s", model_path)
        return str(model_path)

    def print_results(self, result: TrainingRunResult) -> None:
        stats = result.stats
        action_dist = result.action_distribution

        logger.info(
            "=== Training Results for %s=%s ===",
            self.parameter_name,
            self._format_parameter_value(result.parameter_value),
        )
        logger.info(
            "Score: %.6f | Timesteps: %d | Duration: %.1fs",
            result.score,
            result.total_timesteps,
            result.elapsed_seconds,
        )
        logger.info(
            "Avg reward: %.6f (std: %.6f, best: %.6f, worst: %.6f)",
            stats.get("avg_reward", 0.0),
            stats.get("reward_std", 0.0),
            stats.get("best_reward", 0.0),
            stats.get("worst_reward", 0.0),
        )
        logger.info("Episodes: %d", int(stats.get("episode_count", 0)))
        logger.info(
            "Actions — HOLD: %d (%.1f%%) | BUY: %d (%.1f%%) | SELL: %d (%.1f%%) | entropy: %.3f",
            action_dist.get("hold_count", 0),
            action_dist.get("hold_pct", 0.0),
            action_dist.get("buy_count", 0),
            action_dist.get("buy_pct", 0.0),
            action_dist.get("sell_count", 0),
            action_dist.get("sell_pct", 0.0),
            action_dist.get("normalized_entropy", 0.0),
        )
        if result.note:
            logger.info("Stage: %s", result.note)

    def _cache_key(self, value: Union[int, float], timesteps: int) -> Tuple[str, int]:
        if isinstance(value, int):
            return f"int:{value}", timesteps
        return f"float:{float(value).hex()}", timesteps

    def _coerce_parameter_value(self, value: Union[int, float]) -> Union[int, float]:
        min_val, max_val = self.get_parameter_range()
        if isinstance(min_val, int) and isinstance(max_val, int):
            return int(round(float(value)))
        return float(value)

    def _clip_to_range(self, value: Union[int, float]) -> Union[int, float]:
        min_val, max_val = self.get_parameter_range()
        if (
            isinstance(value, float)
            and isinstance(min_val, int)
            and isinstance(max_val, int)
        ):
            value = round(value)
        return max(min_val, min(max_val, value))

    def configure_from_args(self, args: argparse.Namespace) -> None:
        if getattr(args, "no_warmup_grid", False):
            self.enable_warmup_grid = False
        if getattr(args, "no_integer_neighbor_warmup", False):
            self.enable_integer_neighbor_warmup = False
        if getattr(args, "no_deviation_probes", False):
            self.enable_deviation_probes = False
        if getattr(args, "no_refinement", False):
            self.enable_refinement = False
        if getattr(args, "progress_bar", None) is not None:
            self.use_progress_bar = bool(args.progress_bar)
        if getattr(args, "seed", None) is not None:
            self.random_seed = int(args.seed)
        if getattr(args, "search_range", None) is not None:
            # Parse comma-separated values
            range_str = str(args.search_range)
            values: List[Union[int, float]]
            try:
                # Try to parse as integers first
                values = [int(v.strip()) for v in range_str.split(",")]
            except ValueError:
                # Fall back to floats
                values = [float(v.strip()) for v in range_str.split(",")]
            self.custom_search_range = values

    def _is_close(self, a: Union[int, float], b: Union[int, float]) -> bool:
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return math.isclose(
            float(a), float(b), rel_tol=1e-9, abs_tol=self.search_tolerance * 0.1
        )

    def _value_in_range(
        self,
        value: Union[int, float],
        lower: Union[int, float],
        upper: Union[int, float],
    ) -> bool:
        if isinstance(lower, int) and isinstance(upper, int):
            value_int = int(round(float(value)))
            return lower <= value_int <= upper
        value_float = float(value)
        return (
            float(lower) - self.search_tolerance
            <= value_float
            <= float(upper) + self.search_tolerance
        )

    def _unique_candidates(
        self,
        candidates: Iterable[Union[int, float]],
        existing_keys: set[Tuple[str, int]],
        total_timesteps: int,
    ) -> List[Union[int, float]]:
        unique: List[Union[int, float]] = []
        local_seen: set[Tuple[str, int]] = set()
        for candidate in candidates:
            coerced = self._clip_to_range(self._coerce_parameter_value(candidate))
            cache_key = self._cache_key(coerced, total_timesteps)
            if cache_key in existing_keys or cache_key in local_seen:
                continue
            unique.append(coerced)
            local_seen.add(cache_key)
        return unique

    def _generate_warmup_candidates(
        self,
        lower_value: Union[int, float],
        upper_value: Union[int, float],
        is_integer_range: bool,
        existing_keys: set[Tuple[str, int]],
        total_timesteps: int,
    ) -> List[Union[int, float]]:
        if not self.enable_warmup_grid:
            return []

        if is_integer_range:
            lower_int = int(lower_value)
            upper_int = int(upper_value)
            span_int = upper_int - lower_int
            if span_int <= 1:
                return []
            candidates_int: list[int] = []
            if self.enable_integer_neighbor_warmup:
                for neighbor in (lower_int + 1, upper_int - 1):
                    if lower_int < neighbor < upper_int:
                        candidates_int.append(neighbor)
            for quantile in self.warmup_quantiles:
                offset = int(round(span_int * quantile))
                candidate_int = lower_int + offset
                candidate_int = max(lower_int + 1, min(upper_int - 1, candidate_int))
                if (
                    lower_int < candidate_int < upper_int
                    and candidate_int not in candidates_int
                ):
                    candidates_int.append(candidate_int)
            return self._unique_candidates(
                candidates_int, existing_keys, total_timesteps
            )

        lower_float = float(lower_value)
        upper_float = float(upper_value)
        span_float = upper_float - lower_float
        if span_float <= self.search_tolerance:
            return []

        float_candidates: list[float] = []
        for quantile in self.warmup_quantiles:
            float_candidate = lower_float + span_float * quantile
            if self._value_in_range(float_candidate, lower_float, upper_float):
                if not self._is_close(
                    float_candidate, lower_float
                ) and not self._is_close(float_candidate, upper_float):
                    float_candidates.append(float_candidate)
        return self._unique_candidates(float_candidates, existing_keys, total_timesteps)

    def _generate_deviation_probes(
        self,
        midpoint_value: Union[int, float],
        lower_value: Union[int, float],
        upper_value: Union[int, float],
        is_integer_range: bool,
        existing_keys: set[Tuple[str, int]],
        total_timesteps: int,
    ) -> List[Union[int, float]]:
        if not self.enable_deviation_probes:
            return []

        if is_integer_range:
            midpoint_int = int(midpoint_value)
            probes = [midpoint_int - 1, midpoint_int + 1]
            int_candidates = [
                value
                for value in probes
                if self._value_in_range(value, lower_value, upper_value)
            ]
            return self._unique_candidates(
                int_candidates, existing_keys, total_timesteps
            )

        lower_float = float(lower_value)
        upper_float = float(upper_value)
        span_float = upper_float - lower_float
        if span_float <= self.search_tolerance:
            return []

        offset = span_float * self.deviation_probe_fraction
        if offset <= 0:
            return []

        midpoint_float = float(midpoint_value)
        float_candidates: list[float] = []
        for direction in (-1.0, 1.0):
            float_candidate = midpoint_float + direction * offset
            if self._value_in_range(float_candidate, lower_float, upper_float):
                if not self._is_close(float_candidate, midpoint_float):
                    float_candidates.append(float_candidate)
        return self._unique_candidates(float_candidates, existing_keys, total_timesteps)

    def _generate_refinement_candidates(
        self,
        best_value: Union[int, float],
        lower_value: Union[int, float],
        upper_value: Union[int, float],
        is_integer_range: bool,
        existing_keys: set[Tuple[str, int]],
        total_timesteps: int,
    ) -> List[Union[int, float]]:
        if not self.enable_refinement:
            return []

        if is_integer_range:
            best_int = int(best_value)
            candidates = [best_int - 1, best_int + 1]
            filtered = [
                value
                for value in candidates
                if self._value_in_range(value, lower_value, upper_value)
            ]
            return self._unique_candidates(filtered, existing_keys, total_timesteps)

        lower_float = float(lower_value)
        upper_float = float(upper_value)
        span_float = upper_float - lower_float
        offsets = [
            span_float * self.refine_span_fraction,
            self.search_tolerance * self.refine_tolerance_multiplier,
        ]
        best_float = float(best_value)
        float_candidates: list[float] = []
        for offset in offsets:
            if offset <= 0:
                continue
            for direction in (-1.0, 1.0):
                float_candidate = best_float + direction * offset
                if not self._value_in_range(float_candidate, lower_float, upper_float):
                    continue
                if self._is_close(float_candidate, best_float):
                    continue
                float_candidates.append(float_candidate)
        return self._unique_candidates(float_candidates, existing_keys, total_timesteps)

    def _record_result(
        self, result: TrainingRunResult, cache_key: Tuple[str, int]
    ) -> None:
        self.history.append(result)
        self._result_cache[cache_key] = result
        self._append_history_entry(result)

    def _append_history_entry(self, result: TrainingRunResult) -> None:
        history_path = self.results_dir / f"{self.parameter_name}_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(result.to_dict(), ensure_ascii=False))
            fp.write("\n")

    def _log_binary_search_event(self, event: str, payload: Dict[str, Any]) -> None:
        log_entry = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        log_path = self.results_dir / f"{self.parameter_name}_binary_search.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_entry, ensure_ascii=False))
            fp.write("\n")

    def _evaluate_value(
        self,
        value: Union[int, float],
        total_timesteps: int,
        iteration: int,
        note: Optional[str] = None,
        use_cache: bool = True,
    ) -> TrainingRunResult:
        coerced_value = self._clip_to_range(self._coerce_parameter_value(value))
        cache_key = self._cache_key(coerced_value, total_timesteps)

        if use_cache and cache_key in self._result_cache:
            cached = self._result_cache[cache_key]
            logger.info(
                "🔁 Reusing cached result for %s=%s (score=%.6f)",
                self.parameter_name,
                self._format_parameter_value(coerced_value),
                cached.score,
            )
            return cached

        logger.info(
            "=== Training with %s=%s (iteration %d) ===",
            self.parameter_name,
            self._format_parameter_value(coerced_value),
            iteration,
        )

        self.update_ppo_params(coerced_value)
        model, callback, elapsed = self.train_model(total_timesteps)
        score, stats, action_dist = self.evaluate_result(callback)
        model_path = self.save_model(
            model, coerced_value, iteration=iteration, note=note
        )

        result = TrainingRunResult(
            parameter_value=coerced_value,
            score=score,
            stats=stats,
            action_distribution=action_dist,
            total_timesteps=total_timesteps,
            model_path=model_path,
            elapsed_seconds=elapsed,
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            note=note,
        )
        self._record_result(result, cache_key)
        self.print_results(result)
        return result

    def run_single_test(
        self, value: Union[int, float], total_timesteps: int = 100_000
    ) -> float:
        result = self._evaluate_value(
            value,
            total_timesteps=total_timesteps,
            iteration=1,
            note="single",
            use_cache=False,
        )
        return result.score

    def _optimize_custom_range(
        self, search_values: List[Union[int, float]], total_timesteps: int
    ) -> Tuple[Union[int, float], float]:
        """Evaluate specific values from custom search range and return the best."""
        logger.info("=== Custom Range Optimization for %s ===", self.parameter_name)
        logger.info("Evaluating values: %s", search_values)

        best_result: Optional[TrainingRunResult] = None

        for idx, value in enumerate(search_values, start=1):
            result = self._evaluate_value(
                value,
                total_timesteps=total_timesteps,
                iteration=idx,
                note=f"custom_{idx}",
                use_cache=True,
            )

            self._log_binary_search_event(
                "evaluation",
                {
                    "stage": f"custom_{idx}",
                    "iteration": idx,
                    "parameter_value": result.parameter_value,
                    "score": result.score,
                    "total_timesteps": total_timesteps,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )

            if best_result is None or result.score > best_result.score:
                best_result = result
                self._log_binary_search_event(
                    "best_update",
                    {
                        "stage": f"custom_{idx}",
                        "iteration": idx,
                        "parameter_value": result.parameter_value,
                        "score": result.score,
                    },
                )

        if best_result is None:
            raise ValueError("No valid results from custom search range")

        logger.info(
            "Best %s: %s (score: %.6f)",
            self.parameter_name,
            self._format_parameter_value(best_result.parameter_value),
            best_result.score,
        )
        self._log_binary_search_event(
            "complete",
            {
                "iteration": len(search_values),
                "parameter_value": best_result.parameter_value,
                "score": best_result.score,
            },
        )
        return best_result.parameter_value, best_result.score

    def binary_search_optimize(
        self, max_iterations: int = 10, total_timesteps: int = 100_000
    ) -> Tuple[Union[int, float], float]:
        # Use custom search range if provided, otherwise use default range
        if self.custom_search_range is not None:
            logger.info("Using custom search range: %s", self.custom_search_range)
            return self._optimize_custom_range(
                self.custom_search_range, total_timesteps
            )

        min_val, max_val = self.get_parameter_range()
        is_integer_range = isinstance(min_val, int) and isinstance(max_val, int)

        lower_value = self._clip_to_range(min_val)
        upper_value = self._clip_to_range(max_val)

        logger.info("=== Binary Search Optimization for %s ===", self.parameter_name)
        logger.info(
            "Parameter range: %s to %s",
            self._format_parameter_value(lower_value),
            self._format_parameter_value(upper_value),
        )

        evaluated_keys: set[Tuple[str, int]] = set()
        evaluation_index = 0

        def evaluate_candidate(
            raw_value: Union[int, float],
            stage: str,
            iteration_label: Optional[int] = None,
        ) -> TrainingRunResult:
            nonlocal evaluation_index
            evaluation_index += 1
            iteration_id = (
                iteration_label if iteration_label is not None else evaluation_index
            )
            result = self._evaluate_value(
                raw_value,
                total_timesteps=total_timesteps,
                iteration=iteration_id,
                note=stage,
                use_cache=True,
            )
            cache_key = self._cache_key(result.parameter_value, total_timesteps)
            evaluated_keys.add(cache_key)
            self._log_binary_search_event(
                "evaluation",
                {
                    "stage": stage,
                    "iteration": iteration_id,
                    "parameter_value": result.parameter_value,
                    "score": result.score,
                    "total_timesteps": total_timesteps,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )
            return result

        lower_result = evaluate_candidate(lower_value, stage="lower", iteration_label=0)
        upper_result = evaluate_candidate(upper_value, stage="upper", iteration_label=0)

        lower_value = lower_result.parameter_value
        upper_value = upper_result.parameter_value

        best_result = max(lower_result, upper_result, key=lambda r: r.score)
        previous_best_score = best_result.score

        warmup_candidates = self._generate_warmup_candidates(
            lower_value,
            upper_value,
            is_integer_range,
            evaluated_keys,
            total_timesteps,
        )

        for idx, candidate in enumerate(warmup_candidates, start=1):
            result = evaluate_candidate(candidate, stage=f"warmup_{idx}")
            if result.score > best_result.score:
                best_result = result
                self._log_binary_search_event(
                    "best_update",
                    {
                        "stage": result.note or f"warmup_{idx}",
                        "iteration": evaluation_index,
                        "parameter_value": result.parameter_value,
                        "score": result.score,
                    },
                )

        previous_best_score = best_result.score

        for iteration in range(1, max_iterations + 1):
            if is_integer_range:
                span = upper_value - lower_value
                if span <= 1:
                    break
                midpoint_value = (lower_value + upper_value) // 2
                if midpoint_value <= lower_value:
                    midpoint_value = lower_value + 1
                if midpoint_value >= upper_value:
                    break
            else:
                span = float(upper_value) - float(lower_value)
                if span <= self.search_tolerance:
                    break
                midpoint_value = float(lower_value) + span * 0.5

            midpoint_result = evaluate_candidate(
                midpoint_value, stage=f"mid_{iteration}"
            )

            if midpoint_result.score > best_result.score:
                best_result = midpoint_result
                self._log_binary_search_event(
                    "best_update",
                    {
                        "stage": midpoint_result.note or f"mid_{iteration}",
                        "iteration": evaluation_index,
                        "parameter_value": midpoint_result.parameter_value,
                        "score": midpoint_result.score,
                    },
                )

            left_gradient = midpoint_result.score - lower_result.score
            right_gradient = upper_result.score - midpoint_result.score

            if (
                self.enable_deviation_probes
                and midpoint_result.score + self.deviation_score_margin
                < min(lower_result.score, upper_result.score)
            ):
                probes = self._generate_deviation_probes(
                    midpoint_result.parameter_value,
                    lower_value,
                    upper_value,
                    is_integer_range,
                    evaluated_keys,
                    total_timesteps,
                )
                for probe_idx, probe in enumerate(probes, start=1):
                    probe_result = evaluate_candidate(
                        probe, stage=f"probe_{iteration}_{probe_idx}"
                    )
                    if probe_result.score > best_result.score:
                        best_result = probe_result
                        self._log_binary_search_event(
                            "best_update",
                            {
                                "stage": probe_result.note
                                or f"probe_{iteration}_{probe_idx}",
                                "iteration": evaluation_index,
                                "parameter_value": probe_result.parameter_value,
                                "score": probe_result.score,
                            },
                        )

            if right_gradient > left_gradient:
                lower_result = midpoint_result
                lower_value = midpoint_result.parameter_value
                self._log_binary_search_event(
                    "bounds_update",
                    {
                        "side": "lower",
                        "iteration": evaluation_index,
                        "parameter_value": lower_value,
                        "score": lower_result.score,
                    },
                )
            else:
                upper_result = midpoint_result
                upper_value = midpoint_result.parameter_value
                self._log_binary_search_event(
                    "bounds_update",
                    {
                        "side": "upper",
                        "iteration": evaluation_index,
                        "parameter_value": upper_value,
                        "score": upper_result.score,
                    },
                )

            current_best_score = best_result.score
            if is_integer_range:
                span = upper_value - lower_value
                if span <= 1:
                    break
            else:
                span = float(upper_value) - float(lower_value)
                if span <= self.search_tolerance and (
                    abs(current_best_score - previous_best_score)
                    < self.min_score_improvement
                ):
                    break

            previous_best_score = current_best_score

        refinement_candidates = self._generate_refinement_candidates(
            best_result.parameter_value,
            lower_value,
            upper_value,
            is_integer_range,
            evaluated_keys,
            total_timesteps,
        )

        for idx, candidate in enumerate(refinement_candidates, start=1):
            refine_result = evaluate_candidate(candidate, stage=f"refine_{idx}")
            if refine_result.score > best_result.score:
                best_result = refine_result
                self._log_binary_search_event(
                    "best_update",
                    {
                        "stage": refine_result.note or f"refine_{idx}",
                        "iteration": evaluation_index,
                        "parameter_value": refine_result.parameter_value,
                        "score": refine_result.score,
                    },
                )

        logger.info(
            "Best %s: %s (score: %.6f)",
            self.parameter_name,
            self._format_parameter_value(best_result.parameter_value),
            best_result.score,
        )
        self._log_binary_search_event(
            "complete",
            {
                "iteration": evaluation_index,
                "parameter_value": best_result.parameter_value,
                "score": best_result.score,
            },
        )
        return best_result.parameter_value, best_result.score


class BinarySearchArgumentParser:
    """Common argument parser for binary search scripts."""

    @staticmethod
    def create_parser(description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--mode",
            choices=["single", "binary"],
            default="single",
            help=CLIFormatter.format_help(
                "Optimization mode: single test or binary search",
                "single",
                ["single", "binary"],
            ),
        )
        parser.add_argument(
            "--max_iterations",
            type=int,
            default=10,
            help=CLIFormatter.format_help("Maximum iterations for binary search", 10),
        )
        parser.add_argument(
            "--timesteps",
            type=int,
            default=100000,
            help=CLIFormatter.format_help("Total timesteps for training", 100000),
        )
        parser.add_argument(
            "--no-warmup-grid",
            action="store_true",
            help=CLIFormatter.format_help(
                "Disable warmup grid evaluations before the main search", False
            ),
        )
        parser.add_argument(
            "--no-deviation-probes",
            action="store_true",
            help=CLIFormatter.format_help(
                "Disable deviation probes when midpoint underperforms", False
            ),
        )
        parser.add_argument(
            "--no-refinement",
            action="store_true",
            help=CLIFormatter.format_help(
                "Disable final refinement checks around the best value", False
            ),
        )
        parser.add_argument(
            "--no-integer-neighbor-warmup",
            action="store_true",
            help=CLIFormatter.format_help(
                "Disable neighbor sampling for integer ranges during warmup", False
            ),
        )
        parser.add_argument(
            "--progress-bar",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=CLIFormatter.format_help(
                "Enable or disable the training progress bar", True
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=CLIFormatter.format_help(
                "Random seed for reproducibility (optional)", None
            ),
        )
        parser.add_argument(
            "--search_range",
            type=str,
            default=None,
            help=CLIFormatter.format_help(
                "Comma-separated list of specific values to search (e.g., '16,32,64'). Overrides default range.",
                None,
            ),
        )
        return parser

    @staticmethod
    def add_parameter_argument(
        parser: argparse.ArgumentParser,
        param_name: str,
        param_type: type,
        default_value: Union[int, float],
    ) -> None:
        """Add parameter-specific argument to parser."""
        if param_type is int:
            parser.add_argument(
                f"--{param_name}",
                type=int,
                default=default_value,
                help=CLIFormatter.format_help(
                    f"{param_name} value for single test", default_value
                ),
            )
        else:
            parser.add_argument(
                f"--{param_name}",
                type=float,
                default=default_value,
                help=CLIFormatter.format_help(
                    f"{param_name} value for single test", default_value
                ),
            )
