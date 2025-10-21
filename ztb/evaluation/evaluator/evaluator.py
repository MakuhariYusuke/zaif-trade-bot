#!/usr/bin/env python3
"""
Trading Evaluator implementation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from torch.utils.tensorboard import SummaryWriter

from ztb.evaluation.evaluator.types import EvaluationResult, SingleEpisodeResultDict
from ztb.metrics.metrics import (
    calculate_all_metrics,
    multi_market_backtest_analysis,
    seasonality_analysis,
)
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.cache_utils import TTLCache
from ztb.utils.data_utils import load_csv_data
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor

# Optional imports for advanced analysis
try:
    from scripts.analysis.walkforward_analysis import WalkforwardAnalyzer

    WALKFORWARD_AVAILABLE = True
except ImportError:
    WALKFORWARD_AVAILABLE = False
    WalkforwardAnalyzer = None

# Temporarily disable stress test due to pandas version issue
# try:
#     from scripts.validation.stress_test import StressTestAnalyzer
#     STRESS_TEST_AVAILABLE = True
# except ImportError:
#     STRESS_TEST_AVAILABLE = False
#     StressTestAnalyzer = None
STRESS_TEST_AVAILABLE = False
StressTestAnalyzer = None

logger = get_logger(__name__)


class TradingEvaluator:
    """取引モデルの評価クラス"""

    writer: Any  # TensorBoard SummaryWriter
    model: Optional[BaseAlgorithm]
    df: Optional[pd.DataFrame]
    env: HeavyTradingEnv
    results_dir: Path
    tensorboard_log_dir: Path

    def __init__(
        self, model_path: str, data_path: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.config = config or self._get_default_config()

        # Initialize utility components
        self.cache = TTLCache(ttl_seconds=3600.0)  # 1 hour cache for evaluation data
        self.performance_monitor = PerformanceMonitor("trading_evaluator")

        # データの読み込み
        self.df = self._load_data()

        # モデルの読み込み
        self.model = self._load_model()

        # 環境の作成
        self.env = self._create_env()

        # 結果保存ディレクトリ
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard設定
        self.tensorboard_log_dir = Path(
            self.config.get("tensorboard_log", "./tensorboard/")
        )
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(  # type: ignore[no-untyped-call]
            log_dir=str(self.tensorboard_log_dir / "evaluation")
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "results_dir": "./results/",
            "n_eval_episodes": 20,
            "max_steps_per_episode": 10000,
            "render_mode": None,
            "deterministic": True,
            "plot_style": "seaborn",
            "memory_optimization": True,  # Enable memory optimization by default
            "save_states": False,  # Don't save states by default to save memory
        }

    def _load_data(self) -> pd.DataFrame:
        """データの読み込み（キャッシュ最適化付き）"""
        # キャッシュチェック
        cache_path = self.data_path.with_suffix(".pkl")
        if (
            cache_path.exists()
            and cache_path.stat().st_mtime > self.data_path.stat().st_mtime
        ):
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_pickle(cache_path)
        else:
            logger.info(f"Loading data from {self.data_path}")
            df = load_csv_data(self.data_path)
            # キャッシュ保存
            df.to_pickle(cache_path)
        return df

    def _load_model(self) -> Optional[BaseAlgorithm]:
        """モデルの読み込み"""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return None

        try:
            # Use BaseAlgorithm.load to support any SB3 algorithm (PPO, SAC, etc.)
            model = BaseAlgorithm.load(str(self.model_path))
            logger.info(
                f"Model loaded from {self.model_path} (type={type(model).__name__})"
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _create_env(self) -> HeavyTradingEnv:
        """環境の作成"""
        # 既存の EnvironmentConfig を活用
        env_config = EnvironmentConfig.from_dict(self.config.get("env_config", {}))
        return HeavyTradingEnv(df=self.df, config=env_config)

    def evaluate_model(self) -> EvaluationResult:
        """
        モデルを評価し、包括的な結果を返す

        Returns:
            EvaluationResult: 評価結果
        """
        with self.performance_monitor:
            logger.info("Starting model evaluation")

            results = []
            for episode in range(self.config["n_eval_episodes"]):
                logger.info(
                    f"Evaluating episode {episode + 1}/{self.config['n_eval_episodes']}"
                )
                episode_result = self._evaluate_single_episode()
                results.append(episode_result)

            # 結果の集計
            aggregated_result = self._aggregate_results(results)

            logger.info("Model evaluation completed")
            return aggregated_result

    def _evaluate_single_episode(self) -> SingleEpisodeResultDict:
        """単一エピソードの評価"""
        # Reset environment. Different envs may return obs or (obs, info)
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) >= 1:
            obs = reset_result[0]
            # try to get reset info if available
            reset_info = reset_result[1] if len(reset_result) > 1 else {}
        else:
            obs = reset_result
            reset_info = {}
        done = False
        total_reward = 0.0

        rewards = []
        positions = []
        pnls = []
        actions = []
        states = []
        portfolio_values = []
        price_history = []
        timestamps = []

        while not done:
            if self.config.get("save_states", False):
                # obs may be numpy array or pd.Series
                try:
                    states.append(obs.copy())
                except Exception:
                    states.append(np.asarray(obs))

            if self.model is None:
                logger.error("Model not loaded")
                break

            # Predict action using SB3 model (may return scalar, array, or dict depending on policy)
            raw_action, _ = self.model.predict(
                obs, deterministic=self.config["deterministic"]
            )
            # Normalize action to a scalar/compatible form for env.step
            action = self._normalize_action(raw_action)

            step_result = self.env.step(action)
            # step may return (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_result
                truncated = False

            rewards.append(float(reward))
            positions.append(float(info.get("position", 0.0)))
            pnls.append(float(info.get("pnl", 0.0)))
            # store action as int when possible, otherwise raw
            try:
                actions.append(int(action))
            except Exception:
                actions.append(action)

            # Record timestamp for this step if available
            ts = info.get("timestamp") if isinstance(info, dict) else None
            if ts is None:
                # fallback: use env.current_step index to map to df if present
                try:
                    idx = getattr(self.env, "current_step", None)
                    if idx is not None and self.df is not None and idx < len(self.df):
                        ts = (
                            self.df.iloc[idx].get("timestamp")
                            if "timestamp" in self.df.columns
                            else self.df.index[idx]
                        )
                except Exception:
                    ts = None
            timestamps.append(ts)

            # Collect portfolio value and price if available in info
            portfolio_values.append(float(info.get("portfolio_value", np.nan)))
            price = info.get("price") if isinstance(info, dict) else None
            if price is None:
                try:
                    idx = getattr(self.env, "current_step", None)
                    if idx is not None and self.df is not None and idx < len(self.df):
                        price = float(
                            self.df.iloc[idx].get("close")
                            if "close" in self.df.columns
                            else self.df.iloc[idx].iat[0]
                        )
                except Exception:
                    price = np.nan
            price_history.append(float(price) if price is not None else np.nan)

            # attach timestamp to states list only if save_states is enabled
            if self.config.get("save_states", False):
                # keep small representation
                states[-1] = {"obs": states[-1], "timestamp": ts}
            total_reward += reward

            if done or truncated:
                break

        return {
            "rewards": rewards,
            "positions": positions,
            "pnls": pnls,
            "actions": actions,
            "states": states,
            "portfolio_history": portfolio_values,
            "price_history": price_history,
            "timestamps": timestamps,
        }

    def _normalize_action(self, raw_action: Any) -> int | float:
        """Normalize model output to a scalar or discrete action suitable for env.step.

        - If action is a numpy array with shape (n,) and env expects discrete index, map using argmax
        - If action is a float-like, return scalar
        - If action is already int, return as-is
        """
        # If action is array-like
        try:
            if isinstance(raw_action, (list, tuple)):
                raw = np.asarray(raw_action)
            else:
                raw = raw_action

            # numpy array with multiple outputs
            if isinstance(raw, np.ndarray):
                if raw.size == 1:
                    return float(raw.reshape(-1)[0])
                # If env expects discrete index, choose argmax
                return int(np.argmax(raw))

            # scalar-like
            if isinstance(raw, (float, int, np.floating, np.integer)):
                return float(raw) if isinstance(raw, (float, np.floating)) else int(raw)

        except Exception:
            pass

        # Fallback: return raw_action as int or float
        if isinstance(raw_action, (int, float)):
            return raw_action
        elif isinstance(raw_action, (list, tuple)) and len(raw_action) == 1:
            return raw_action[0]
        else:
            return 0  # Default action

    def _aggregate_results(
        self, results: List[SingleEpisodeResultDict]
    ) -> EvaluationResult:
        """結果を集計"""
        # 基本的な集計処理
        all_rewards = [r for result in results for r in result["rewards"]]
        all_pnls = [p for result in results for p in result["pnls"]]

        # Convert PnLs to returns for metrics calculation
        pnl_returns = np.array(all_pnls)

        # Use comprehensive metrics from metrics.py
        metrics = calculate_all_metrics(pnl_returns)

        total_trades = len(
            [a for result in results for a in result["actions"] if a != 0]
        )  # Non-hold actions

        # Flatten action history and compute streaks for BUY(1)/SELL(2)
        action_history = [a for result in results for a in result.get("actions", [])]

        def compute_streaks(actions_list, target):
            max_streak = 0
            streaks = []
            cur = 0
            for a in actions_list:
                if a == target:
                    cur += 1
                else:
                    if cur > 0:
                        streaks.append(cur)
                    max_streak = max(max_streak, cur)
                    cur = 0
            if cur > 0:
                streaks.append(cur)
                max_streak = max(max_streak, cur)
            avg_streak = float(np.mean(streaks)) if streaks else 0.0
            return {
                "max": int(max_streak),
                "avg": float(avg_streak),
                "count": len(streaks),
            }

        buy_streaks = compute_streaks(action_history, 1)
        sell_streaks = compute_streaks(action_history, 2)

        # Extract timestamps for seasonality analysis
        timestamps = []
        if self.df is not None:
            for result in results:
                # Assume each step corresponds to one timestamp
                episode_timestamps = [
                    self.df.index[i]
                    for i in range(len(result["rewards"]))
                    if i < len(self.df)
                ]
                timestamps.extend(episode_timestamps)

        # Perform seasonality analysis if we have enough data
        seasonality_results = {}
        if len(timestamps) >= 30:  # Need at least a month of data
            seasonality_results = seasonality_analysis(pnl_returns, timestamps)

        # Perform multi-market analysis if we have price data
        market_analysis_results = {}
        if self.df is not None and hasattr(self.df, "close") and len(self.df) > 20:
            try:
                prices = (
                    self.df["close"].iloc[: len(pnl_returns)]
                    if len(self.df) >= len(pnl_returns)
                    else self.df["close"]
                )
                market_analysis_results = multi_market_backtest_analysis(
                    pnl_returns, prices
                )
            except Exception as e:
                logger.warning(f"Could not perform market analysis: {e}")

        # Perform walk-forward analysis if available and we have enough data
        walkforward_results = {}
        if (
            WALKFORWARD_AVAILABLE
            and WalkforwardAnalyzer is not None
            and len(pnl_returns) >= 100
        ):  # Need substantial data
            try:
                analyzer = WalkforwardAnalyzer()
                # Create synthetic time series for walk-forward analysis
                synthetic_returns = pd.Series(
                    pnl_returns,
                    index=pd.date_range(
                        start="2020-01-01", periods=len(pnl_returns), freq="D"
                    ),
                )
                wf_result = analyzer.run_walkforward_analysis(synthetic_returns)
                walkforward_results = {
                    "available": True,
                    "windows_count": len(wf_result.windows),
                    "average_sharpe": np.mean(wf_result.rolling_sharpe)
                    if wf_result.rolling_sharpe
                    else 0.0,
                    "sharpe_volatility": np.std(wf_result.rolling_sharpe)
                    if wf_result.rolling_sharpe
                    else 0.0,
                }
            except Exception as e:
                logger.warning(f"Could not perform walk-forward analysis: {e}")
                walkforward_results = {"available": False, "error": str(e)}

        # Perform stress test analysis if available
        stress_test_results = {}
        if (
            STRESS_TEST_AVAILABLE
            and StressTestAnalyzer is not None
            and len(pnl_returns) >= 50
        ):
            try:
                analyzer = StressTestAnalyzer()
                # Run basic stress tests on the returns
                stress_result = analyzer.run_stress_test(pd.Series(pnl_returns))
                stress_test_results = {
                    "available": True,
                    "scenarios_tested": len(stress_result.results)
                    if hasattr(stress_result, "results")
                    else 0,
                    "average_survival_rate": np.mean(
                        [r.survival_probability for r in stress_result.results]
                    )
                    if hasattr(stress_result, "results") and stress_result.results
                    else 0.0,
                }
            except Exception as e:
                logger.warning(f"Could not perform stress test analysis: {e}")
                stress_test_results = {"available": False, "error": str(e)}

        # Prepare trade pnls (attempt to collect per-trade pnl if available in results)
        trade_pnls = []
        for res in results:
            # If environment reported per-step 'pnls' we can aggregate contiguous non-zero pnls as trades
            pnls = res.get("pnls", [])
            # simple heuristic: non-zero pnls indicate trades
            trade_pnls.extend([p for p in pnls if p != 0])

        continuous_action_stats = {
            "action_streaks": {
                "max_buy_streak": buy_streaks["max"],
                "avg_buy_streak": buy_streaks["avg"],
                "buy_streak_count": buy_streaks["count"],
                "max_sell_streak": sell_streaks["max"],
                "avg_sell_streak": sell_streaks["avg"],
                "sell_streak_count": sell_streaks["count"],
            }
        }

        return {
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "calmar_ratio": metrics["calmar_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "expected_value": metrics["expected_value"],
            "recovery_factor": metrics["recovery_factor"],
            "total_trades": total_trades,
            "avg_trade_return": float(np.mean(all_pnls)) if all_pnls else 0.0,
            "volatility": metrics["volatility"],
            "seasonality_analysis": seasonality_results,
            "market_regime_analysis": market_analysis_results,
            "walkforward_analysis": walkforward_results,
            "stress_test_analysis": stress_test_results,
            "rewards": all_rewards,
            "positions": [p for result in results for p in result["positions"]],
            "pnls": all_pnls,
            "actions": [a for result in results for a in result["actions"]],
            "states": [s for result in results for s in result["states"]],
            # compatibility with analyze_backtest.py expectations
            "action_history": action_history,
            "portfolio_history": [
                v for result in results for v in result.get("portfolio_history", [])
            ],
            "price_history": [
                v for result in results for v in result.get("price_history", [])
            ],
            "timestamps": [
                t for result in results for t in result.get("timestamps", [])
            ],
            "trade_pnls": trade_pnls,
            "continuous_action_stats": continuous_action_stats,
            "model_path": str(self.model_path),
            "evaluation_config": self.config,
        }

    def close(self) -> None:
        """Clean up resources."""
        if self.writer:
            self.writer.close()

    def create_visualizations(self) -> None:
        """Create evaluation visualizations."""
        logger.info("Creating visualizations")
        # TODO: Implement visualization logic
        pass

    def compare_models(
        self, model_paths: List[str], model_names: Optional[List[str]] = None
    ) -> None:
        """Compare multiple models."""
        logger.info(f"Comparing {len(model_paths)} models")
        # TODO: Implement model comparison logic
        pass
