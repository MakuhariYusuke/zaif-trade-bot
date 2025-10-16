#!/usr/bin/env python3
"""
Trading Evaluator implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from ztb.evaluation.evaluator.types import EvaluationResult, ModelConfigDict, SingleEpisodeResultDict
from ztb.metrics.metrics import (
    calculate_all_metrics,
    seasonality_analysis,
    multi_market_backtest_analysis,
    classify_market_regime,
    perform_statistical_tests,
    p_mean_method,
)
from ztb.trading.environment.environment import HeavyTradingEnv
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

try:
    from scripts.validation.stress_test import StressTestAnalyzer
    STRESS_TEST_AVAILABLE = True
except ImportError:
    STRESS_TEST_AVAILABLE = False

logger = get_logger(__name__)


class TradingEvaluator:
    """取引モデルの評価クラス"""

    writer: Any  # TensorBoard SummaryWriter
    model: Optional[PPO]
    df: Optional[pd.DataFrame]

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

    def _load_model(self) -> Optional[PPO]:
        """モデルの読み込み"""
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return None

        try:
            model = PPO.load(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _create_env(self) -> HeavyTradingEnv:
        """環境の作成"""
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig(
            max_steps=self.config["max_steps_per_episode"],
            initial_balance=10000.0,
            transaction_cost=0.0005,
        )

        return HeavyTradingEnv(
            df=self.df,
            config=config,
        )

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
                logger.info(f"Evaluating episode {episode + 1}/{self.config['n_eval_episodes']}")
                episode_result = self._evaluate_single_episode()
                results.append(episode_result)

            # 結果の集計
            aggregated_result = self._aggregate_results(results)

            logger.info("Model evaluation completed")
            return aggregated_result

    def _evaluate_single_episode(self) -> SingleEpisodeResultDict:
        """単一エピソードの評価"""
        obs, _ = self.env.reset()
        done = False
        total_reward = 0.0

        rewards = []
        positions = []
        pnls = []
        actions = []
        states = []

        while not done:
            if self.config.get("save_states", False):
                states.append(obs.copy())

            if self.model is None:
                logger.error("Model not loaded")
                break

            action, _ = self.model.predict(obs, deterministic=self.config["deterministic"])
            obs, reward, done, truncated, info = self.env.step(action)

            rewards.append(float(reward))
            positions.append(float(info.get("position", 0.0)))
            pnls.append(float(info.get("pnl", 0.0)))
            actions.append(int(action))
            total_reward += reward

            if done or truncated:
                break

        return {
            "rewards": rewards,
            "positions": positions,
            "pnls": pnls,
            "actions": actions,
            "states": states,
        }

    def _aggregate_results(self, results: List[SingleEpisodeResultDict]) -> EvaluationResult:
        """結果を集計"""
        # 基本的な集計処理
        all_rewards = [r for result in results for r in result["rewards"]]
        all_pnls = [p for result in results for p in result["pnls"]]

        # Convert PnLs to returns for metrics calculation
        pnl_returns = np.array(all_pnls)

        # Use comprehensive metrics from metrics.py
        metrics = calculate_all_metrics(pnl_returns)

        total_trades = len([a for result in results for a in result["actions"] if a != 0])  # Non-hold actions

        # Extract timestamps for seasonality analysis
        timestamps = []
        if self.df is not None:
            for result in results:
                # Assume each step corresponds to one timestamp
                episode_timestamps = [self.df.index[i] for i in range(len(result["rewards"])) if i < len(self.df)]
                timestamps.extend(episode_timestamps)

        # Perform seasonality analysis if we have enough data
        seasonality_results = {}
        if len(timestamps) >= 30:  # Need at least a month of data
            seasonality_results = seasonality_analysis(pnl_returns, timestamps)

        # Perform multi-market analysis if we have price data
        market_analysis_results = {}
        if self.df is not None and hasattr(self.df, 'close') and len(self.df) > 20:
            try:
                prices = self.df['close'].iloc[:len(pnl_returns)] if len(self.df) >= len(pnl_returns) else self.df['close']
                market_analysis_results = multi_market_backtest_analysis(pnl_returns, prices)
            except Exception as e:
                logger.warning(f"Could not perform market analysis: {e}")

        # Perform walk-forward analysis if available and we have enough data
        walkforward_results = {}
        if WALKFORWARD_AVAILABLE and len(pnl_returns) >= 100:  # Need substantial data
            try:
                analyzer = WalkforwardAnalyzer()
                # Create synthetic time series for walk-forward analysis
                synthetic_returns = pd.Series(pnl_returns, 
                    index=pd.date_range(start='2020-01-01', periods=len(pnl_returns), freq='D'))
                wf_result = analyzer.run_walkforward_analysis(synthetic_returns)
                walkforward_results = {
                    'available': True,
                    'windows_count': len(wf_result.windows),
                    'average_sharpe': np.mean(wf_result.rolling_sharpe) if wf_result.rolling_sharpe else 0.0,
                    'sharpe_volatility': np.std(wf_result.rolling_sharpe) if wf_result.rolling_sharpe else 0.0,
                }
            except Exception as e:
                logger.warning(f"Could not perform walk-forward analysis: {e}")
                walkforward_results = {'available': False, 'error': str(e)}

        # Perform stress test analysis if available
        stress_test_results = {}
        if STRESS_TEST_AVAILABLE and len(pnl_returns) >= 50:
            try:
                analyzer = StressTestAnalyzer()
                # Run basic stress tests on the returns
                stress_result = analyzer.run_stress_test(pd.Series(pnl_returns))
                stress_test_results = {
                    'available': True,
                    'scenarios_tested': len(stress_result.results) if hasattr(stress_result, 'results') else 0,
                    'average_survival_rate': np.mean([r.survival_probability for r in stress_result.results]) 
                        if hasattr(stress_result, 'results') and stress_result.results else 0.0,
                }
            except Exception as e:
                logger.warning(f"Could not perform stress test analysis: {e}")
                stress_test_results = {'available': False, 'error': str(e)}

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

    def compare_models(self, model_paths: List[str], model_names: Optional[List[str]] = None) -> None:
        """Compare multiple models."""
        logger.info(f"Comparing {len(model_paths)} models")
        # TODO: Implement model comparison logic
        pass