#!/usr/bin/env python3
"""
V433 Unified Optimizer: 統合最適化システム
現実データ中心主義に基づく包括的な最適化フレームワーク
"""

import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.training.reward_function_optimizer.reward_function_optimizer import RewardFunctionOptimizer
from ztb.training.system_optimizer import SystemOptimizer
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.logging_utils import get_logger
from ztb.utils.system_utils import check_library_availability

logger = get_logger(__name__)

# 依存ライブラリのチェック
OPTUNA_AVAILABLE = check_library_availability('optuna', 'Bayesian optimization')
TQDM_AVAILABLE = check_library_availability('tqdm', 'Progress bars')
PSUTIL_AVAILABLE = check_library_availability('psutil', 'System monitoring')

# 条件付きインポート
if OPTUNA_AVAILABLE:
    import optuna
else:
    optuna = None

if TQDM_AVAILABLE:
    from tqdm import tqdm
else:
    tqdm = None

if PSUTIL_AVAILABLE:
    import psutil
else:
    psutil = None


@dataclass
class OptimizationConfig:
    """統合最適化の設定"""

    # ハイパーパラメータ最適化
    enable_hyperparameter_optimization: bool = True
    optimization_method: str = "bayesian"  # "bayesian", "grid", "random"
    max_trials: int = 100
    timeout_hours: float = 24.0

    # システム最適化
    enable_system_optimization: bool = True
    memory_threshold_mb: float = 100.0
    gc_interval_steps: int = 100
    enable_io_caching: bool = True
    cache_ttl_seconds: int = 300

    # 報酬関数最適化
    enable_reward_optimization: bool = True
    reward_objectives: List[str] = field(
        default_factory=lambda: ["win_rate", "profit_factor", "max_drawdown"]
    )

    # 適応最適化
    enable_adaptive_optimization: bool = True
    adaptation_interval_steps: int = 1000
    market_regime_awareness: bool = True

    # パフォーマンス監視
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 100

    # 並列処理
    max_parallel_trials: int = 4
    enable_distributed: bool = False


@dataclass
class OptimizationResult:
    """最適化結果"""

    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    execution_time: float
    convergence_info: Dict[str, Any]
    recommendations: List[str]


class BaseOptimizer(ABC):
    """最適化器の基底クラス"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger(__name__)

    @abstractmethod
    def optimize(
        self, objective_function: Callable, search_space: Dict[str, Any]
    ) -> OptimizationResult:
        """最適化を実行"""
        pass

    @abstractmethod
    def get_optimizer_info(self) -> Dict[str, Any]:
        """最適化器の情報を取得"""
        pass


class BayesianOptimizer(BaseOptimizer):
    """ベイズ最適化器 (Optuna使用)"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")

    def optimize(
        self, objective_function: Callable, search_space: Dict[str, Any]
    ) -> OptimizationResult:
        """ベイズ最適化を実行"""
        start_time = time.time()

        def optuna_objective(trial):
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config.get("type", "float")

                if param_type == "float":
                    low = param_config.get("low", 0.0)
                    high = param_config.get("high", 1.0)
                    params[param_name] = trial.suggest_float(param_name, low, high)
                elif param_type == "int":
                    low = param_config.get("low", 0)
                    high = param_config.get("high", 100)
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif param_type == "categorical":
                    choices = param_config.get("choices", [])
                    params[param_name] = trial.suggest_categorical(param_name, choices)

            return objective_function(params)

        # Optuna studyの作成
        study = optuna.create_study(direction="maximize")
        study.optimize(
            optuna_objective,
            n_trials=self.config.max_trials,
            timeout=self.config.timeout_hours * 3600,
        )

        # 結果の収集
        best_params = study.best_params
        best_score = study.best_value

        optimization_history = []
        for trial in study.trials:
            optimization_history.append(
                {
                    "trial_id": trial.number,
                    "params": trial.params,
                    "score": trial.value,
                    "datetime": trial.datetime_start.isoformat()
                    if trial.datetime_start
                    else None,
                }
            )

        execution_time = time.time() - start_time

        convergence_info = {
            "total_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "convergence_score": self._calculate_convergence_score(study),
        }

        recommendations = self._generate_recommendations(study, search_space)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            execution_time=execution_time,
            convergence_info=convergence_info,
            recommendations=recommendations,
        )

    def _calculate_convergence_score(self, study) -> float:
        """収束スコアを計算"""
        if len(study.trials) < 10:
            return 0.0

        # 最終10試行のベストスコアの変動係数を計算
        recent_scores = [t.value for t in study.trials[-10:]]
        if len(recent_scores) < 2:
            return 0.0

        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)

        # 変動係数 (低いほど収束している)
        cv = std_score / abs(mean_score) if mean_score != 0 else 1.0
        convergence_score = max(0, 1.0 - cv)

        return convergence_score

    def _generate_recommendations(
        self, study, search_space: Dict[str, Any]
    ) -> List[str]:
        """最適化結果に基づく推奨事項を生成"""
        recommendations = []

        # パラメータ重要度の分析
        try:
            importance = optuna.importance.get_param_importances(study)
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]

            if top_params:
                recommendations.append(
                    f"最も重要なパラメータ: {', '.join([p[0] for p in top_params])}"
                )
        except Exception as e:
            self.logger.warning(f"Failed to calculate parameter importance: {e}")

        # 収束状況の分析
        convergence_score = self._calculate_convergence_score(study)
        if convergence_score > 0.8:
            recommendations.append("最適化が良く収束しています")
        elif convergence_score < 0.3:
            recommendations.append(
                "最適化が収束していない可能性があります。試行回数を増やすことを検討してください"
            )

        # 探索範囲の分析
        for param_name, param_config in search_space.items():
            if param_config.get("type") == "float":
                param_values = [
                    t.params.get(param_name)
                    for t in study.trials
                    if param_name in t.params
                ]
                if param_values:
                    value_range = max(param_values) - min(param_values)
                    search_range = param_config["high"] - param_config["low"]

                    if value_range / search_range < 0.1:
                        recommendations.append(
                            f"{param_name}の探索範囲を狭くすることを検討してください"
                        )

        return recommendations

    def get_optimizer_info(self) -> Dict[str, Any]:
        return {
            "name": "Bayesian Optimizer",
            "method": "TPE (Tree-structured Parzen Estimator)",
            "library": "Optuna",
            "parallel_support": True,
        }


class GridOptimizer(BaseOptimizer):
    """グリッド検索最適化器"""

    def optimize(
        self, objective_function: Callable, search_space: Dict[str, Any]
    ) -> OptimizationResult:
        """グリッド検索を実行"""
        start_time = time.time()

        # グリッドの生成
        param_grids = {}
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type", "float")

            if param_type == "categorical":
                param_grids[param_name] = param_config.get("choices", [])
            elif param_type in ["float", "int"]:
                low = param_config.get("low", 0)
                high = param_config.get("high", 1)
                num_values = param_config.get("num_values", 5)

                if param_type == "float":
                    param_grids[param_name] = np.linspace(
                        low, high, num_values
                    ).tolist()
                else:
                    param_grids[param_name] = list(
                        range(low, high + 1, max(1, (high - low) // num_values))
                    )

        # 全ての組み合わせを生成
        import itertools

        param_names = list(param_grids.keys())
        param_combinations = list(
            itertools.product(*[param_grids[name] for name in param_names])
        )

        self.logger.info(f"Grid search with {len(param_combinations)} combinations")

        optimization_history = []

        best_score = float("-inf")
        best_params = None

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))

            try:
                score = objective_function(params)

                optimization_history.append(
                    {
                        "trial_id": i,
                        "params": params,
                        "score": score,
                        "datetime": datetime.now().isoformat(),
                    }
                )

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                self.logger.warning(f"Failed evaluation for params {params}: {e}")
                optimization_history.append(
                    {
                        "trial_id": i,
                        "params": params,
                        "score": None,
                        "error": str(e),
                        "datetime": datetime.now().isoformat(),
                    }
                )

        execution_time = time.time() - start_time

        convergence_info = {
            "total_trials": len(param_combinations),
            "evaluated_trials": len(
                [h for h in optimization_history if h.get("score") is not None]
            ),
            "convergence_score": 1.0,  # グリッド検索は常に収束
        }

        recommendations = [
            "グリッド検索が完了しました",
            f"総組み合わせ数: {len(param_combinations)}",
            f"評価成功数: {convergence_info['evaluated_trials']}",
        ]

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score if best_score != float("-inf") else 0.0,
            optimization_history=optimization_history,
            execution_time=execution_time,
            convergence_info=convergence_info,
            recommendations=recommendations,
        )

    def get_optimizer_info(self) -> Dict[str, Any]:
        return {
            "name": "Grid Search Optimizer",
            "method": "Exhaustive Grid Search",
            "library": "itertools",
            "parallel_support": True,
        }


class RewardFunctionOptimizer:
    """報酬関数最適化器"""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def optimize_reward_structure(
        self, evaluation_function: Callable, current_reward_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """報酬関数構造の最適化"""
        self.logger.info("Starting reward function optimization")

        # 報酬関数のパラメータ空間を定義
        reward_search_space = {
            "profit_weight": {"type": "float", "low": 0.0, "high": 2.0},
            "risk_weight": {"type": "float", "low": 0.0, "high": 2.0},
            "holding_penalty": {"type": "float", "low": 0.0, "high": 0.1},
            "transaction_cost_weight": {"type": "float", "low": 0.0, "high": 0.01},
            "time_decay_factor": {"type": "float", "low": 0.9, "high": 1.0},
            "market_regime_factor": {"type": "float", "low": 0.0, "high": 1.0},
        }

        def reward_objective(params) -> float:
            # 報酬関数パラメータを更新
            test_config = current_reward_config.copy()
            test_config.update(params)

            # 評価関数でスコアを計算
            score = evaluation_function(test_config)
            return score

        # 最適化実行
        optimizer = BayesianOptimizer(self.config)
        result = optimizer.optimize(reward_objective, reward_search_space)

        optimized_config = current_reward_config.copy()
        optimized_config.update(result.best_params)

        self.logger.info(
            f"Reward optimization completed. Best score: {result.best_score:.4f}"
        )

        return {
            "optimized_config": optimized_config,
            "optimization_result": result,
            "improvement": result.best_score
            - evaluation_function(current_reward_config),
        }

    def set_console_output(self, verbose: bool = False, show_progress: bool = False) -> None:
        """Configure console output verbosity and progress display for optimizer."""
        self._console_verbose = bool(verbose)
        self._show_progress = bool(show_progress)


class UnifiedOptimizer:
    """
    V433統合最適化システム
    ハイパーパラメータ、システム、報酬関数を統合した最適化
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        # 各最適化器の初期化
        self.hyperparameter_optimizer = None
        self.system_optimizer = SystemOptimizer(config)
        self.reward_optimizer = RewardFunctionOptimizer(config)

        # 新機能: マルチタイムフレーム最適化器とA/Bテストフレームワーク
        self.multi_timeframe_optimizer = MultiTimeframeOptimizer(config)
        self.ab_testing_framework = ABTestingFramework(config)

        # 自動最適化パイプライン
        self.automatic_pipeline = AutomaticOptimizationPipeline(config, self.system_optimizer)

        # 最適化結果持続化
        self.persistence = OptimizationResultPersistence()

        # 並列最適化実行器
        self.parallel_optimizer = ParallelOptimizer(config)

        # 最適化履歴
        self.optimization_history = []
        self.current_best_params = {}

        # ハイパーパラメータ最適化器の初期化（optimization_methodに従う）
        self.initialize_hyperparameter_optimizer(self.config.optimization_method)

    def initialize_hyperparameter_optimizer(self, method: str = "bayesian") -> None:
        """ハイパーパラメータ最適化器の初期化"""
        if method == "bayesian" and OPTUNA_AVAILABLE:
            self.hyperparameter_optimizer = BayesianOptimizer(self.config)
        elif method == "grid":
            self.hyperparameter_optimizer = GridOptimizer(self.config)
        else:
            self.logger.warning(f"Unsupported optimization method: {method}")
            self.hyperparameter_optimizer = None

    def optimize_hyperparameters(
        self,
        objective_function: Callable,
        search_space: Dict[str, Any],
        method: str = "bayesian",
    ) -> OptimizationResult:
        """ハイパーパラメータの最適化"""
        if not self.config.enable_hyperparameter_optimization:
            return OptimizationResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                execution_time=0.0,
                convergence_info={},
                recommendations=["Hyperparameter optimization disabled"],
            )

        if self.hyperparameter_optimizer is None:
            self.initialize_hyperparameter_optimizer(method)

        if self.hyperparameter_optimizer is None:
            raise ValueError(f"Could not initialize {method} optimizer")

        self.logger.info(f"Starting hyperparameter optimization using {method} method")
        result = self.hyperparameter_optimizer.optimize(
            objective_function, search_space
        )

        # 結果を保存
        self.current_best_params.update(result.best_params)
        self.optimization_history.append(
            {
                "type": "hyperparameter",
                "method": method,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def optimize_system(self) -> Dict[str, Any]:
        """システム最適化を実行"""
        if not self.config.enable_system_optimization:
            return {"status": "disabled"}

        self.logger.info("Starting system optimization")
        memory_result = self.system_optimizer.optimize_memory_usage()

        result = {
            "memory_optimization": memory_result,
            "system_status": self.system_optimizer.get_system_status(),
            "timestamp": datetime.now().isoformat(),
        }

        self.optimization_history.append(
            {
                "type": "system",
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def optimize_reward_function(
        self, evaluation_function: Callable, current_reward_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """報酬関数の最適化"""
        if not self.config.enable_reward_optimization:
            return {"status": "disabled"}

        self.logger.info("Starting reward function optimization")
        result = self.reward_optimizer.optimize_reward_structure(
            evaluation_function, current_reward_config
        )

        self.optimization_history.append(
            {
                "type": "reward",
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def adaptive_optimize(
        self, current_performance: Dict[str, Any], market_regime: str = "neutral"
    ) -> Dict[str, Any]:
        """適応的最適化（市場状況に応じた動的調整）"""
        if not self.config.enable_adaptive_optimization:
            return {"status": "disabled"}

        self.logger.info(
            f"Starting adaptive optimization for market regime: {market_regime}"
        )

        # 市場レジームに応じた最適化戦略
        regime_strategies = {
            "bull": {
                "profit_weight_multiplier": 1.2,
                "risk_weight_multiplier": 0.8,
                "learning_rate_multiplier": 1.1,
            },
            "bear": {
                "profit_weight_multiplier": 0.8,
                "risk_weight_multiplier": 1.3,
                "learning_rate_multiplier": 0.9,
            },
            "volatile": {
                "profit_weight_multiplier": 0.9,
                "risk_weight_multiplier": 1.5,
                "learning_rate_multiplier": 0.8,
            },
            "neutral": {
                "profit_weight_multiplier": 1.0,
                "risk_weight_multiplier": 1.0,
                "learning_rate_multiplier": 1.0,
            },
        }

        strategy = regime_strategies.get(market_regime, regime_strategies["neutral"])

        # パフォーマンスに基づく調整
        performance_score = current_performance.get("score", 0.5)

        if performance_score < 0.4:
            # パフォーマンスが悪い場合、保守的に調整
            strategy["risk_weight_multiplier"] *= 1.2
            strategy["learning_rate_multiplier"] *= 0.8
        elif performance_score > 0.7:
            # パフォーマンスが良い場合、積極的に調整
            strategy["profit_weight_multiplier"] *= 1.1
            strategy["risk_weight_multiplier"] *= 0.9

        result = {
            "market_regime": market_regime,
            "strategy": strategy,
            "performance_score": performance_score,
            "recommendations": self._generate_adaptive_recommendations(
                strategy, performance_score
            ),
        }

        self.optimization_history.append(
            {
                "type": "adaptive",
                "market_regime": market_regime,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return result

    def _generate_adaptive_recommendations(
        self, strategy: Dict[str, float], performance_score: float
    ) -> List[str]:
        """適応最適化の推奨事項を生成"""
        recommendations = []

        if strategy["profit_weight_multiplier"] > 1.1:
            recommendations.append("利益重視の戦略を推奨")
        elif strategy["risk_weight_multiplier"] > 1.2:
            recommendations.append("リスク管理を強化することを推奨")

        if performance_score < 0.4:
            recommendations.append("モデルの再訓練を検討してください")
        elif performance_score > 0.7:
            recommendations.append("現在の戦略を継続することを推奨")

        return recommendations

    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化のサマリーを取得"""
        summary = {
            "total_optimizations": len(self.optimization_history),
            "optimization_types": {},
            "best_parameters": self.current_best_params,
            "system_status": self.system_optimizer.get_system_status(),
            "recent_history": self.optimization_history[-5:]
            if self.optimization_history
            else [],
        }

        # 最適化タイプの集計
        for opt in self.optimization_history:
            opt_type = opt.get("type", "unknown")
            summary["optimization_types"][opt_type] = (
                summary["optimization_types"].get(opt_type, 0) + 1
            )

        return summary

    def save_optimization_results(self, filepath: str) -> None:
        """最適化結果を保存（拡張版）"""
        results = {
            "config": self.config.__dict__,
            "current_best_params": self.current_best_params,
            "optimization_history": self.optimization_history,
            "summary": self.get_optimization_summary(),
            "timestamp": datetime.now().isoformat(),
        }

        # 従来のJSON保存
        safe_json_dump(results, filepath, indent=2, ensure_ascii=False, default=str)

        # バージョン管理システムにも保存
        try:
            version_id = self.persistence.save_optimization_result(
                result=results,
                result_type="unified_optimizer_snapshot",
                metadata={"filepath": filepath},
                tags=["snapshot", "unified_optimizer"]
            )
            self.logger.info(f"Saved optimization snapshot: {version_id}")
        except Exception as e:
            self.logger.warning(f"Failed to save to version control: {e}")

        self.logger.info(f"Optimization results saved to {filepath}")

    def save_result_to_version_control(
        self,
        result: Dict[str, Any],
        result_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """結果をバージョン管理システムに保存"""
        return self.persistence.save_optimization_result(result, result_type, metadata, tags)

    def load_result_from_version_control(self, version_id: str) -> Optional[Dict[str, Any]]:
        """バージョン管理システムから結果を読み込み"""
        return self.persistence.load_optimization_result(version_id)

    def search_optimization_results(
        self,
        result_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_score: Optional[float] = None,
        max_p_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """最適化結果を検索"""
        return self.persistence.search_results(
            result_type, tags, date_from, date_to, min_score, max_p_value
        )

    def compare_optimization_results(
        self,
        version_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """最適化結果を比較"""
        return self.persistence.get_result_comparison(version_ids, metrics)

    def get_persistence_stats(self) -> Dict[str, Any]:
        """持続化統計を取得"""
        return self.persistence.get_persistence_stats()

    def cleanup_old_results(self, keep_recent: int = 50):
        """古い最適化結果を削除"""
        self.persistence.cleanup_old_versions(keep_recent)

    def load_optimization_results(self, filepath: str):
        """最適化結果を読み込み"""
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.current_best_params = results.get("current_best_params", {})
        self.optimization_history = results.get("optimization_history", [])

        self.logger.info(f"Optimization results loaded from {filepath}")

    def optimize_multi_timeframe(
        self,
        objective_functions: Dict[str, Callable],
        search_spaces: Dict[str, Dict[str, Any]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, OptimizationResult]:
        """マルチタイムフレーム最適化を実行"""
        if not self.config.enable_hyperparameter_optimization:
            return {"integrated": OptimizationResult(
                best_params={},
                best_score=0.0,
                optimization_history=[],
                execution_time=0.0,
                convergence_info={},
                recommendations=["Multi-timeframe optimization disabled"]
            )}

        self.logger.info("Starting multi-timeframe optimization via UnifiedOptimizer")
        result = self.multi_timeframe_optimizer.optimize_multi_timeframe(
            objective_functions, search_spaces, correlation_matrix
        )

        # 結果を保存
        self.current_best_params.update(result.get("integrated", OptimizationResult({}, 0.0, [], 0.0, {}, [])).best_params)
        self.optimization_history.append({
            "type": "multi_timeframe",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def create_ab_test(
        self,
        test_name: str,
        control_params: Dict[str, Any],
        variant_params: Dict[str, Any],
        evaluation_function: Callable,
        sample_size_per_group: int = 100,
        test_duration_days: int = 7
    ) -> str:
        """A/Bテストを作成"""
        return self.ab_testing_framework.create_ab_test(
            test_name, control_params, variant_params,
            evaluation_function, sample_size_per_group, test_duration_days
        )

    def run_ab_test(self, test_id: str, num_iterations: int = 10) -> Dict[str, Any]:
        """A/Bテストを実行"""
        self.logger.info(f"Running A/B test via UnifiedOptimizer: {test_id}")
        result = self.ab_testing_framework.run_ab_test(test_id, num_iterations)

        # 結果を保存
        self.optimization_history.append({
            "type": "ab_test",
            "test_id": test_id,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def get_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """A/Bテスト結果を取得"""
        return self.ab_testing_framework.get_test_results(test_id)

    def get_multi_timeframe_analysis(self) -> Dict[str, Any]:
        """マルチタイムフレーム分析結果を取得"""
        return self.multi_timeframe_optimizer.get_multi_timeframe_analysis()

    def get_ab_testing_summary(self) -> Dict[str, Any]:
        """A/Bテストサマリーを取得"""
        return self.ab_testing_framework.get_ab_testing_summary()

    def run_automatic_pipeline(
        self,
        base_params: Dict[str, Any],
        objective_function: Callable,
        search_space: Dict[str, Any],
        multi_timeframe_functions: Optional[Dict[str, Callable]] = None,
        validation_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """自動最適化パイプラインを実行"""
        self.logger.info("Running automatic optimization pipeline via UnifiedOptimizer")
        result = self.automatic_pipeline.run_full_pipeline(
            base_params, objective_function, search_space,
            multi_timeframe_functions, validation_function
        )

        # 結果を保存
        self.optimization_history.append({
            "type": "automatic_pipeline",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        # パイプラインが成功した場合、最適パラメータを更新
        if result.get("success", False) and result["final_recommendation"].get("action") == "deploy":
            self.current_best_params.update(result["final_recommendation"]["params"])

        return result

    def get_pipeline_history(self) -> List[Dict[str, Any]]:
        """パイプライン実行履歴を取得"""
        return self.automatic_pipeline.get_pipeline_history()

    def get_pipeline_status(self) -> Dict[str, Any]:
        """パイプラインの現在の状態を取得"""
        return self.automatic_pipeline.get_pipeline_status()

    def run_parallel_optimization(
        self,
        optimization_tasks: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """並列最適化を実行"""
        self.logger.info("Running parallel optimization via UnifiedOptimizer")
        result = self.parallel_optimizer.run_parallel_optimization(optimization_tasks, progress_callback)

        # 結果を保存
        self.optimization_history.append({
            "type": "parallel_optimization",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def run_parameter_sweep(
        self,
        objective_function: Callable,
        parameter_combinations: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """パラメータスイープを実行"""
        self.logger.info("Running parameter sweep via UnifiedOptimizer")
        result = self.parallel_optimizer.run_parameter_sweep(
            self.hyperparameter_optimizer, objective_function,
            parameter_combinations, progress_callback
        )

        # 結果を保存
        self.optimization_history.append({
            "type": "parameter_sweep",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def get_parallel_status(self) -> Dict[str, Any]:
        """並列実行の状態を取得"""
        return self.parallel_optimizer.get_active_task_status()

    def shutdown_parallel_executor(self) -> None:
        """並列実行器をシャットダウン"""
        self.parallel_optimizer.shutdown()


class MultiTimeframeOptimizer:
    """
    マルチタイムフレームパラメータ最適化器
    1分/5分/15分足の各タイムフレームで最適なパラメータを探索
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        # タイムフレーム設定
        self.timeframes = ["1m", "5m", "15m"]
        self.timeframe_weights = {"1m": 0.5, "5m": 0.3, "15m": 0.2}  # 重要度重み

        # 各タイムフレームの最適化器
        self.timeframe_optimizers = {}
        for tf in self.timeframes:
            self.timeframe_optimizers[tf] = BayesianOptimizer(config)

        # クロスタイムフレーム最適化結果
        self.cross_timeframe_results = {}

    def optimize_multi_timeframe(
        self,
        objective_functions: Dict[str, Callable],
        search_spaces: Dict[str, Dict[str, Any]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, OptimizationResult]:
        """
        マルチタイムフレーム最適化を実行

        Args:
            objective_functions: タイムフレームごとの目的関数
            search_spaces: タイムフレームごとの探索空間
            correlation_matrix: タイムフレーム間の相関行列（オプション）
        """
        self.logger.info("Starting multi-timeframe optimization")

        results = {}
        start_time = time.time()

        # 各タイムフレームで個別に最適化
        for tf in self.timeframes:
            if tf not in objective_functions:
                self.logger.warning(f"No objective function for timeframe {tf}")
                continue

            self.logger.info(f"Optimizing parameters for {tf} timeframe")

            # タイムフレーム固有の探索空間を使用（デフォルトは共通）
            tf_search_space = search_spaces.get(tf, search_spaces.get("default", {}))

            result = self.timeframe_optimizers[tf].optimize(
                objective_functions[tf], tf_search_space
            )

            results[tf] = result

            self.logger.info(f"{tf} optimization completed. Best score: {result.best_score:.4f}")

        # クロスタイムフレーム分析
        if len(results) > 1:
            self._analyze_cross_timeframe_patterns(results, correlation_matrix)

        # 統合パラメータの生成
        integrated_params = self._generate_integrated_parameters(results)

        execution_time = time.time() - start_time
        self.logger.info(f"Multi-timeframe optimization completed in {execution_time:.2f}s")

        # 結果に統合パラメータを追加
        results["integrated"] = OptimizationResult(
            best_params=integrated_params,
            best_score=self._calculate_integrated_score(results),
            optimization_history=[],
            execution_time=execution_time,
            convergence_info=self._get_integrated_convergence_info(results),
            recommendations=self._generate_integrated_recommendations(results)
        )

        return results

    def _analyze_cross_timeframe_patterns(
        self,
        results: Dict[str, OptimizationResult],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> None:
        """クロスタイムフレームパターンを分析"""
        self.cross_timeframe_results = {
            "parameter_correlations": {},
            "timeframe_importance": {},
            "pattern_recommendations": []
        }

        # パラメータ相関の分析
        common_params = set()
        for tf, result in results.items():
            if tf == "integrated":
                continue
            common_params.update(result.best_params.keys())

        for param in common_params:
            param_values = []
            timeframes_with_param = []

            for tf, result in results.items():
                if tf == "integrated":
                    continue
                if param in result.best_params:
                    param_values.append(result.best_params[param])
                    timeframes_with_param.append(tf)

            if len(param_values) > 1:
                # パラメータ値の相関分析（2つのタイムフレームの場合のみ）
                if len(param_values) == 2:
                    correlation = np.corrcoef([param_values[0]], [param_values[1]])[0, 1]
                else:
                    correlation = 0  # 3つ以上の場合は相関を計算しない
                self.cross_timeframe_results["parameter_correlations"][param] = {
                    "correlation": correlation,
                    "timeframes": timeframes_with_param,
                    "values": dict(zip(timeframes_with_param, param_values))
                }

        # タイムフレーム重要度の分析
        for tf, result in results.items():
            if tf == "integrated":
                continue
            weight = self.timeframe_weights.get(tf, 1.0)
            score = result.best_score
            self.cross_timeframe_results["timeframe_importance"][tf] = {
                "weight": weight,
                "score": score,
                "weighted_score": weight * score
            }

        # パターン推奨事項の生成
        self._generate_pattern_recommendations()

    def _generate_pattern_recommendations(self) -> None:
        """パターン分析に基づく推奨事項を生成"""
        recommendations = []

        # パラメータ相関に基づく推奨
        for param, data in self.cross_timeframe_results["parameter_correlations"].items():
            correlation = data["correlation"]
            if abs(correlation) > 0.8:
                if correlation > 0:
                    recommendations.append(
                        f"{param}はタイムフレーム間で正の相関が強い（{correlation:.2f}）。統一パラメータの使用を検討"
                    )
                else:
                    recommendations.append(
                        f"{param}はタイムフレーム間で負の相関（{correlation:.2f}）。タイムフレーム別最適化を推奨"
                    )

        # タイムフレーム重要度に基づく推奨
        importance_scores = self.cross_timeframe_results["timeframe_importance"]
        sorted_tf = sorted(importance_scores.items(), key=lambda x: x[1]["weighted_score"], reverse=True)

        if sorted_tf:
            top_tf = sorted_tf[0][0]
            recommendations.append(f"最も重要なタイムフレーム: {top_tf}（重み付きスコア: {importance_scores[top_tf]['weighted_score']:.4f}）")

        self.cross_timeframe_results["pattern_recommendations"] = recommendations

    def _generate_integrated_parameters(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """統合パラメータを生成"""
        integrated_params = {}

        # 各パラメータの統合値計算
        param_values = defaultdict(list)
        param_weights = defaultdict(list)

        for tf, result in results.items():
            if tf == "integrated":
                continue

            weight = self.timeframe_weights.get(tf, 1.0)

            for param, value in result.best_params.items():
                param_values[param].append(value)
                param_weights[param].append(weight)

        # 重み付き平均で統合
        for param, values in param_values.items():
            weights = param_weights[param]
            integrated_value = np.average(values, weights=weights)
            integrated_params[param] = integrated_value

        return integrated_params

    def _calculate_integrated_score(self, results: Dict[str, OptimizationResult]) -> float:
        """統合スコアを計算"""
        total_weighted_score = 0
        total_weight = 0

        for tf, result in results.items():
            if tf == "integrated":
                continue

            weight = self.timeframe_weights.get(tf, 1.0)
            score = result.best_score

            total_weighted_score += weight * score
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0

    def _get_integrated_convergence_info(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """統合収束情報を取得"""
        convergence_scores = []
        total_trials = 0
        evaluated_trials = 0

        for tf, result in results.items():
            if tf == "integrated":
                continue

            conv_info = result.convergence_info
            if "convergence_score" in conv_info:
                weight = self.timeframe_weights.get(tf, 1.0)
                convergence_scores.append((conv_info["convergence_score"], weight))

            total_trials += conv_info.get("total_trials", 0)
            evaluated_trials += conv_info.get("evaluated_trials", 0)

        # 重み付き平均収束スコア
        if convergence_scores:
            weighted_convergence = sum(score * weight for score, weight in convergence_scores)
            total_weight = sum(weight for _, weight in convergence_scores)
            avg_convergence = weighted_convergence / total_weight if total_weight > 0 else 0
        else:
            avg_convergence = 0

        return {
            "average_convergence_score": avg_convergence,
            "total_trials": total_trials,
            "evaluated_trials": evaluated_trials,
            "timeframes_optimized": len([tf for tf in results.keys() if tf != "integrated"])
        }

    def _generate_integrated_recommendations(self, results: Dict[str, OptimizationResult]) -> List[str]:
        """統合推奨事項を生成"""
        recommendations = ["マルチタイムフレーム最適化が完了しました"]

        # 各タイムフレームの結果サマリー
        for tf, result in results.items():
            if tf == "integrated":
                continue
            recommendations.append(f"{tf}: 最適スコア {result.best_score:.4f}")

        # クロスタイムフレーム分析結果
        if self.cross_timeframe_results.get("pattern_recommendations"):
            recommendations.extend(self.cross_timeframe_results["pattern_recommendations"])

        return recommendations

    def get_multi_timeframe_analysis(self) -> Dict[str, Any]:
        """マルチタイムフレーム分析結果を取得"""
        return {
            "timeframes": self.timeframes,
            "timeframe_weights": self.timeframe_weights,
            "cross_timeframe_results": self.cross_timeframe_results,
            "optimizer_info": {tf: opt.get_optimizer_info() for tf, opt in self.timeframe_optimizers.items()}
        }


class ABTestingFramework:
    """
    A/Bテストフレームワーク
    統計的有意差検定機能付きパラメータ比較
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # テスト設定
        self.confidence_level = 0.95
        self.min_sample_size = 30
        self.max_test_duration_days = 30

        # テスト結果
        self.test_results = {}
        self.active_tests = {}

    def create_ab_test(
        self,
        test_name: str,
        control_params: Dict[str, Any],
        variant_params: Dict[str, Any],
        evaluation_function: Callable,
        sample_size_per_group: int = 100,
        test_duration_days: int = 7
    ) -> str:
        """
        A/Bテストを作成

        Args:
            test_name: テスト名
            control_params: コントロールグループのパラメータ
            variant_params: バリアントグループのパラメータ
            evaluation_function: 評価関数
            sample_size_per_group: グループごとのサンプルサイズ
            test_duration_days: テスト期間（日）

        Returns:
            テストID
        """
        test_id = f"{test_name}_{int(time.time())}"

        test_config = {
            "test_id": test_id,
            "test_name": test_name,
            "control_params": control_params,
            "variant_params": variant_params,
            "evaluation_function": evaluation_function,
            "sample_size_per_group": sample_size_per_group,
            "test_duration_days": min(test_duration_days, self.max_test_duration_days),
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "control_results": [],
            "variant_results": [],
            "significance_tests": []
        }

        self.active_tests[test_id] = test_config
        self.logger.info(f"Created A/B test: {test_id}")

        return test_id

    def run_ab_test(self, test_id: str, num_iterations: int = 10) -> Dict[str, Any]:
        """
        A/Bテストを実行

        Args:
            test_id: テストID
            num_iterations: イテレーション回数

        Returns:
            テスト結果
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_config = self.active_tests[test_id]
        test_config["status"] = "running"
        test_config["started_at"] = datetime.now().isoformat()

        self.logger.info(f"Running A/B test: {test_id}")

        # 各イテレーションで評価を実行
        for i in range(num_iterations):
            try:
                # コントロールグループの評価
                control_score = test_config["evaluation_function"](test_config["control_params"])
                test_config["control_results"].append({
                    "iteration": i,
                    "score": control_score,
                    "timestamp": datetime.now().isoformat()
                })

                # バリアントグループの評価
                variant_score = test_config["evaluation_function"](test_config["variant_params"])
                test_config["variant_results"].append({
                    "iteration": i,
                    "score": variant_score,
                    "timestamp": datetime.now().isoformat()
                })

                self.logger.debug(f"Test {test_id} iteration {i}: Control={control_score:.4f}, Variant={variant_score:.4f}")

            except Exception as e:
                self.logger.warning(f"Failed evaluation in iteration {i}: {e}")
                continue

        # 統計的有意差検定
        significance_result = self._perform_significance_test(test_config)
        test_config["significance_tests"].append(significance_result)

        # テスト完了判定
        test_config["status"] = self._determine_test_status(test_config)

        result = {
            "test_id": test_id,
            "status": test_config["status"],
            "control_results": test_config["control_results"],
            "variant_results": test_config["variant_results"],
            "significance_test": significance_result,
            "recommendations": self._generate_test_recommendations(test_config, significance_result)
        }

        self.test_results[test_id] = result
        self.logger.info(f"A/B test {test_id} completed with status: {test_config['status']}")

        return result

    def _perform_significance_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """統計的有意差検定を実行"""
        control_scores = [r["score"] for r in test_config["control_results"]]
        variant_scores = [r["score"] for r in test_config["variant_results"]]

        if len(control_scores) < self.min_sample_size or len(variant_scores) < self.min_sample_size:
            return {
                "test_performed": False,
                "reason": f"Insufficient sample size (min: {self.min_sample_size})",
                "control_samples": len(control_scores),
                "variant_samples": len(variant_scores)
            }

        try:
            # t-test実行
            from scipy import stats
            t_statistic, p_value = stats.ttest_ind(control_scores, variant_scores)

            # 効果量計算（Cohen's d）
            control_mean = np.mean(control_scores)
            variant_mean = np.mean(variant_scores)
            control_std = np.std(control_scores, ddof=1)
            variant_std = np.std(variant_scores, ddof=1)

            pooled_std = np.sqrt((control_std**2 + variant_std**2) / 2)
            cohens_d = (variant_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            # 統計的有意性判定
            is_significant = p_value < (1 - self.confidence_level)

            return {
                "test_performed": True,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "is_significant": is_significant,
                "confidence_level": self.confidence_level,
                "effect_size": cohens_d,
                "control_mean": control_mean,
                "variant_mean": variant_mean,
                "control_std": control_std,
                "variant_std": variant_std,
                "improvement": variant_mean - control_mean,
                "relative_improvement": (variant_mean - control_mean) / abs(control_mean) if control_mean != 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"Significance test failed: {e}")
            return {
                "test_performed": False,
                "error": str(e)
            }

    def _determine_test_status(self, test_config: Dict[str, Any]) -> str:
        """テスト状態を判定"""
        significance_tests = test_config.get("significance_tests", [])
        if not significance_tests:
            return "insufficient_data"

        latest_test = significance_tests[-1]
        if not latest_test.get("test_performed", False):
            return "insufficient_data"

        if latest_test.get("is_significant", False):
            # 有意差がある場合
            improvement = latest_test.get("improvement", 0)
            if improvement > 0:
                return "variant_better"
            else:
                return "control_better"
        else:
            # 有意差がない場合
            if len(test_config["control_results"]) >= test_config["sample_size_per_group"]:
                return "no_significant_difference"
            else:
                return "running"

    def _generate_test_recommendations(
        self, test_config: Dict[str, Any], significance_result: Dict[str, Any]
    ) -> List[str]:
        """テスト結果に基づく推奨事項を生成"""
        recommendations = []

        if not significance_result.get("test_performed", False):
            min_sample = getattr(self, "min_sample_size", 30)
            recommendations.append(f"統計的有意差検定を行うための十分なサンプルサイズがありません（必要サンプル数: {min_sample}）")
            return recommendations

        if significance_result.get("is_significant", False):
            improvement = significance_result.get("improvement", 0)
            relative_improvement = significance_result.get("relative_improvement", 0)

            if improvement > 0:
                recommendations.append(
                    f"バリアントが統計的有意に優位（改善: {improvement:.4f}, 相対改善: {relative_improvement:.2%}）"
                )
                recommendations.append("バリアントパラメータの採用を推奨")
            else:
                recommendations.append(
                    f"コントロールが統計的有意に優位（差: {improvement:.4f}）"
                )
                recommendations.append("現在のコントロールパラメータを継続")
        else:
            recommendations.append("統計的有意な差は検出されませんでした")
            recommendations.append("より多くのサンプルでの再テストを検討")

        # 効果量の解釈
        effect_size = significance_result.get("effect_size", 0)
        if abs(effect_size) > 0.8:
            recommendations.append("大きな効果サイズが検出されました")
        elif abs(effect_size) > 0.5:
            recommendations.append("中程度の効果サイズが検出されました")
        elif abs(effect_size) > 0.2:
            recommendations.append("小さな効果サイズが検出されました")

        return recommendations

    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """テスト結果を取得"""
        return self.test_results.get(test_id)

    def get_active_tests(self) -> Dict[str, Dict[str, Any]]:
        """アクティブなテストを取得"""
        return self.active_tests

    def get_ab_testing_summary(self) -> Dict[str, Any]:
        """A/Bテストのサマリーを取得"""
        summary = {
            "total_tests": len(self.test_results),
            "active_tests": len(self.active_tests),
            "completed_tests": len([t for t in self.test_results.values() if t["status"] != "running"]),
            "test_status_distribution": {}
        }

        # ステータス分布
        for test_result in self.test_results.values():
            status = test_result["status"]
            summary["test_status_distribution"][status] = summary["test_status_distribution"].get(status, 0) + 1

        return summary


class AutomaticOptimizationPipeline:
    """
    自動最適化パイプライン
    ハイパーパラメータ、システム、報酬関数の最適化を統合的に実行
    """

    def __init__(self, config: OptimizationConfig, system_optimizer: Optional['SystemOptimizer'] = None) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        # パイプライン設定
        self.stages = ["hyperparameter_optimization", "system_optimization", "reward_optimization"]
        self.max_stage_duration = 3600  # 各ステージの最大実行時間（秒）

        # システムオプティマイザー（UnifiedOptimizerから渡される）
        self.system_optimizer = system_optimizer

    def run_full_pipeline(
        self,
        base_params: Dict[str, Any],
        objective_function: Callable,
        search_space: Dict[str, Any],
        multi_timeframe_functions: Optional[Dict[str, Callable]] = None,
        validation_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        完全自動最適化パイプラインを実行

        Args:
            base_params: ベースパラメータ
            objective_function: 目的関数
            search_space: 探索空間
            multi_timeframe_functions: マルチタイムフレーム用の目的関数（オプション）
            validation_function: 検証関数（オプション）

        Returns:
            パイプライン結果
        """
        self.logger.info("Starting automatic optimization pipeline")

        start_time = time.time()
        # ステージ結果
        stage_results = {}

        # ハイパーパラメータ最適化
        if "hyperparameter_optimization" in self.stages:
            self.logger.info("Stage 1: Hyperparameter Optimization")
            hyperparameter_result = self._optimize_hyperparameters_stage(
                base_params, objective_function, search_space
            )
            stage_results["hyperparameter_optimization"] = hyperparameter_result

            # ベストパラメータをベースパラメータに統合
            base_params.update(hyperparameter_result.best_params)

            # タイムアウトチェック
            if time.time() - start_time > self.max_stage_duration:
                self.logger.warning("Hyperparameter optimization stage timed out")
                return {"success": False, "error": "Timeout in hyperparameter optimization"}

        # システム最適化
        if "system_optimization" in self.stages:
            self.logger.info("Stage 2: System Optimization")
            system_result = self._optimize_system_stage()
            stage_results["system_optimization"] = system_result

        # 報酬関数最適化
        if "reward_optimization" in self.stages:
            self.logger.info("Stage 3: Reward Function Optimization")
            reward_result = self._optimize_reward_stage(
                objective_function, base_params
            )
            stage_results["reward_optimization"] = reward_result

        # マルチタイムフレーム最適化（オプション）
        if multi_timeframe_functions:
            self.logger.info("Multi-timeframe optimization")
            mt_result = self._optimize_multi_timeframe_stage(
                multi_timeframe_functions, search_space
            )
            stage_results["multi_timeframe_optimization"] = mt_result

        # 検証（オプション）
        if validation_function:
            self.logger.info("Validation stage")
            validation_result = self._validate_pipeline(
                base_params, validation_function
            )
            stage_results["validation"] = validation_result

        # 統合結果の生成
        integrated_result = self._generate_integrated_result(stage_results)

        self.logger.info("Automatic optimization pipeline completed")

        return integrated_result

    def _optimize_hyperparameters_stage(
        self, base_params: Dict[str, Any], objective_function: Callable, search_space: Dict[str, Any]
    ) -> OptimizationResult:
        """ハイパーパラメータ最適化ステージ"""
        self.logger.info("Starting hyperparameter optimization")

        # ベースパラメータの設定
        def constrained_objective(params):
            # ベースパラメータを更新
            full_params = base_params.copy()
            full_params.update(params)

            # 目的関数の評価
            return objective_function(full_params)

        # 最適化実行
        optimizer = BayesianOptimizer(self.config)
        result = optimizer.optimize(constrained_objective, search_space)

        self.logger.info(f"Hyperparameter optimization completed. Best score: {result.best_score:.4f}")

        return result

    def _optimize_system_stage(self) -> Dict[str, Any]:
        """システム最適化ステージ"""
        self.logger.info("Starting system optimization")

        result = self.system_optimizer.optimize_memory_usage()

        self.logger.info(f"System optimization completed. Memory status: {result}")

        return result

    def _optimize_reward_stage(
        self, evaluation_function: Callable, base_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """報酬関数最適化ステージ"""
        self.logger.info("Starting reward function optimization")

        # ベースパラメータの設定
        def constrained_reward_objective(params):
            # ベースパラメータを更新
            reward_params = base_params.copy()
            reward_params.update(params)

            # 評価関数の実行
            return evaluation_function(reward_params)

        # 探索空間の定義
        reward_search_space = {
            "profit_weight": {"type": "float", "low": 0.0, "high": 2.0},
            "risk_weight": {"type": "float", "low": 0.0, "high": 2.0},
            "holding_penalty": {"type": "float", "low": 0.0, "high": 0.1},
            "transaction_cost_weight": {"type": "float", "low": 0.0, "high": 0.01},
            "time_decay_factor": {"type": "float", "low": 0.9, "high": 1.0},
            "market_regime_factor": {"type": "float", "low": 0.0, "high": 1.0},
        }

        # 最適化実行
        optimizer = BayesianOptimizer(self.config)
        result = optimizer.optimize(constrained_reward_objective, reward_search_space)

        self.logger.info(f"Reward function optimization completed. Best score: {result.best_score:.4f}")

        return result

    def _optimize_multi_timeframe_stage(
        self, multi_timeframe_functions: Dict[str, Callable], search_space: Dict[str, Any]
    ) -> Dict[str, OptimizationResult]:
        """マルチタイムフレーム最適化ステージ"""
        self.logger.info("Starting multi-timeframe optimization")

        results = {}
        start_time = time.time()

        # 各タイムフレームで個別に最適化
        for tf, objective in multi_timeframe_functions.items():
            self.logger.info(f"Optimizing parameters for {tf} timeframe")

            # タイムフレーム固有の探索空間を使用（デフォルトは共通）
            tf_search_space = search_space.get(tf, search_space.get("default", {}))

            result = self.multi_timeframe_optimizer.optimize(
                {tf: objective}, {tf: tf_search_space}
            )

            results[tf] = result

            self.logger.info(f"{tf} optimization completed. Best score: {result.best_score:.4f}")

        # 統合パラメータの生成
        integrated_params = self.multi_timeframe_optimizer._generate_integrated_parameters(results)

        execution_time = time.time() - start_time
        self.logger.info(f"Multi-timeframe optimization completed in {execution_time:.2f}s")

        # 結果に統合パラメータを追加
        results["integrated"] = OptimizationResult(
            best_params=integrated_params,
            best_score=self.multi_timeframe_optimizer._calculate_integrated_score(results),
            optimization_history=[],
            execution_time=execution_time,
            convergence_info=self.multi_timeframe_optimizer._get_integrated_convergence_info(results),
            recommendations=self.multi_timeframe_optimizer._generate_integrated_recommendations(results)
        )

        return results

    def _validate_pipeline(
        self, base_params: Dict[str, Any], validation_function: Callable
    ) -> Dict[str, Any]:
        """パイプライン検証ステージ"""
        self.logger.info("Starting pipeline validation")

        # ベースパラメータの設定
        def validation_objective(params):
            # ベースパラメータを更新
            test_params = base_params.copy()
            test_params.update(params)

            # 検証関数の実行
            return validation_function(test_params)

        # 探索空間の定義（例として狭い範囲）
        validation_search_space = {
            "profit_weight": {"type": "float", "low": 0.5, "high": 1.5},
            "risk_weight": {"type": "float", "low": 0.5, "high": 1.5},
        }

        # 最適化実行
        optimizer = BayesianOptimizer(self.config)
        result = optimizer.optimize(validation_objective, validation_search_space)

        self.logger.info(f"Pipeline validation completed. Best score: {result.best_score:.4f}")

        return result

    def _generate_integrated_result(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """統合結果を生成"""
        integrated_result = {
            "success": True,
            "stages": {},
            "final_recommendation": {}
        }

        # 各ステージの結果を統合
        for stage, result in stage_results.items():
            integrated_result["stages"][stage] = result

        # 最終推奨事項の生成
        if "reward_optimization" in integrated_result["stages"]:
            reward_result = integrated_result["stages"]["reward_optimization"]
            integrated_result["final_recommendation"] = {
                "action": "deploy",
                "params": reward_result.best_params,
                "justification": "報酬関数最適化に基づく推奨"
            }
        else:
            integrated_result["final_recommendation"] = {
                "action": "continue",
                "reasoning": "最適化が完了しましたが、特定のアクションは推奨されていません"
            }

        return integrated_result


class ParallelOptimizer:
    """
    並列最適化実行器
    複数の最適化タスクを並列で実行
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

        # 並列実行設定
        self.max_workers = min(config.max_parallel_trials, 8)  # 最大8並列
        self.executor = None

        # 実行中のタスク
        self.active_tasks = {}
        self.task_results = {}

    def initialize_executor(self) -> None:
        """実行器を初期化"""
        if self.executor is None:
            try:
                from concurrent.futures import ThreadPoolExecutor
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.info(f"Initialized thread pool executor with {self.max_workers} workers")
            except Exception as e:
                self.logger.warning(f"Failed to initialize parallel executor: {e}")
                self.executor = None

    def run_parallel_optimization(
        self,
        optimization_tasks: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        並列最適化を実行

        Args:
            optimization_tasks: 最適化タスクのリスト
                [{"task_id": str, "optimizer": BaseOptimizer, "objective": Callable, "search_space": Dict, ...}]
            progress_callback: 進捗コールバック関数

        Returns:
            並列実行結果
        """
        if not self.executor:
            self.initialize_executor()

        if not self.executor:
            # フォールバック: 順次実行
            return self._run_sequential_fallback(optimization_tasks, progress_callback)

        self.logger.info(f"Starting parallel optimization with {len(optimization_tasks)} tasks")

        # タスクの実行
        futures = {}
        for task in optimization_tasks:
            task_id = task["task_id"]
            future = self.executor.submit(self._execute_optimization_task, task)
            futures[future] = task_id
            self.active_tasks[task_id] = {"future": future, "task": task}

        # 結果の収集
        results = {}
        completed_count = 0

        for future in futures:
            task_id = futures[future]
            try:
                result = future.result(timeout=3600)  # 1時間タイムアウト
                results[task_id] = result
                completed_count += 1

                if progress_callback:
                    progress_callback(completed_count, len(optimization_tasks))

            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                results[task_id] = {"success": False, "error": str(e)}

        # 結果の統合
        parallel_result = {
            "success": True,
            "total_tasks": len(optimization_tasks),
            "completed_tasks": completed_count,
            "failed_tasks": len(optimization_tasks) - completed_count,
            "results": results,
            "summary": self._generate_parallel_summary(results)
        }

        self.logger.info(f"Parallel optimization completed: {completed_count}/{len(optimization_tasks)} tasks successful")
        return parallel_result

    def _execute_optimization_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """単一の最適化タスクを実行"""
        try:
            optimizer = task["optimizer"]
            objective = task["objective"]
            search_space = task["search_space"]

            # 追加パラメータ
            method = task.get("method", "bayesian")
            task.get("max_trials", self.config.max_trials)

            # 最適化実行
            if hasattr(optimizer, 'optimize_hyperparameters'):
                # UnifiedOptimizerの場合
                result = optimizer.optimize_hyperparameters(objective, search_space, method)
            else:
                # BaseOptimizerの場合
                result = optimizer.optimize(objective, search_space)

            return {
                "success": True,
                "result": result,
                "execution_time": getattr(result, 'execution_time', 0),
                "best_score": getattr(result, 'best_score', 0)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _run_sequential_fallback(
        self,
        optimization_tasks: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """順次実行フォールバック"""
        self.logger.info("Running sequential optimization (parallel execution not available)")

        results = {}
        completed_count = 0

        for task in optimization_tasks:
            task_id = task["task_id"]
            result = self._execute_optimization_task(task)
            results[task_id] = result

            if result["success"]:
                completed_count += 1

            if progress_callback:
                progress_callback(completed_count, len(optimization_tasks))

        return {
            "success": True,
            "total_tasks": len(optimization_tasks),
            "completed_tasks": completed_count,
            "failed_tasks": len(optimization_tasks) - completed_count,
            "results": results,
            "summary": self._generate_parallel_summary(results),
            "execution_mode": "sequential"
        }

    def _generate_parallel_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """並列実行のサマリーを生成"""
        summary = {
            "best_overall_score": float("-inf"),
            "best_task_id": None,
            "average_execution_time": 0,
            "success_rate": 0,
            "performance_distribution": {}
        }

        successful_tasks = []
        execution_times = []

        for task_id, result in results.items():
            if result.get("success", False):
                successful_tasks.append(task_id)
                execution_times.append(result.get("execution_time", 0))

                score = result.get("best_score", float("-inf"))
                if score > summary["best_overall_score"]:
                    summary["best_overall_score"] = score
                    summary["best_task_id"] = task_id

        if successful_tasks:
            summary["success_rate"] = len(successful_tasks) / len(results)
            summary["average_execution_time"] = np.mean(execution_times) if execution_times else 0

            # パフォーマンス分布
            scores = [results[tid].get("best_score", 0) for tid in successful_tasks]
            if scores:
                summary["performance_distribution"] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores)
                }

        return summary

    def run_parameter_sweep(
        self,
        base_optimizer: BaseOptimizer,
        objective_function: Callable,
        parameter_combinations: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        パラメータスイープを実行

        Args:
            base_optimizer: ベース最適化器
            objective_function: 目的関数
            parameter_combinations: パラメータ組み合わせのリスト
            progress_callback: 進捗コールバック

        Returns:
            スイープ結果
        """
        self.logger.info(f"Running parameter sweep with {len(parameter_combinations)} combinations")

        # タスクの作成
        sweep_tasks = []
        for i, params in enumerate(parameter_combinations):
            # パラメータを固定した目的関数
            def fixed_objective(fixed_params=params):
                return objective_function(fixed_params)

            task = {
                "task_id": f"sweep_{i:03d}",
                "optimizer": base_optimizer,
                "objective": fixed_objective,
                "search_space": {},  # 空の探索空間（パラメータ固定）
                "fixed_params": params
            }
            sweep_tasks.append(task)

        # 並列実行
        sweep_result = self.run_parallel_optimization(sweep_tasks, progress_callback)

        # スイープ特化のサマリー
        sweep_summary = self._generate_sweep_summary(sweep_result, parameter_combinations)
        sweep_result["sweep_summary"] = sweep_summary

        return sweep_result

    def _generate_sweep_summary(
        self,
        sweep_result: Dict[str, Any],
        parameter_combinations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """パラメータスイープのサマリーを生成"""
        summary = {
            "total_combinations": len(parameter_combinations),
            "parameter_ranges": {},
            "best_combination": None,
            "best_score": float("-inf"),
            "parameter_importance": {}
        }

        # パラメータ範囲の分析
        all_params = set()
        for combo in parameter_combinations:
            all_params.update(combo.keys())

        for param in all_params:
            values = [combo.get(param) for combo in parameter_combinations if param in combo]
            if values:
                summary["parameter_ranges"][param] = {
                    "min": min(values),
                    "max": max(values),
                    "unique_values": len(set(values))
                }

        # 最適組み合わせの特定
        results = sweep_result.get("results", {})
        for task_id, result in results.items():
            if result.get("success", False):
                score = result.get("best_score", float("-inf"))
                if score > summary["best_score"]:
                    summary["best_score"] = score
                    task_idx = int(task_id.split("_")[1])
                    summary["best_combination"] = parameter_combinations[task_idx]

        # パラメータ重要度の分析（相関分析）
        if summary["best_combination"]:
            summary["parameter_importance"] = self._analyze_parameter_importance(
                parameter_combinations, results
            )

        return summary

    def _analyze_parameter_importance(
        self,
        parameter_combinations: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> Dict[str, float]:
        """パラメータ重要度を分析"""
        importance = {}

        # スコアを取得
        scores = []
        valid_combinations = []

        for i, combo in enumerate(parameter_combinations):
            task_id = f"sweep_{i:03d}"
            result = results.get(task_id, {})
            if result.get("success", False):
                score = result.get("best_score", 0)
                scores.append(score)
                valid_combinations.append(combo)

        if not scores or len(valid_combinations) < 2:
            return importance

        # 各パラメータの相関係数を計算
        for param in valid_combinations[0].keys():
            param_values = []
            for combo in valid_combinations:
                if param in combo:
                    param_values.append(combo[param])

            if len(param_values) == len(scores):
                try:
                    correlation = np.corrcoef(param_values, scores)[0, 1]
                    importance[param] = abs(correlation)  # 絶対値
                except Exception:
                    importance[param] = 0

        return importance

    def get_active_task_status(self) -> Dict[str, Any]:
        """アクティブなタスクの状態を取得"""
        status = {
            "active_tasks": len(self.active_tasks),
            "task_details": {}
        }

        for task_id, task_info in self.active_tasks.items():
            future = task_info["future"]
            status["task_details"][task_id] = {
                "running": not future.done(),
                "done": future.done(),
                "task_type": task_info["task"]["optimizer"].__class__.__name__
            }

        return status

    def shutdown(self) -> None:
        """実行器をシャットダウン"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            self.logger.info("Parallel optimizer executor shut down")


class OptimizationResultPersistence:
    """
    最適化結果の持続化とバージョン管理
    """

    def __init__(self, base_dir: str = "optimization_results") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)

        # バージョン管理
        self.version_file = self.base_dir / "versions.json"
        self.current_version = self._load_current_version()

    def _load_current_version(self) -> int:
        """現在のバージョンを読み込み"""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("current_version", 0)
            except Exception as e:
                self.logger.warning(f"Failed to load version file: {e}")
        return 0

    def _save_current_version(self) -> None:
        """現在のバージョンを保存"""
        version_data = {
            "current_version": self.current_version,
            "last_updated": datetime.now().isoformat(),
            "total_versions": len(self._list_versions())
        }

        with open(self.version_file, 'w', encoding='utf-8') as f:
            safe_json_dump(version_data, self.version_file, indent=2, ensure_ascii=False)

    def save_optimization_result(
        self,
        result: Dict[str, Any],
        result_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        最適化結果を保存

        Args:
            result: 最適化結果
            result_type: 結果タイプ ("hyperparameter", "multi_timeframe", "ab_test", etc.)
            metadata: 追加メタデータ
            tags: タグリスト

        Returns:
            バージョンID
        """
        self.current_version += 1
        version_id = f"v{self.current_version:04d}"

        # 結果データの作成
        result_data = {
            "version_id": version_id,
            "result_type": result_type,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "metadata": metadata or {},
            "tags": tags or [],
            "performance_metrics": self._extract_performance_metrics(result)
        }

        # ファイル保存
        timestamp_str = datetime.now().strftime("%Y%m%d")
        filename = f"{result_type}_{version_id}_{timestamp_str}.json"
        filepath = self.base_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            safe_json_dump(result_data, filepath, indent=2, ensure_ascii=False, default=str)

        # バージョン情報を更新
        self._save_current_version()

        # インデックス更新
        self._update_index(version_id, result_data)

        self.logger.info(f"Saved optimization result: {version_id} ({result_type})")
        return version_id

    def _extract_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """結果からパフォーマンスメトリクスを抽出"""
        metrics = {}

        # スコア関連
        if "best_score" in result:
            metrics["best_score"] = result["best_score"]
        if "improvement" in result:
            metrics["improvement"] = result["improvement"]
        if "relative_improvement" in result:
            metrics["relative_improvement"] = result["relative_improvement"]

        # 時間関連
        if "execution_time" in result:
            metrics["execution_time"] = result["execution_time"]

        # 統計的有意性
        if "is_significant" in result:
            metrics["is_significant"] = result["is_significant"]
        if "p_value" in result:
            metrics["p_value"] = result["p_value"]

        return metrics

    def _update_index(self, version_id: str, result_data: Dict[str, Any]):
        """検索インデックスを更新"""
        index_file = self.base_dir / "index.json"

        # 既存インデックス読み込み
        index = {}
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load index: {e}")

        # 新しいエントリ追加
        index[version_id] = {
            "result_type": result_data["result_type"],
            "timestamp": result_data["timestamp"],
            "tags": result_data["tags"],
            "performance_metrics": result_data["performance_metrics"],
            "filename": f"{result_data['result_type']}_{version_id}_{result_data['timestamp'][:10].replace('-', '')}.json"
        }

        # インデックス保存
        safe_json_dump(index, index_file, indent=2, ensure_ascii=False)

    def load_optimization_result(self, version_id: str) -> Optional[Dict[str, Any]]:
        """最適化結果を読み込み"""
        index = self._load_index()
        if version_id not in index:
            return None

        entry = index[version_id]
        filepath = self.base_dir / entry["filename"]

        if not filepath.exists():
            self.logger.warning(f"Result file not found: {filepath}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load result {version_id}: {e}")
            return None

    def _load_index(self) -> Dict[str, Any]:
        """インデックスを読み込み"""
        index_file = self.base_dir / "index.json"
        if not index_file.exists():
            return {}

        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load index: {e}")
            return {}

    def search_results(
        self,
        result_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_score: Optional[float] = None,
        max_p_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        結果を検索

        Args:
            result_type: 結果タイプでフィルタ
            tags: タグでフィルタ
            date_from: 開始日（YYYY-MM-DD）
            date_to: 終了日（YYYY-MM-DD）
            min_score: 最小スコア
            max_p_value: 最大p値

        Returns:
            マッチした結果のリスト
        """
        index = self._load_index()
        results = []

        for version_id, entry in index.items():
            # タイプフィルタ
            if result_type and entry["result_type"] != result_type:
                continue

            # タグフィルタ
            if tags:
                if not all(tag in entry["tags"] for tag in tags):
                    continue

            # 日付フィルタ
            timestamp = entry["timestamp"]
            if date_from and timestamp < date_from:
                continue
            if date_to and timestamp > date_to:
                continue

            # パフォーマンスフィルタ
            metrics = entry["performance_metrics"]
            if min_score is not None and metrics.get("best_score", 0) < min_score:
                continue
            if max_p_value is not None and metrics.get("p_value", 1.0) > max_p_value:
                continue

            results.append({
                "version_id": version_id,
                **entry
            })

        # 新しい順にソート
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results

    def get_result_comparison(
        self,
        version_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        複数バージョンの結果を比較

        Args:
            version_ids: 比較するバージョンIDリスト
            metrics: 比較するメトリクス

        Returns:
            比較結果
        """
        comparison = {
            "versions": version_ids,
            "results": {},
            "comparison": {}
        }

        default_metrics = ["best_score", "execution_time", "improvement", "p_value"]
        compare_metrics = metrics or default_metrics

        # 各バージョンの結果を取得
        for vid in version_ids:
            result = self.load_optimization_result(vid)
            if result:
                comparison["results"][vid] = result

                # メトリクス抽出
                result_metrics = {}
                perf_metrics = result.get("performance_metrics", {})
                for metric in compare_metrics:
                    result_metrics[metric] = perf_metrics.get(metric)

                comparison["results"][vid]["comparison_metrics"] = result_metrics

        # 比較分析
        if len(comparison["results"]) > 1:
            comparison["comparison"] = self._analyze_comparison(comparison["results"], compare_metrics)

        return comparison

    def _analyze_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """比較分析を実行"""
        analysis = {}

        for metric in metrics:
            values = []
            versions = []

            for vid, result in results.items():
                value = result.get("comparison_metrics", {}).get(metric)
                if value is not None:
                    values.append(value)
                    versions.append(vid)

            if values:
                analysis[metric] = {
                    "values": dict(zip(versions, values)),
                    "best_version": versions[np.argmax(values)] if metric != "p_value" else versions[np.argmin(values)],
                    "best_value": max(values) if metric != "p_value" else min(values),
                    "mean": np.mean(values),
                    "std": np.std(values)
                }

        return analysis

    def _list_versions(self) -> List[str]:
        """全バージョンをリスト"""
        index = self._load_index()
        return list(index.keys())

    def cleanup_old_versions(self, keep_recent: int = 50) -> None:
        """古いバージョンを削除"""
        versions = self._list_versions()
        if len(versions) <= keep_recent:
            return

        # バージョン番号でソート（新しい順）
        versions.sort(key=lambda x: int(x[1:]), reverse=True)

        # 削除対象
        to_delete = versions[keep_recent:]

        for version_id in to_delete:
            index = self._load_index()
            if version_id in index:
                filename = index[version_id]["filename"]
                filepath = self.base_dir / filename

                # ファイルを削除
                if filepath.exists():
                    filepath.unlink()

                # インデックスから削除
                del index[version_id]
                index_file = self.base_dir / "index.json"
                safe_json_dump(index, index_file, indent=2, ensure_ascii=False)

        self.logger.info(f"Cleaned up {len(to_delete)} old versions")

    def get_persistence_stats(self) -> Dict[str, Any]:
        """持続化統計を取得"""
        index = self._load_index()
        stats = {
            "total_versions": len(index),
            "result_types": {},
            "date_range": {},
            "storage_size_mb": 0
        }

        if index:
            # 結果タイプ分布
            for entry in index.values():
                rt = entry["result_type"]
                stats["result_types"][rt] = stats["result_types"].get(rt, 0) + 1

            # 日付範囲
            timestamps = [entry["timestamp"] for entry in index.values()]
            stats["date_range"] = {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }

            # ストレージサイズ
            total_size = 0
            for entry in index.values():
                filepath = self.base_dir / entry["filename"]
                if filepath.exists():
                    total_size += filepath.stat().st_size

            stats["storage_size_mb"] = total_size / BYTES_PER_MB

        return stats


def example_usage() -> None:
    """使い方の例"""
    config = OptimizationConfig(
        enable_hyperparameter_optimization=True,
        enable_system_optimization=True,
        enable_reward_optimization=True,
        max_trials=10,
        max_parallel_trials=2,
    )

    optimizer = UnifiedOptimizer(config)

    # ハイパーパラメータ最適化の例
    def objective_function(params):
        x = params.get("x", 0)
        y = params.get("y", 0)
        return -(x - 2) ** 2 - (y - 3) ** 2

    search_space = {
        "x": {"type": "float", "low": 0, "high": 5},
        "y": {"type": "float", "low": 0, "high": 5}
    }

    result = optimizer.optimize_hyperparameters(objective_function, search_space)
    print("Best hyperparameters:", result.best_params)

    # システム最適化の例
    system_result = optimizer.optimize_system()
    print("System optimization result:", system_result)

    # 報酬関数最適化の例
    reward_result = optimizer.optimize_reward_function(
        evaluation_function=objective_function,
        current_reward_config={"profit_weight": 1.0, "risk_weight": 1.0},
    )
    print("Reward function optimization result:", reward_result)

    # マルチタイムフレーム最適化の例
    mt_result = optimizer.optimize_multi_timeframe(
        objective_functions={"1m": objective_function, "5m": objective_function},
        search_spaces={
            "1m": {"x": {"type": "float", "low": 0, "high": 5}},
            "5m": {"y": {"type": "float", "low": 0, "high": 5}},
        },
    )
    print("Multi-timeframe optimization result:", mt_result)

    # A/Bテストの例
    test_id = optimizer.create_ab_test(
        "test_ab",
        control_params={"x": 1.0},
        variant_params={"x": 2.0},
        evaluation_function=objective_function,
    )
    ab_result = optimizer.run_ab_test(test_id, num_iterations=5)
    print("A/B test result:", ab_result)

    # 自動最適化パイプラインの例
    pipeline_result = optimizer.run_automatic_pipeline(
        base_params={"x": 1.0},
        objective_function=objective_function,
        search_space=search_space,
    )
    print("Automatic optimization pipeline result:", pipeline_result)

    # 結果の保存と読み込みの例
    version_id = optimizer.save_result_to_version_control(
        {"test_metric": 0.95}, "test_result"
    )
    loaded_result = optimizer.load_result_from_version_control(version_id)
    print("Loaded result:", loaded_result)


if __name__ == "__main__":
    example_usage()
