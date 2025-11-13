#!/usr/bin/env python3
"""
V433 Unified Optimizer: 統合最適化システム
現実データ中心主義に基づく包括的な最適化フレームワーク
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ztb.trading.environment.constants import BYTES_PER_MB
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 依存ライブラリのチェック
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Bayesian optimization will be disabled.")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available. Progress bars will be disabled.")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. System monitoring will be limited.")


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
        self.logger = get_logger(self.__class__.__name__)

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


class SystemOptimizer:
    """システムレベル最適化器"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # メモリトラッカー
        self.memory_usage = []
        self.gc_counter = 0

        # パフォーマンスモニター
        self.performance_metrics = []

        # I/Oキャッシュ
        self.cache = {}
        self.cache_timestamps = {}

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量の最適化"""
        if not PSUTIL_AVAILABLE:
            return {"status": "skipped", "reason": "psutil not available"}

        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            current_memory_mb = memory_info.rss / BYTES_PER_MB
            self.memory_usage.append(current_memory_mb)

            # メモリ閾値チェック
            if current_memory_mb > self.config.memory_threshold_mb:
                self._trigger_gc()
                return {
                    "status": "optimized",
                    "action": "garbage_collection",
                    "memory_before": current_memory_mb,
                    "memory_after": process.memory_info().rss / BYTES_PER_MB,
                }

            return {"status": "normal", "current_memory_mb": current_memory_mb}

        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return {"status": "error", "error": str(e)}

    def _trigger_gc(self):
        """ガベージコレクションを実行"""
        import gc

        collected = gc.collect()
        self.gc_counter += 1

        if self.gc_counter % 10 == 0:  # 10回ごとにログ
            self.logger.info(
                f"Garbage collection triggered (collected: {collected} objects)"
            )

    def optimize_io_operations(self, key: str, data_generator: Callable) -> Any:
        """I/O操作の最適化（キャッシュ使用）"""
        if not self.config.enable_io_caching:
            return data_generator()

        current_time = time.time()

        # キャッシュチェック
        if key in self.cache:
            cache_time = self.cache_timestamps.get(key, 0)
            if current_time - cache_time < self.config.cache_ttl_seconds:
                return self.cache[key]

        # データ生成
        data = data_generator()

        # キャッシュ保存
        self.cache[key] = data
        self.cache_timestamps[key] = current_time

        return data

    def monitor_performance(self, step: int, metrics: Dict[str, Any]):
        """パフォーマンス監視"""
        if not self.config.enable_performance_monitoring:
            return

        if step % self.config.performance_log_interval == 0:
            metrics_with_timestamp = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **metrics,
            }
            self.performance_metrics.append(metrics_with_timestamp)

            # 定期的なログ出力
            self.logger.info(f"Performance at step {step}: {metrics}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        status = {
            "memory_usage_history": self.memory_usage[-10:]
            if self.memory_usage
            else [],
            "gc_count": self.gc_counter,
            "cache_size": len(self.cache),
            "performance_metrics_count": len(self.performance_metrics),
        }

        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                status.update(
                    {
                        "current_memory_mb": memory_info.rss / BYTES_PER_MB,
                        "cpu_percent": process.cpu_percent(interval=0.1),
                    }
                )
            except Exception as e:
                status["system_monitoring_error"] = str(e)

        return status


class RewardFunctionOptimizer:
    """報酬関数最適化器"""

    def __init__(self, config: OptimizationConfig):
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

        def reward_objective(params):
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


class UnifiedOptimizer:
    """
    V433統合最適化システム
    ハイパーパラメータ、システム、報酬関数を統合した最適化
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # 各最適化器の初期化
        self.hyperparameter_optimizer = None
        self.system_optimizer = SystemOptimizer(config)
        self.reward_optimizer = RewardFunctionOptimizer(config)

        # 最適化履歴
        self.optimization_history = []
        self.current_best_params = {}

    def initialize_hyperparameter_optimizer(self, method: str = "bayesian"):
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

    def save_optimization_results(self, filepath: str):
        """最適化結果を保存"""
        results = {
            "config": self.config.__dict__,
            "current_best_params": self.current_best_params,
            "optimization_history": self.optimization_history,
            "summary": self.get_optimization_summary(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Optimization results saved to {filepath}")

    def load_optimization_results(self, filepath: str):
        """最適化結果を読み込み"""
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.current_best_params = results.get("current_best_params", {})
        self.optimization_history = results.get("optimization_history", [])

        self.logger.info(f"Optimization results loaded from {filepath}")


def create_unified_optimizer(
    config: Optional[OptimizationConfig] = None,
) -> UnifiedOptimizer:
    """UnifiedOptimizerのファクトリ関数"""
    if config is None:
        config = OptimizationConfig()

    return UnifiedOptimizer(config)


# 使用例とテスト関数
def example_usage():
    """使用例"""
    # 設定の作成
    config = OptimizationConfig(
        enable_hyperparameter_optimization=True,
        enable_system_optimization=True,
        enable_reward_optimization=True,
        max_trials=50,
        max_parallel_trials=2,
    )

    # UnifiedOptimizerの作成
    optimizer = create_unified_optimizer(config)

    # ハイパーパラメータ最適化の例
    def dummy_objective(params):
        # ダミーの目的関数
        learning_rate = params.get("learning_rate", 0.001)
        batch_size = params.get("batch_size", 32)
        return -((learning_rate - 0.01) ** 2) - (batch_size - 64) ** 2 / 1000  # 最大化

    search_space = {
        "learning_rate": {"type": "float", "low": 0.0001, "high": 0.1},
        "batch_size": {"type": "int", "low": 16, "high": 128},
    }

    # 最適化実行
    result = optimizer.optimize_hyperparameters(dummy_objective, search_space)
    print(f"Best parameters: {result.best_params}")
    print(f"Best score: {result.best_score}")

    # システム最適化
    system_result = optimizer.optimize_system()
    print(f"System optimization: {system_result}")

    # 結果保存
    optimizer.save_optimization_results("optimization_results.json")


if __name__ == "__main__":
    example_usage()
