"""
ハイパーパラメータ最適化の基底クラスと共通データ構造

全ての最適化手法が継承する基底クラスと、
結果を保存するためのデータ構造を定義します。
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ParameterType(Enum):
    """パラメータの型"""

    CONTINUOUS = "continuous"  # 連続値 (e.g., learning_rate: 0.0001-0.01)
    INTEGER = "integer"  # 整数 (e.g., batch_size: 64, 128, 256)
    CATEGORICAL = "categorical"  # カテゴリカル (e.g., optimizer: 'adam', 'sgd')
    LOG_UNIFORM = "log_uniform"  # 対数スケール (e.g., learning_rate: 1e-5 to 1e-2)


@dataclass
class ParameterSpace:
    """探索するパラメータ空間の定義"""

    name: str
    param_type: ParameterType
    low: Optional[float] = None  # 最小値（CONTINUOUS, INTEGER, LOG_UNIFORMで使用）
    high: Optional[float] = None  # 最大値（CONTINUOUS, INTEGER, LOG_UNIFORMで使用）
    choices: Optional[List[Any]] = None  # 選択肢（CATEGORICALで使用）
    default: Optional[Any] = None  # デフォルト値

    def __post_init__(self) -> None:
        """バリデーション"""
        if self.param_type in [
            ParameterType.CONTINUOUS,
            ParameterType.INTEGER,
            ParameterType.LOG_UNIFORM,
        ]:
            if self.low is None or self.high is None:
                raise ValueError(
                    f"{self.name}: low and high must be specified for {self.param_type}"
                )
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"{self.name}: choices must be specified for CATEGORICAL"
                )

    def sample(self, method: str = "random") -> Any:
        """パラメータをサンプリング"""
        import random

        import numpy as np

        if self.param_type == ParameterType.CONTINUOUS:
            return random.uniform(self.low, self.high)
        elif self.param_type == ParameterType.INTEGER:
            return random.randint(int(self.low), int(self.high))
        elif self.param_type == ParameterType.LOG_UNIFORM:
            log_low = np.log10(self.low)
            log_high = np.log10(self.high)
            return 10 ** random.uniform(log_low, log_high)
        elif self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.choices)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "name": self.name,
            "param_type": self.param_type.value,
            "low": self.low,
            "high": self.high,
            "choices": self.choices,
            "default": self.default,
        }


@dataclass
class TrialResult:
    """1回の試行（トライアル）の結果"""

    trial_id: int
    parameters: Dict[str, Any]
    metrics: Dict[str, float]  # e.g., {'critic_loss': 0.08, 'actor_loss': -4.26}
    objective_value: float  # 最適化したい目的関数の値（小さいほど良い）
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "trial_id": self.trial_id,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "objective_value": self.objective_value,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class OptimizationResult:
    """最適化全体の結果"""

    optimizer_name: str
    best_parameters: Dict[str, Any]
    best_objective_value: float
    best_metrics: Dict[str, float]
    all_trials: List[TrialResult]
    total_duration_seconds: float
    n_trials: int
    success_rate: float

    # 統計情報
    objective_mean: float = 0.0
    objective_std: float = 0.0
    objective_min: float = float("inf")
    objective_max: float = float("-inf")

    def __post_init__(self) -> None:
        """統計情報を計算"""
        successful_trials = [t for t in self.all_trials if t.success]
        if successful_trials:
            objectives = [t.objective_value for t in successful_trials]
            import numpy as np

            self.objective_mean = np.mean(objectives)
            self.objective_std = np.std(objectives)
            self.objective_min = min(objectives)
            self.objective_max = max(objectives)

    def save(self, output_path: Path) -> None:
        """結果をJSONファイルに保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "optimizer_name": self.optimizer_name,
            "best_parameters": self.best_parameters,
            "best_objective_value": self.best_objective_value,
            "best_metrics": self.best_metrics,
            "total_duration_seconds": self.total_duration_seconds,
            "n_trials": self.n_trials,
            "success_rate": self.success_rate,
            "statistics": {
                "objective_mean": self.objective_mean,
                "objective_std": self.objective_std,
                "objective_min": self.objective_min,
                "objective_max": self.objective_max,
            },
            "all_trials": [t.to_dict() for t in self.all_trials],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ 最適化結果を保存: {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> "OptimizationResult":
        """JSONファイルから結果を読み込み"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_trials = [
            TrialResult(
                trial_id=t["trial_id"],
                parameters=t["parameters"],
                metrics=t["metrics"],
                objective_value=t["objective_value"],
                duration_seconds=t["duration_seconds"],
                success=t["success"],
                error_message=t.get("error_message"),
                metadata=t.get("metadata", {}),
            )
            for t in data["all_trials"]
        ]

        return cls(
            optimizer_name=data["optimizer_name"],
            best_parameters=data["best_parameters"],
            best_objective_value=data["best_objective_value"],
            best_metrics=data["best_metrics"],
            all_trials=all_trials,
            total_duration_seconds=data["total_duration_seconds"],
            n_trials=data["n_trials"],
            success_rate=data["success_rate"],
        )

    def print_summary(self) -> None:
        """結果のサマリーを表示"""
        print("=" * 80)
        print(f"  {self.optimizer_name} - 最適化結果")
        print("=" * 80)
        print()
        print("🎯 ベストパラメータ:")
        for param, value in self.best_parameters.items():
            print(f"  {param}: {value}")
        print()
        print("📊 ベストメトリクス:")
        for metric, value in self.best_metrics.items():
            print(f"  {metric}: {value:.6f}")
        print()
        print("📈 統計:")
        print(f"  試行回数: {self.n_trials}")
        print(f"  成功率: {self.success_rate:.1%}")
        print(f"  目的関数値: {self.best_objective_value:.6f}")
        print(f"  平均: {self.objective_mean:.6f} ± {self.objective_std:.6f}")
        print(f"  範囲: [{self.objective_min:.6f}, {self.objective_max:.6f}]")
        print(f"  総所要時間: {self.total_duration_seconds:.1f}秒")
        print("=" * 80)


class OptimizerBase(ABC):
    """
    ハイパーパラメータ最適化の基底クラス

    全ての最適化手法はこのクラスを継承して実装します。
    """

    def __init__(
        self,
        parameter_spaces: List[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], TrialResult],
        n_trials: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Args:
            parameter_spaces: 探索するパラメータ空間のリスト
            objective_function: パラメータを受け取り、TrialResultを返す関数
            n_trials: 試行回数（手法によっては不要）
            random_state: 乱数シード
        """
        self.parameter_spaces = {p.name: p for p in parameter_spaces}
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.random_state = random_state

        # 結果保存用
        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        self.start_time: Optional[float] = None

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """
        最適化を実行

        Returns:
            OptimizationResult: 最適化結果
        """
        pass

    def _run_trial(self, trial_id: int, parameters: Dict[str, Any]) -> TrialResult:
        """
        1回の試行を実行

        Args:
            trial_id: トライアルID
            parameters: 試行するパラメータ

        Returns:
            TrialResult: 試行結果
        """
        print(
            f"Trial {trial_id}/{self.n_trials if self.n_trials else '?'}: {parameters}"
        )

        # numpy型を標準Python型に変換
        clean_params = {}
        for key, value in parameters.items():
            if hasattr(value, "item"):  # numpy型
                clean_params[key] = value.item()
            else:
                clean_params[key] = value

        trial_start = time.time()
        try:
            objective_value = self.objective_function(clean_params)
            duration = time.time() - trial_start

            # floatが返された場合、TrialResultを作成
            if isinstance(objective_value, (int, float)):
                result = TrialResult(
                    trial_id=trial_id,
                    parameters=clean_params,
                    objective_value=float(objective_value),
                    metrics={},
                    duration_seconds=duration,
                    success=True,
                    error_message=None,
                )
            else:
                # 既にTrialResultの場合
                result = objective_value
                result.trial_id = trial_id
                result.duration_seconds = duration

            # ベストトライアルの更新
            if (
                self.best_trial is None
                or result.objective_value < self.best_trial.objective_value
            ):
                self.best_trial = result
                print(f"  ✨ New best! Objective: {result.objective_value:.6f}")

            self.trials.append(result)
            return result

        except Exception as e:
            duration = time.time() - trial_start
            error_result = TrialResult(
                trial_id=trial_id,
                parameters=clean_params,
                metrics={},
                objective_value=float("inf"),
                duration_seconds=duration,
                success=False,
                error_message=str(e),
            )
            self.trials.append(error_result)
            print(f"  ❌ Failed: {str(e)}")
            return error_result

    def _create_result(self) -> OptimizationResult:
        """OptimizationResultを作成"""
        total_duration = time.time() - self.start_time if self.start_time else 0
        successful_trials = [t for t in self.trials if t.success]
        success_rate = len(successful_trials) / len(self.trials) if self.trials else 0

        return OptimizationResult(
            optimizer_name=self.__class__.__name__,
            best_parameters=self.best_trial.parameters if self.best_trial else {},
            best_objective_value=self.best_trial.objective_value
            if self.best_trial
            else float("inf"),
            best_metrics=self.best_trial.metrics if self.best_trial else {},
            all_trials=self.trials,
            total_duration_seconds=total_duration,
            n_trials=len(self.trials),
            success_rate=success_rate,
        )
