"""
SAC専用のハイパーパラメータ最適化ユーティリティ

SACアルゴリズムに特化したパラメータ空間定義と目的関数を提供します。
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict
from enum import Enum

from ztb.types.common import ConfigDict
from ztb.utils.file_utils import safe_json_dump


class ParameterType(Enum):
    CONTINUOUS = "continuous"
    LOG_UNIFORM = "log_uniform"
    CATEGORICAL = "categorical"


class ParameterSpace:
    def __init__(self, name: str, param_type: ParameterType, low: float = None, high: float = None, choices: list = None, default: float = None):
        self.name = name
        self.param_type = param_type
        self.low = low
        self.high = high
        self.choices = choices
        self.default = default


class TrialResult:
    def __init__(self, trial_id: int, parameters: dict, metrics: dict, objective_value: float, duration_seconds: float, success: bool, error_message: str = None):
        self.trial_id = trial_id
        self.parameters = parameters
        self.metrics = metrics
        self.objective_value = objective_value
        self.duration_seconds = duration_seconds
        self.success = success
        self.error_message = error_message


def get_sac_parameter_spaces(preset: str = "full") -> Dict[str, ParameterSpace]:
    """
    SAC用のパラメータ空間定義

    Args:
        preset: プリセット名
                - 'full': 全パラメータ（計算コスト大）
                - 'essential': 重要なパラメータのみ
                - 'learning': 学習率関連のみ
                - 'buffer': バッファ関連のみ

    Returns:
        Dict[str, ParameterSpace]: パラメータ空間の辞書
    """

    # 全パラメータ空間
    all_spaces = {
        # 学習率
        "learning_rate": ParameterSpace(
            name="learning_rate",
            param_type=ParameterType.LOG_UNIFORM,
            low=1e-5,
            high=1e-2,
            default=3e-4,
        ),
        # バッチサイズ
        "batch_size": ParameterSpace(
            name="batch_size",
            param_type=ParameterType.CATEGORICAL,
            choices=[32, 64, 128, 256, 512],
            default=128,
        ),
        # バッファサイズ
        "buffer_size": ParameterSpace(
            name="buffer_size",
            param_type=ParameterType.CATEGORICAL,
            choices=[10000, 20000, 50000, 100000],
            default=20000,
        ),
        # 割引率
        "gamma": ParameterSpace(
            name="gamma",
            param_type=ParameterType.CONTINUOUS,
            low=0.95,
            high=0.9999,
            default=0.99,
        ),
        # Soft update係数
        "tau": ParameterSpace(
            name="tau",
            param_type=ParameterType.CONTINUOUS,
            low=0.001,
            high=0.01,
            default=0.005,
        ),
        # Target Entropy（自動の場合は-action_dim）
        "target_entropy": ParameterSpace(
            name="target_entropy",
            param_type=ParameterType.CONTINUOUS,
            low=-3.0,
            high=-0.5,
            default=-1.0,
        ),
        # 学習開始ステップ
        "learning_starts": ParameterSpace(
            name="learning_starts",
            param_type=ParameterType.CATEGORICAL,
            choices=[100, 500, 1000, 2000],
            default=500,
        ),
        # 訓練頻度
        "train_freq": ParameterSpace(
            name="train_freq",
            param_type=ParameterType.CATEGORICAL,
            choices=[1, 2, 4],
            default=1,
        ),
        # 勾配ステップ数
        "gradient_steps": ParameterSpace(
            name="gradient_steps",
            param_type=ParameterType.CATEGORICAL,
            choices=[1, 2, 4],
            default=1,
        ),
    }

    # プリセットごとに返すパラメータを選択
    if preset == "full":
        return all_spaces

    elif preset == "essential":
        # 最も重要なパラメータのみ
        return {
            k: v
            for k, v in all_spaces.items()
            if k in ["learning_rate", "batch_size", "gamma", "tau"]
        }

    elif preset == "learning":
        # 学習率関連
        return {
            k: v
            for k, v in all_spaces.items()
            if k in ["learning_rate", "learning_starts", "train_freq", "gradient_steps"]
        }

    elif preset == "buffer":
        # バッファ関連
        return {
            k: v
            for k, v in all_spaces.items()
            if k in ["buffer_size", "batch_size", "learning_starts"]
        }

    else:
        raise ValueError(f"Unknown preset: {preset}")


def create_sac_objective_function(
    base_config_path: Path,
    total_timesteps: int = 5000,
    metric: str = "critic_loss",
    lower_is_better: bool = True,
) -> Callable[[dict[str, ConfigDict]], TrialResult]:
    """
    SAC training objective function creator.

    Args:
        base_config_path: Base config file path
        total_timesteps: Training timesteps
        metric: Metric to optimize ('critic_loss', 'actor_loss', 'episode_reward', etc.)
        lower_is_better: Whether lower metric values are better

    Returns:
        Callable: Objective function that takes parameters and returns TrialResult
    """

    # ベース設定を読み込み
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    def objective_function(parameters: ConfigDict) -> TrialResult:
        """
        指定されたパラメータでSACを訓練し、結果を返す

        Args:
            parameters: ハイパーパラメータの辞書

        Returns:
            TrialResult: 訓練結果
        """
        import tempfile
        import time

        # 設定をマージ
        config = base_config.copy()
        config["total_timesteps"] = total_timesteps

        # SACハイパーパラメータを更新
        sac_key = "sac_hyperparameters"
        if sac_key not in config and "sac_params" in config:
            sac_key = "sac_params"
        if sac_key not in config:
            config[sac_key] = {}

        for param_name, param_value in parameters.items():
            config[sac_key][param_name] = param_value

        # 一時的な設定ファイルを作成
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            safe_json_dump(config, f.name, indent=2)
            temp_config_path = f.name

        try:
            # 訓練を実行
            start_time = time.time()

            # train_v395i.pyを実行（または適切な訓練スクリプト）
            result = subprocess.run(
                [
                    sys.executable,
                    "ztb/training/scripts/train_v395i.py",
                    "--config",
                    temp_config_path,
                ],
                capture_output=True,
                text=True,
                timeout=3600,  # 1時間でタイムアウト
            )

            duration = time.time() - start_time

            if result.returncode != 0:
                return TrialResult(
                    trial_id=0,  # 後で上書きされる
                    parameters=parameters,
                    metrics={},
                    objective_value=float("inf"),
                    duration_seconds=duration,
                    success=False,
                    error_message=f"訓練失敗: {result.stderr}",
                )

            # TensorBoardログから指標を読み取り
            # TODO: 実装を完成させる
            # ここでは仮の実装
            metrics = {
                "critic_loss": 0.1,
                "actor_loss": -4.0,
                "ent_coef": 0.5,
            }  # 実際はログから読み取り

            objective_value = metrics.get(metric, float("inf"))
            if not lower_is_better:
                objective_value = -objective_value

            return TrialResult(
                trial_id=0,
                parameters=parameters,
                metrics=metrics,
                objective_value=objective_value,
                duration_seconds=duration,
                success=True,
            )

        except subprocess.TimeoutExpired:
            return TrialResult(
                trial_id=0,
                parameters=parameters,
                metrics={},
                objective_value=float("inf"),
                duration_seconds=3600,
                success=False,
                error_message="訓練タイムアウト",
            )

        except Exception as e:
            return TrialResult(
                trial_id=0,
                parameters=parameters,
                metrics={},
                objective_value=float("inf"),
                duration_seconds=0,
                success=False,
                error_message=str(e),
            )

        finally:
            # 一時ファイルを削除
            Path(temp_config_path).unlink(missing_ok=True)

    return objective_function


def create_mock_objective_function(
    noise_level: float = 0.1,
) -> Callable[[dict[str, ConfigDict]], TrialResult]:
    """
    Create mock objective function for testing.

    Args:
        noise_level: Noise level (0-1)

    Returns:
        Callable: Mock objective function
    """
    import random
    import time

    def mock_objective(parameters: ConfigDict) -> TrialResult:
        """
        Mock objective function for testing.

        Simulates a simple function where learning_rate close to 3e-4 is better.
        """
        # Slight delay simulation
        time.sleep(0.1)
        # わずかな遅延を模擬
        time.sleep(0.1)

        # パラメータから目的関数値を計算（仮想）
        lr = parameters.get("learning_rate", 3e-4)
        batch_size = parameters.get("batch_size", 128)
        gamma = parameters.get("gamma", 0.99)

        # 仮想的な最適値からの距離を計算
        optimal_lr = 3e-4
        optimal_batch = 128
        optimal_gamma = 0.99

        objective = (
            abs(lr - optimal_lr) / optimal_lr * 100
            + abs(batch_size - optimal_batch) / optimal_batch * 10
            + abs(gamma - optimal_gamma) / optimal_gamma * 50
        )

        # ノイズを追加
        objective += random.uniform(-noise_level, noise_level)

        # 仮想的なメトリクス
        metrics = {
            "critic_loss": objective * 0.01,
            "actor_loss": -objective * 0.1,
            "ent_coef": 0.5 + random.uniform(-0.2, 0.2),
        }

        return TrialResult(
            trial_id=0,
            parameters=parameters,
            metrics=metrics,
            objective_value=objective,
            duration_seconds=0.1,
            success=True,
        )

    return mock_objective
