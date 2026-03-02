"""
Training Utilities
トレーニング関連のユーティリティ関数
"""

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    dict,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from ztb.utils.types import TrainingResult, ValidationResult

@runtime_checkable
class SaveableModel(Protocol):
    def save(self, path: str) -> None:
        ...

@runtime_checkable
class LoadableClass(Protocol):
    def load(self, path: str) -> Any:
        ...

if TYPE_CHECKING:
    # For type checking only; avoid importing stable_baselines3 at module import time
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # type: ignore
    from stable_baselines3 import SAC, PPO  # type: ignore
    from sb3_contrib import MaskablePPO  # type: ignore

from ztb.utils.logging_utils import get_logger

from ztb.utils.file_utils import safe_json_dump

logger = get_logger(__name__)

def create_checkpoint_callback(
    save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 1
) -> "CheckpointCallback":
    """
    CheckpointCallbackを作成

    Args:
        save_freq: 保存頻度（ステップ数）
        save_path: 保存パス
        name_prefix: ファイル名のプレフィックス
        verbose: 詳細度

    Returns:
        CheckpointCallbackインスタンス
    """
    os.makedirs(save_path, exist_ok=True)

    # Import SB3 checkpoint callback lazily to avoid loading torch at module import time
    from stable_baselines3.common.callbacks import CheckpointCallback

    return CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        verbose=verbose,
    )

def create_eval_callback(
    eval_env,
    eval_freq: int,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
    verbose: int = 1,
    best_model_save_path: str | None = None,
    log_path: str | None = None,
) -> "EvalCallback":
    """
    EvalCallbackを作成

    Args:
        eval_env: 評価環境
        eval_freq: 評価頻度
        n_eval_episodes: 評価エピソード数
        deterministic: 決定論的行動
        verbose: 詳細度
        best_model_save_path: 最良モデルの保存パス
        log_path: ログパス

    Returns:
        EvalCallbackインスタンス
    """
    if best_model_save_path:
        os.makedirs(best_model_save_path, exist_ok=True)

    if log_path:
        os.makedirs(log_path, exist_ok=True)

    # Import EvalCallback lazily
    from stable_baselines3.common.callbacks import EvalCallback

    return EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        verbose=verbose,
    )

def save_model(model: SaveableModel, model_path: str, verbose: bool = True) -> bool:
    """
    モデルを保存

    Args:
        model: 保存するモデル
        model_path: 保存パス
        verbose: 詳細出力

    Returns:
        保存成功かどうか
    """
    try:
        # ディレクトリ作成
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # モデル保存
        model.save(model_path)

        if verbose:
            logger.info(f"Model saved to: {model_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        return False

def save_model_with_metadata(
    model: SaveableModel,
    model_path: str,
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
) -> bool:
    """
    モデルをメタデータ付きで保存

    Args:
        model: 保存するモデル
        model_path: 保存パス
        metadata: メタデータ辞書
        verbose: 詳細出力

    Returns:
        保存成功かどうか
    """
    success = save_model(model, model_path, verbose)
    if success and metadata:
        try:
            metadata_path = f"{model_path}.metadata.json"
            safe_json_dump(metadata, metadata_path, indent=2, default=str)
            if verbose:
                logger.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
    return success

def load_model(
    model_path: str, algorithm: str | None = None, verbose: bool = True
) -> "SAC" | "PPO" | "MaskablePPO" | None:
    """
    モデルを読み込み（統一版）

    Args:
        model_path: モデルパス
        algorithm: アルゴリズム指定（"sac", "ppo", "ppo_maskable"）またはNoneで自動検出
        verbose: 詳細出力

    Returns:
        読み込んだモデル、失敗時はNone
    """
    try:
        if not os.path.exists(model_path + ".zip"):
            logger.error(f"Model file not found: {model_path}.zip")
            return None

        if algorithm is None:
            # Auto-detect algorithm
            try:
                from sb3_contrib import MaskablePPO

                model = MaskablePPO.load(model_path)
                detected_alg = "ppo_maskable"
            except Exception:
                try:
                    from stable_baselines3 import PPO

                    model = PPO.load(model_path)
                    detected_alg = "ppo"
                except Exception:
                    from stable_baselines3 import SAC

                    model = SAC.load(model_path)
                    detected_alg = "sac"
            if verbose:
                logger.info(f"Auto-detected algorithm: {detected_alg}")
        else:
            # Use specified algorithm
            if algorithm == "sac":
                from stable_baselines3 import SAC

                model = SAC.load(model_path)
            elif algorithm == "ppo":
                from stable_baselines3 import PPO

                model = PPO.load(model_path)
            elif algorithm == "ppo_maskable":
                from sb3_contrib import MaskablePPO

                model = MaskablePPO.load(model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        if verbose:
            logger.info(f"Model loaded from: {model_path}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

def save_training_results(
    results: TrainingResult, output_path: str, verbose: bool = True
) -> bool:
    """
    トレーニング結果を保存

    Args:
        results: 保存する結果
        output_path: 保存パス
        verbose: 詳細出力

    Returns:
        保存成功かどうか
    """
    try:
        # ディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # JSON保存
        safe_json_dump(results, output_path, indent=2, ensure_ascii=False, default=str)

        if verbose:
            logger.info(f"Training results saved to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save training results to {output_path}: {e}")
        return False

def validate_training_config(config: dict[str, Any]) -> ValidationResult:
    """
    トレーニング設定をバリデーション

    Args:
        config: 設定辞書

    Returns:
        バリデーション結果 {"valid": bool, "errors": list[str], "warnings": list[str]}
    """
    errors = []
    warnings = []

    # 必須項目チェック
    required_keys = ["model", "training", "environment"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config section: {key}")

    # trainingセクション
    if "training" in config:
        training = config["training"]

        # total_timesteps
        if "total_timesteps" not in training:
            errors.append("training.total_timesteps is required")
        elif (
            not isinstance(training["total_timesteps"], int)
            or training["total_timesteps"] <= 0
        ):
            errors.append("training.total_timesteps must be a positive integer")

        # learning_rate
        if "learning_rate" in training:
            lr = training["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append("training.learning_rate must be a positive number")

        # batch_size
        if "batch_size" in training:
            bs = training["batch_size"]
            if not isinstance(bs, int) or bs <= 0:
                errors.append("training.batch_size must be a positive integer")

    # environmentセクション
    if "environment" in config:
        env = config["environment"]

        # 警告: 推奨設定
        if "transaction_cost" in env and env["transaction_cost"] > 0.01:
            warnings.append("High transaction cost detected (>1%), consider reducing")

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings}

def get_metric_from_logger(model, metric_name: str) -> float | None:
    """
    Get metric value from model logger.

    Args:
        model: The model with logger
        metric_name: Name of the metric to retrieve

    Returns:
        Metric value or None if not found
    """
    if not hasattr(model, "logger") or model.logger is None:
        return None

    try:
        name_to_value = model.logger.name_to_value

        # Check possible metric name variations
        possible_names = [
            metric_name,
            f"train/{metric_name}",
            f"rollout/{metric_name}",
        ]

        for name in possible_names:
            if name in name_to_value:
                return float(name_to_value[name])

        return None
    except (AttributeError, KeyError, TypeError):
        return None

class _DummyLoss:
    """Dummy loss function for fallback when torch is not available."""

    def __call__(self, *args, **kwargs):
        return 0

def get_safe_loss_function(loss_class, *args, **kwargs):
    """
    Safely get a loss function with fallback.

    Args:
        loss_class: The loss class to instantiate
        *args, **kwargs: Arguments for the loss class

    Returns:
        Loss function instance or dummy
    """
    try:
        return loss_class(*args, **kwargs)
    except Exception:
        return _DummyLoss()

def display_training_complete(
    final_metrics: dict[str, Any], training_time: float
) -> None:
    """
    Display training completion message.

    Args:
        final_metrics: Final training metrics
        training_time: Total training time in seconds
    """
    success = bool(final_metrics)  # Assume success if metrics provided
    if success:
        print("\n✅ Training completed successfully!")
        print(f"⏱️  Total training time: {training_time:.1f}s")
        if final_metrics:
            print("📊 Final Statistics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("\n❌ Training failed")
