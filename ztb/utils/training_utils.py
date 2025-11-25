"""
Training Utilities
トレーニング関連のユーティリティ関数
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    # For type checking only; avoid importing stable_baselines3 at module import time
    from stable_baselines3 import SAC  # type: ignore
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # type: ignore

from ztb.utils.logging_utils import get_logger

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
    best_model_save_path: Optional[str] = None,
    log_path: Optional[str] = None,
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


def save_model(model: "SAC", model_path: str, verbose: bool = True) -> bool:
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


def load_model(model_path: str, verbose: bool = True) -> Optional["SAC"]:
    """
    モデルを読み込み

    Args:
        model_path: モデルパス
        verbose: 詳細出力

    Returns:
        読み込んだモデル、失敗時はNone
    """
    try:
        if not os.path.exists(model_path + ".zip"):
            logger.error(f"Model file not found: {model_path}.zip")
            return None

        from stable_baselines3 import SAC as _SAC

        model = _SAC.load(model_path)

        if verbose:
            logger.info(f"Model loaded from: {model_path}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def save_training_results(
    results: Dict[str, Any], output_path: str, verbose: bool = True
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
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        if verbose:
            logger.info(f"Training results saved to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save training results to {output_path}: {e}")
        return False


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    トレーニング設定をバリデーション

    Args:
        config: 設定辞書

    Returns:
        バリデーション結果 {"valid": bool, "errors": List[str], "warnings": List[str]}
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

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
