#!/usr/bin/env python3
"""
Training Logger Utility - 統一訓練ログユーティリティ

訓練中の標準化されたログ出力を提供します。
- 訓練開始/終了バナー
- 設定サマリー表示
- メトリクス整形
- 進捗情報の統一フォーマット
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingLogger:
    """訓練ログの統一インターフェース"""

    def __init__(self, algorithm: str, model_name: str, verbose: bool = True):
        """
        Args:
            algorithm: アルゴリズム名（'sac', 'ppo', など）
            model_name: モデル名
            verbose: 詳細出力を行うか
        """
        self.algorithm = algorithm.upper()
        self.model_name = model_name
        self.verbose = verbose
        self.start_time = None

    def print_banner(self, title: str, width: int = 80, char: str = "=") -> None:
        """バナーを出力"""
        print(char * width)
        print(f"{title:^{width}}")
        print(char * width)

    def print_section(self, title: str, width: int = 80, char: str = "-") -> None:
        """セクションタイトルを出力"""
        print(f"\n{char * width}")
        print(f"  {title}")
        print(char * width)

    def format_value(self, value: Any, precision: int = 6) -> str:
        """値を適切にフォーマット"""
        if isinstance(value, float):
            # 科学的記法が必要な大きな/小さな数値
            if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
                return f"{value:.{precision}e}"
            else:
                return f"{value:.{precision}f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, bool):
            return "✓" if value else "✗"
        elif value is None:
            return "N/A"
        else:
            return str(value)

    def print_training_start_banner(
        self,
        config: Dict[str, Any],
        total_timesteps: int,
        data_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        訓練開始時のバナーを表示

        Args:
            config: 訓練設定
            total_timesteps: 総訓練ステップ数
            data_info: データ情報（行数、期間など）
        """
        self.start_time = datetime.now()

        print("\n")
        self.print_banner(f"🚀 {self.algorithm} Training - {self.model_name}", char="=")

        # 基本情報
        print("\n📋 Training Configuration:")
        print(f"  Algorithm:      {self.algorithm}")
        print(f"  Model Name:     {self.model_name}")
        print(f"  Total Steps:    {total_timesteps:,}")
        print(f"  Start Time:     {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # データ情報
        if data_info:
            print("\n📊 Data Information:")
            for key, value in data_info.items():
                formatted_value = self.format_value(value)
                print(f"  {key:20s}: {formatted_value}")

        # アルゴリズム別設定
        if self.algorithm == "SAC":
            self._print_sac_config(config)
        elif self.algorithm == "PPO":
            self._print_ppo_config(config)

        # 環境設定
        self._print_environment_config(config)

        print("\n" + "=" * 80 + "\n")

    def _print_sac_config(self, config: Dict[str, Any]) -> None:
        """SAC設定を表示"""
        sac_params = config.get("sac_hyperparameters") or config.get("sac_params", {})

        if not sac_params:
            return

        self.print_section("SAC Hyperparameters")

        # 主要パラメータ
        key_params = [
            ("learning_rate", "Learning Rate"),
            ("batch_size", "Batch Size"),
            ("gamma", "Gamma (Discount)"),
            ("tau", "Tau (Soft Update)"),
            ("target_update_interval", "Target Update Interval"),
            ("learning_starts", "Learning Starts"),
            ("buffer_size", "Replay Buffer Size"),
            ("ent_coef", "Entropy Coef"),
            ("target_entropy", "Target Entropy"),
        ]

        for key, label in key_params:
            if key in sac_params:
                value = sac_params[key]
                formatted_value = self.format_value(value)
                print(f"  {label:25s}: {formatted_value}")

    def _print_ppo_config(self, config: Dict[str, Any]) -> None:
        """PPO設定を表示"""
        ppo_params = config.get("ppo_hyperparameters", {})

        if not ppo_params:
            return

        self.print_section("PPO Hyperparameters")

        # 主要パラメータ
        key_params = [
            ("learning_rate", "Learning Rate"),
            ("n_steps", "Steps per Update"),
            ("batch_size", "Batch Size"),
            ("n_epochs", "Epochs per Update"),
            ("gamma", "Gamma (Discount)"),
            ("gae_lambda", "GAE Lambda"),
            ("clip_range", "Clip Range"),
            ("ent_coef", "Entropy Coef"),
            ("vf_coef", "Value Function Coef"),
            ("max_grad_norm", "Max Grad Norm"),
        ]

        for key, label in key_params:
            if key in ppo_params:
                value = ppo_params[key]
                formatted_value = self.format_value(value)
                print(f"  {label:25s}: {formatted_value}")

    def _print_environment_config(self, config: Dict[str, Any]) -> None:
        """環境設定を表示"""
        env_config = config.get("environment", {})

        if not env_config:
            return

        self.print_section("Environment Configuration")

        # 主要パラメータ
        key_params = [
            ("initial_balance", "Initial Balance"),
            ("transaction_cost", "Transaction Cost"),
            ("max_position_size", "Max Position Size"),
            ("use_continuous_actions", "Continuous Actions"),
            ("enable_action_masking", "Action Masking"),
            ("use_standardized_observations", "Standardized Obs"),
        ]

        for key, label in key_params:
            if key in env_config:
                value = env_config[key]
                formatted_value = self.format_value(value)
                print(f"  {label:25s}: {formatted_value}")

        # 報酬設定
        reward_config = env_config.get("reward_settings", {})
        if reward_config:
            print("\n  Reward Settings:")
            for key, value in reward_config.items():
                formatted_value = self.format_value(value)
                print(f"    {key:30s}: {formatted_value}")

    def print_training_progress(
        self, current_step: int, total_steps: int, metrics: Dict[str, float]
    ) -> None:
        """
        訓練進捗を表示

        Args:
            current_step: 現在のステップ
            total_steps: 総ステップ数
            metrics: メトリクス辞書
        """
        progress_pct = (current_step / total_steps) * 100

        parts = [f"Step {current_step:,}/{total_steps:,} ({progress_pct:.1f}%)"]

        for key, value in metrics.items():
            formatted_value = self.format_value(value, precision=6)
            parts.append(f"{key}={formatted_value}")

        message = " | ".join(parts)
        print(message)

    def print_training_complete_banner(
        self, result: Dict[str, Any], final_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        訓練完了時のバナーを表示

        Args:
            result: 訓練結果
            final_metrics: 最終メトリクス
        """
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None

        print("\n")
        self.print_banner(f"✅ {self.algorithm} Training Completed", char="=")

        # 基本情報
        print("\n📊 Training Summary:")
        print(f"  End Time:       {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            hours = duration.total_seconds() / 3600
            print(f"  Duration:       {hours:.2f} hours ({duration})")

        # 結果情報
        if "model_path" in result:
            print(f"  Model Saved:    {result['model_path']}")
        if "log_path" in result:
            print(f"  Logs Saved:     {result['log_path']}")
        if "total_timesteps" in result:
            print(f"  Total Steps:    {result['total_timesteps']:,}")

        # 最終メトリクス
        if final_metrics:
            print("\n📈 Final Metrics:")
            for key, value in final_metrics.items():
                formatted_value = self.format_value(value, precision=6)
                print(f"  {key:20s}: {formatted_value}")

        print("\n" + "=" * 80 + "\n")

    def print_error_banner(self, error: Exception, context: str = "") -> None:
        """
        エラー時のバナーを表示

        Args:
            error: エラーオブジェクト
            context: エラーコンテキスト
        """
        print("\n")
        self.print_banner(f"❌ {self.algorithm} Training Failed", char="=")

        print("\n⚠️ Error Details:")
        print(f"  Type:           {type(error).__name__}")
        print(f"  Message:        {str(error)}")
        if context:
            print(f"  Context:        {context}")

        print("\n" + "=" * 80 + "\n")

    def save_metrics_json(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """
        メトリクスをJSON形式で保存（最適化スクリプト用）

        Args:
            metrics: メトリクス辞書
            output_path: 出力先パス
        """
        # 出力可能な形式に変換
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = str(value)

        # タイムスタンプ追加
        serializable_metrics["timestamp"] = datetime.now().isoformat()
        serializable_metrics["algorithm"] = self.algorithm
        serializable_metrics["model_name"] = self.model_name

        # ファイルに保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"📝 Metrics saved to: {output_path}")


def create_training_logger(
    algorithm: str, model_name: str, verbose: bool = True
) -> TrainingLogger:
    """
    TrainingLoggerインスタンスを作成

    Args:
        algorithm: アルゴリズム名
        model_name: モデル名
        verbose: 詳細出力

    Returns:
        TrainingLoggerインスタンス
    """
    return TrainingLogger(algorithm, model_name, verbose)
