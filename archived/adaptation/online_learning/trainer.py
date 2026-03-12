"""
Online Learning SAC Trainer
リアルタイム適応機能を統合したSACトレーナー
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

import torch

from ztb.adaptation.online_learning.config import OnlineLearningConfig
from ztb.adaptation.online_learning.pipeline import OnlineLearningPipeline
from ztb.adaptation.online_learning.types import DataBatch, LearningState
from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OnlineLearningSACTrainer(SACAlgorithmTrainer):
    """
    オンライン学習機能を統合したSACトレーナー
    リアルタイムでデータを処理し、モデルを適応的に更新
    """

    def __init__(
        self,
        online_config: OnlineLearningConfig,
        sac_config: Dict[str, Any],
        env_config: Dict[str, Any],
    ):
        # SACAlgorithmTrainerの初期化をスキップ
        self.online_config = online_config
        self.sac_config = sac_config
        self.env_config = env_config
        self.logger = get_logger(__name__)

        # オンライン学習パイプライン
        # 仮のモデルを作成（TODO: 実際のSACモデルと統合）
        dummy_model = torch.nn.Linear(10, 1)  # 仮モデル
        self.online_pipeline = OnlineLearningPipeline(
            config=online_config, model=dummy_model
        )

        # ストリーミング制御
        self.is_online_learning_active = False
        self.online_thread: Optional[threading.Thread] = None
        self.data_stream: Optional[Iterator[DataBatch]] = None

        logger.info("Online Learning SAC Trainer initialized")

    def start_online_learning(self, data_stream: Iterator[DataBatch]) -> None:
        """
        オンライン学習を開始

        Args:
            data_stream: データストリームイテレータ
        """
        if self.is_online_learning_active:
            logger.warning("Online learning already active")
            return

        self.data_stream = data_stream
        self.is_online_learning_active = True

        # オンライン学習スレッドを開始
        self.online_thread = threading.Thread(
            target=self._online_learning_worker, daemon=True
        )
        self.online_thread.start()

        logger.info("Online learning started")

    def stop_online_learning(self) -> None:
        """オンライン学習を停止"""
        self.is_online_learning_active = False

        if self.online_thread:
            self.online_thread.join(timeout=10.0)

        if self.online_pipeline.is_streaming:
            self.online_pipeline.stop_streaming()

        logger.info("Online learning stopped")

    def _online_learning_worker(self) -> None:
        """オンライン学習ワーカー"""
        try:
            if self.data_stream:
                self.online_pipeline.start_streaming(self.data_stream)

            while self.is_online_learning_active:
                time.sleep(1.0)  # 定期チェック

                # 学習状態の監視
                learning_state = self.online_pipeline.learning_state

                # パフォーマンスチェック
                if self._should_trigger_adaptation(learning_state):
                    self._perform_adaptation()

        except Exception as e:
            logger.error(f"Online learning worker error: {e}")
            self.is_online_learning_active = False

    def _should_trigger_adaptation(self, learning_state: LearningState) -> bool:
        """適応トリガーの判定"""
        # 損失の急激な変化を検知
        if len(learning_state.loss_history) >= 10:
            recent_losses = learning_state.loss_history[-10:]
            avg_recent = sum(recent_losses) / len(recent_losses)
            avg_overall = sum(learning_state.loss_history) / len(
                learning_state.loss_history
            )

            # 最近の損失が全体平均の150%以上になった場合
            if avg_recent > avg_overall * 1.5:
                return True

        # メモリ使用量のチェック
        if (
            learning_state.memory_usage_mb
            > self.online_config.max_memory_usage_mb * 0.9
        ):
            return True

        return False

    def _perform_adaptation(self) -> None:
        """適応処理の実行"""
        logger.info("Performing adaptation...")

        try:
            # 学習率の調整
            current_lr = self.online_pipeline.learning_state.current_learning_rate
            new_lr = current_lr * 0.8  # 学習率を20%減少

            # オプティマイザの学習率更新
            for param_group in self.online_pipeline.optimizer.param_groups:
                param_group["lr"] = new_lr

            self.online_pipeline.learning_state.current_learning_rate = new_lr

            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Adaptation completed: learning rate adjusted to {new_lr}")

        except Exception as e:
            logger.error(f"Adaptation failed: {e}")

    def train_with_online_adaptation(
        self, total_timesteps: int, data_stream: Optional[Iterator[DataBatch]] = None
    ) -> Dict[str, Any]:
        """
        オンライン適応を伴う学習を実行

        Args:
            total_timesteps: 総学習ステップ数
            data_stream: オプションのデータストリーム

        Returns:
            学習結果
        """
        logger.info("Starting training with online adaptation...")

        # データストリームが提供された場合、オンライン学習を開始
        if data_stream:
            self.start_online_learning(data_stream)

        try:
            # 標準的なSAC学習を実行
            training_result = self.train(total_timesteps, use_online_adaptation=True)

            # オンライン学習のメトリクスを追加
            online_metrics = self._get_online_metrics()

            result = {
                **training_result,
                "online_metrics": online_metrics,
                "training_type": "online_adaptive_sac",
            }

            logger.info("Training with online adaptation completed")
            return result

        finally:
            # クリーンアップ
            self.stop_online_learning()

    def _get_online_metrics(self) -> Dict[str, Any]:
        """オンライン学習のメトリクスを取得"""
        learning_state = self.online_pipeline.learning_state

        return {
            "total_samples_processed": learning_state.total_samples_processed,
            "current_learning_rate": learning_state.current_learning_rate,
            "gradient_norm": learning_state.gradient_norm,
            "loss_history_length": len(learning_state.loss_history),
            "memory_usage_mb": learning_state.memory_usage_mb,
            "gpu_memory_usage_mb": learning_state.gpu_memory_usage_mb,
            "last_update_time": learning_state.last_update_time.isoformat(),
            "online_learning_active": self.is_online_learning_active,
        }

    def get_adaptation_status(self) -> Dict[str, Any]:
        """適応状態を取得"""
        return {
            "online_learning_active": self.is_online_learning_active,
            "learning_state": self.online_pipeline.learning_state.__dict__,
            "pipeline_status": {
                "is_streaming": self.online_pipeline.is_streaming,
                "streaming_buffer_size": len(self.online_pipeline.streaming_buffer),
                "memory_buffer_size": len(self.online_pipeline.memory_buffer),
            },
        }

    def save_online_model(self, path: str) -> None:
        """オンライン学習モデルを保存"""
        torch.save(
            {
                "model_state_dict": self.algorithm.model.state_dict(),
                "online_config": self.online_config,
                "learning_state": self.online_pipeline.learning_state,
                "timestamp": datetime.now().isoformat(),
            },
            path,
        )
        logger.info(f"Online learning model saved to {path}")

    def load_online_model(self, path: str) -> None:
        """オンライン学習モデルを読み込み"""
        checkpoint = torch.load(path)
        # self.algorithm.model.load_state_dict(checkpoint['model_state_dict'])  # TODO: algorithm統合
        self.online_pipeline.learning_state = checkpoint["learning_state"]
        logger.info(f"Online learning model loaded from {path}")
