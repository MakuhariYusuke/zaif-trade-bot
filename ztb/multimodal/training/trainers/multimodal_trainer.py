"""
Multimodal Learning Trainer for SAC v421
マルチモーダル学習専用のトレーナー
"""

from datetime import datetime
from typing import Any

import numpy as np
import torch

from ztb.multimodal.config import MultimodalConfig
from ztb.multimodal.models.architectures.multimodal_architecture import (
    MultiModalTradingAgent,
)
from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class MultimodalSACTrainer(SACAlgorithmTrainer):
    """
    マルチモーダルSACトレーナー
    SACTrainerを拡張してマルチモーダル学習をサポート
    """

    def __init__(
        self,
        multimodal_config: MultimodalConfig,
        sac_config: dict[str, Any],
        env_config: dict[str, Any],
    ):
        # SACAlgorithmTrainerの初期化をスキップして直接初期化
        self.multimodal_config = multimodal_config
        self.sac_config = sac_config
        self.env_config = env_config
        self.logger = get_logger(__name__)

        # マルチモーダルモデル
        self.multimodal_agent = MultiModalTradingAgent(
            price_feature_dim=getattr(
                multimodal_config.model, "price_feature_dim", 156
            ),
            text_embedding_dim=getattr(
                multimodal_config.features, "embedding_dim", 768
            ),
            economic_feature_dim=getattr(
                multimodal_config.model, "economic_feature_dim", 10
            ),
            action_dim=getattr(multimodal_config.model, "action_dim", 3),
            hidden_dim=getattr(multimodal_config.model, "attention_dim", 256),
            num_heads=getattr(multimodal_config.model, "num_heads", 8),
        )

        # データローダー（仮実装）
        self.data_loader = None  # TODO: Implement MultimodalDataLoader

        # SACアルゴリズムをマルチモーダルモデルで置き換え（仮実装）
        # self.algorithm.model = self.multimodal_agent
        # self.algorithm.actor = self.multimodal_agent.actor
        # self.algorithm.critic1 = self.multimodal_agent.critic1
        # self.algorithm.critic2 = self.multimodal_agent.critic2

        logger.info("Multimodal SAC Trainer initialized")

    def train_multimodal(
        self,
        total_timesteps: int,
        price_data: torch.Tensor,
        text_data: torch.Tensor,
        economic_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        マルチモーダル学習を実行

        Args:
            total_timesteps: 総学習ステップ数
            price_data: 価格データ [batch, seq_len, price_dim]
            text_data: テキストデータ [batch, seq_len, text_dim]
            economic_data: 経済指標データ [batch, seq_len, economic_dim]
            attention_mask: アテンションマスク

        Returns:
            学習結果
        """
        logger.info("Starting multimodal training...")

        # マルチモーダル特徴量のエンコード
        encoded_features = self.multimodal_agent.encode_features(
            price_data, text_data, economic_data, attention_mask
        )

        # 標準的なSAC学習を実行
        # 注意: 環境からのデータではなく、エンコードされた特徴量を使用
        training_result = self.train(total_timesteps, use_multimodal=True)

        # マルチモーダル特有のメトリクスを追加
        multimodal_metrics = self._compute_multimodal_metrics(
            price_data, text_data, economic_data, encoded_features
        )

        result = {
            **training_result,
            "multimodal_metrics": multimodal_metrics,
            "training_type": "multimodal_sac",
        }

        logger.info("Multimodal training completed")
        return result

    def _compute_multimodal_metrics(
        self,
        price_data: torch.Tensor,
        text_data: torch.Tensor,
        economic_data: torch.Tensor,
        encoded_features: torch.Tensor,
    ) -> dict[str, float]:
        """マルチモーダル特有のメトリクスを計算"""
        metrics = {}

        # モダリティの貢献度分析
        with torch.no_grad():
            # 各モダリティの重要度を計算
            price_importance = self._compute_modality_importance(price_data, "price")
            text_importance = self._compute_modality_importance(text_data, "text")
            economic_importance = self._compute_modality_importance(
                economic_data, "economic"
            )

            metrics["price_importance"] = price_importance
            metrics["text_importance"] = text_importance
            metrics["economic_importance"] = economic_importance

            # クロスモーダル相関
            metrics["cross_modal_correlation"] = self._compute_cross_modal_correlation(
                price_data, text_data, economic_data
            )

            # 特徴量エンコーディングの品質
            metrics["encoding_quality"] = self._compute_encoding_quality(
                encoded_features
            )

        return metrics

    def _compute_modality_importance(
        self, modality_data: torch.Tensor, modality_name: str
    ) -> float:
        """モダリティの重要度を計算"""
        # 簡易的な重要度計算（実際にはより洗練された方法を使用）
        if modality_data.numel() == 0:
            return 0.0

        # データの分散を重要度のプロキシとして使用
        importance = modality_data.var().item()
        return importance

    def _compute_cross_modal_correlation(
        self,
        price_data: torch.Tensor,
        text_data: torch.Tensor,
        economic_data: torch.Tensor,
    ) -> float:
        """クロスモーダル相関を計算"""
        # 簡易的な相関計算
        correlations = []

        # 価格 vs テキスト
        if price_data.numel() > 0 and text_data.numel() > 0:
            price_flat = price_data.view(-1).cpu().numpy()
            text_flat = text_data.view(-1).cpu().numpy()
            if len(price_flat) == len(text_flat):
                corr = np.corrcoef(price_flat, text_flat)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

        # 価格 vs 経済
        if price_data.numel() > 0 and economic_data.numel() > 0:
            price_flat = price_data.view(-1).cpu().numpy()
            econ_flat = economic_data.view(-1).cpu().numpy()
            if len(price_flat) == len(econ_flat):
                corr = np.corrcoef(price_flat, econ_flat)[0, 1]
                correlations.append(abs(corr) if not np.isnan(corr) else 0.0)

        return np.mean(correlations) if correlations else 0.0

    def _compute_encoding_quality(self, encoded_features: torch.Tensor) -> float:
        """エンコーディング品質を計算"""
        # エンコーディングの多様性を品質指標として使用
        if encoded_features.numel() == 0:
            return 0.0

        # 特徴量の標準偏差を計算
        std_per_feature = encoded_features.std(dim=0)
        avg_std = std_per_feature.mean().item()

        # 0-1の範囲に正規化
        quality = min(avg_std / 10.0, 1.0)  # 経験的なスケーリング
        return quality

    def evaluate_multimodal(
        self,
        price_data: torch.Tensor,
        text_data: torch.Tensor,
        economic_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        マルチモーダルモデルの評価

        Args:
            price_data: 価格データ
            text_data: テキストデータ
            economic_data: 経済指標データ
            attention_mask: アテンションマスク

        Returns:
            評価結果
        """
        logger.info("Evaluating multimodal model...")

        self.multimodal_agent.eval()

        with torch.no_grad():
            # 特徴量エンコーディング
            encoded_features = self.multimodal_agent.encode_features(
                price_data, text_data, economic_data, attention_mask
            )

            # 行動選択
            actions, log_probs = self.multimodal_agent.get_action(
                encoded_features, deterministic=True
            )

            # Q値計算
            q1, q2 = self.multimodal_agent.get_value(encoded_features, actions)

            evaluation_result = {
                "encoded_features_shape": encoded_features.shape,
                "actions_shape": actions.shape,
                "q_values_mean": (q1.mean() + q2.mean()).item() / 2,
                "q_values_std": ((q1.std() + q2.std()) / 2).item(),
                "log_probs_mean": log_probs.mean().item(),
                "evaluation_timestamp": datetime.now().isoformat(),
            }

        logger.info("Multimodal evaluation completed")
        return evaluation_result

    def save_multimodal_model(self, path: str) -> None:
        """マルチモーダルモデルを保存"""
        torch.save(
            {
                "model_state_dict": self.multimodal_agent.state_dict(),
                "config": self.multimodal_config,
                "timestamp": datetime.now().isoformat(),
            },
            path,
        )
        logger.info(f"Multimodal model saved to {path}")

    def load_multimodal_model(self, path: str) -> None:
        """マルチモーダルモデルを読み込み"""
        checkpoint = torch.load(path)
        self.multimodal_agent.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Multimodal model loaded from {path}")
