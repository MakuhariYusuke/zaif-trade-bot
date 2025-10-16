"""モデル圧縮モジュール

知識蒸留、プルーニング、モデル圧縮手法を提供。
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class KnowledgeDistillation:
    """知識蒸留クラス

    大規模な教師モデルから小型の生徒モデルへ知識を転移。
    """

    def __init__(self,
                 temperature: float = 2.0,
                 alpha: float = 0.5,
                 distillation_loss: str = "kl"):
        """
        Args:
            temperature: 蒸留温度
            alpha: 蒸留損失とハード損失のバランス係数
            distillation_loss: 蒸留損失関数 ('kl' or 'mse')
        """
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_loss = distillation_loss

        if distillation_loss == "kl":
            self.distill_criterion = nn.KLDivLoss(reduction='batchmean')
        elif distillation_loss == "mse":
            self.distill_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported distillation loss: {distillation_loss}")

    def compute_distillation_loss(self,
                                  student_logits: torch.Tensor,
                                  teacher_logits: torch.Tensor,
                                  hard_targets: Optional[torch.Tensor] = None,
                                  labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """蒸留損失の計算

        Args:
            student_logits: 生徒モデルの出力
            teacher_logits: 教師モデルの出力
            hard_targets: ハードターゲット（オプション）
            labels: 正解ラベル（オプション）

        Returns:
            蒸留損失
        """
        # 教師モデルのソフトターゲット
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=-1)

        # 生徒モデルのソフト出力
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=-1)

        # 蒸留損失
        distill_loss = self.distill_criterion(student_soft, teacher_soft) * (self.temperature ** 2)

        total_loss = distill_loss

        # ハードターゲットがある場合は追加
        if hard_targets is not None and labels is not None:
            hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return total_loss

class ModelPruning:
    """モデルプルーニングクラス

    重要度の低い重みを削除してモデルを圧縮。
    """

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        """
        Args:
            model: プルーニング対象のモデル
            pruning_ratio: プルーニング率 (0.0-1.0)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.masks = {}

    def apply_l1_unstructured(self, name: str, param: torch.Tensor):
        """L1ノルムベースの非構造化プルーニング"""
        threshold = torch.quantile(torch.abs(param), self.pruning_ratio)
        mask = torch.abs(param) > threshold
        param.data *= mask.float()
        self.masks[name] = mask

    def apply_l2_structured(self, name: str, param: torch.Tensor):
        """L2ノルムベースの構造化プルーニング（チャネル単位）"""
        if len(param.shape) == 4:  # Conv2d weights
            # 出力チャネルごとのL2ノルムを計算
            l2_norms = torch.norm(param.view(param.shape[0], -1), p=2, dim=1)
            threshold = torch.quantile(l2_norms, self.pruning_ratio)
            mask = l2_norms > threshold
            # マスクを適用
            param.data = param.data * mask.view(-1, 1, 1, 1).float()
            self.masks[name] = mask

    def prune_model(self):
        """モデル全体にプルーニングを適用"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.apply_l1_unstructured(f"{name}.weight", module.weight)
                if module.bias is not None:
                    self.apply_l1_unstructured(f"{name}.bias", module.bias)
            elif isinstance(module, nn.Conv2d):
                self.apply_l2_structured(f"{name}.weight", module.weight)

    def get_sparsity(self) -> float:
        """モデルのスパース性を計算"""
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

class ModelCompression:
    """モデル圧縮統合クラス

    蒸留、プルーニング、量子化を統合した圧縮パイプライン。
    """

    def __init__(self,
                 teacher_model: Optional[nn.Module] = None,
                 student_model: Optional[nn.Module] = None):
        """
        Args:
            teacher_model: 教師モデル（蒸留用）
            student_model: 生徒モデル（圧縮対象）
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation = KnowledgeDistillation() if teacher_model is not None else None
        self.pruning = ModelPruning(student_model) if student_model is not None else None

    def compress_model(self,
                      compression_config: Dict[str, Any]) -> nn.Module:
        """モデル圧縮パイプライン

        Args:
            compression_config: 圧縮設定
                {
                    'method': 'distillation' | 'pruning' | 'combined',
                    'pruning_ratio': 0.3,
                    'temperature': 2.0,
                    'quantization': 'dynamic' | 'static' | None
                }

        Returns:
            圧縮されたモデル
        """
        method = compression_config.get('method', 'pruning')
        model = self.student_model or self.teacher_model

        if model is None:
            raise ValueError("モデルが指定されていません")

        logger.info(f"モデル圧縮を開始: method={method}")

        # プルーニング
        if method in ['pruning', 'combined']:
            pruning_ratio = compression_config.get('pruning_ratio', 0.3)
            self.pruning = ModelPruning(model, pruning_ratio)
            self.pruning.prune_model()

            sparsity = self.pruning.get_sparsity()
            logger.info(f"プルーニング完了: sparsity={sparsity:.3f}")

        # 蒸留（教師モデルがある場合）
        if method in ['distillation', 'combined'] and self.teacher_model is not None:
            logger.info("知識蒸留を開始")
            # 蒸留トレーニングは別途実行する必要がある
            logger.info("蒸留トレーニングが必要: distillation.train() を実行してください")

        # 量子化
        quantization = compression_config.get('quantization')
        if quantization:
            from .quantization import DynamicQuantization
            quantizer = DynamicQuantization()
            model = quantizer.quantize_model(model)
            logger.info(f"量子化完了: method={quantization}")

        return model

    def get_compression_stats(self) -> Dict[str, Any]:
        """圧縮統計を取得"""
        stats = {}

        if self.pruning:
            stats['sparsity'] = self.pruning.get_sparsity()

        # パラメータ数計算
        if self.student_model:
            total_params = sum(p.numel() for p in self.student_model.parameters())
            stats['total_parameters'] = total_params

        return stats