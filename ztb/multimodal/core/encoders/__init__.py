"""エンコーダーモジュール

各種モダリティの特徴量エンコーディングを提供。
"""

__version__ = "1.0.0"

from typing import List, Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore


class BaseEncoder(nn.Module):
    """基本エンコーダークラス"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # エンコーダー層の構築
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_dim = hidden_dim

        # 出力層
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        return self.encoder(x)


class PriceEncoder(BaseEncoder):
    """価格データエンコーダー

    為替レートや価格データをエンコードする。
    """

    def __init__(
        self,
        input_dim: int = 156,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 64,
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64]

        super().__init__(input_dim, hidden_dims, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """価格データをエンコード

        Args:
            x: 価格特徴量テンソル (batch_size, seq_len, feature_dim) または (batch_size, feature_dim)

        Returns:
            エンコードされた特徴量 (batch_size, output_dim)
        """
        # シーケンスの平均を取って固定長に
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch_size, feature_dim)

        return super().forward(x)


class TextEncoder(nn.Module):
    """テキストデータエンコーダー

    BERTベースのテキストエンコーダー。
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 768,
        fine_tune: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.fine_tune = fine_tune

        # BERTモデルの初期化（実際の実装ではtransformersライブラリを使用）
        # ここではプレースホルダー
        self.bert = nn.Linear(768, 768)  # ダミーのBERTモデル
        self.projection = (
            nn.Linear(768, output_dim) if output_dim != 768 else nn.Identity()
        )

        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """テキストをエンコード

        Args:
            input_ids: トークン化されたテキスト (batch_size, seq_len)
            attention_mask: アテンションマスク (batch_size, seq_len)

        Returns:
            エンコードされた特徴量 (batch_size, output_dim)
        """
        # BERTの出力（CLSトークンを使用）
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # cls_output = outputs.last_hidden_state[:, 0, :]  # CLSトークン

        # プレースホルダー実装
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        bert_output = torch.randn(batch_size, seq_len, 768)  # ダミー出力
        cls_output = bert_output[:, 0, :]  # CLSトークン

        return self.projection(cls_output)


class EconomicEncoder(BaseEncoder):
    """経済指標エンコーダー

    経済指標データをエンコードする。
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 32,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 32]

        super().__init__(input_dim, hidden_dims, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """経済指標データをエンコード

        Args:
            x: 経済指標特徴量 (batch_size, feature_dim)

        Returns:
            エンコードされた特徴量 (batch_size, output_dim)
        """
        return super().forward(x)
