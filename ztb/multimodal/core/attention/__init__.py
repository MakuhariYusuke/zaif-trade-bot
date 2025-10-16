"""注意機構モジュール

クロスモーダル・アテンションと注意ベースの特徴量統合を提供。
"""

__version__ = "1.0.0"

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Optional, Tuple

class CrossModalAttention(nn.Module):
    """クロスモーダル・アテンション機構

    複数モダリティ間の相互作用を学習し、統合された特徴量を生成する。
    マルチヘッド・アテンションに基づくTransformerアーキテクチャを使用。
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_layers: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # マルチヘッド・アテンション層
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # 層正規化とFFN
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])

        # Feed Forward Network
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])

        # モダリティ間フュージョン用の投影層
        self.modality_projection = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self,
                price_features: torch.Tensor,
                text_features: torch.Tensor,
                economic_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        クロスモーダル・アテンションの順伝播

        Args:
            price_features: 価格特徴量 (batch_size, seq_len, hidden_dim)
            text_features: テキスト特徴量 (batch_size, seq_len, hidden_dim)
            economic_features: 経済指標特徴量 (batch_size, seq_len, hidden_dim)
            attention_mask: アテンションマスク (batch_size, seq_len)

        Returns:
            統合された特徴量 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = price_features.shape

        # モダリティ特徴量を統合
        combined_features = torch.cat([
            price_features,      # 価格情報
            text_features,       # テキスト情報
            economic_features    # 経済情報
        ], dim=-1)  # (batch_size, seq_len, hidden_dim * 3)

        # モダリティ間フュージョン
        fused_features = self.modality_projection(combined_features)
        # (batch_size, seq_len, hidden_dim)

        # マルチヘッド・アテンション層の適用
        for i, (attn, norm1, norm2, ffn) in enumerate(zip(
            self.attention_layers,
            self.norm_layers[::2],
            self.norm_layers[1::2],
            self.ffn_layers
        )):
            # Self-attention with residual connection
            attn_output, _ = attn(
                fused_features, fused_features, fused_features,
                key_padding_mask=attention_mask
            )

            # Add & Norm
            fused_features = norm1(fused_features + attn_output)

            # Feed Forward Network
            ffn_output = ffn(fused_features)
            fused_features = norm2(fused_features + ffn_output)

        return fused_features

class MultiHeadCrossAttention(nn.Module):
    """マルチヘッド・クロス・アテンション

    異なるモダリティ間のクロス・アテンションを計算。
    Query-Key-Valueのクロスモーダル相互作用を学習。
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # クロス・アテンション層（モダリティ間）
        self.price_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.price_to_economic_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_economic_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # 出力投影層
        self.output_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                price_features: torch.Tensor,
                text_features: torch.Tensor,
                economic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチヘッド・クロス・アテンションの順伝播

        Args:
            price_features: 価格特徴量 (batch_size, seq_len, hidden_dim)
            text_features: テキスト特徴量 (batch_size, seq_len, hidden_dim)
            economic_features: 経済指標特徴量 (batch_size, seq_len, hidden_dim)

        Returns:
            統合特徴量とアテンション重み (batch_size, seq_len, hidden_dim), attention_weights
        """
        # クロスモーダル・アテンションの計算
        # 価格 → テキスト
        price_text_output, price_text_weights = self.price_to_text_attn(
            price_features, text_features, text_features
        )

        # 価格 → 経済
        price_economic_output, price_economic_weights = self.price_to_economic_attn(
            price_features, economic_features, economic_features
        )

        # テキスト → 経済
        text_economic_output, text_economic_weights = self.text_to_economic_attn(
            text_features, economic_features, economic_features
        )

        # 特徴量の統合
        combined_features = torch.cat([
            price_text_output,
            price_economic_output,
            text_economic_output
        ], dim=-1)  # (batch_size, seq_len, hidden_dim * 3)

        # 出力投影
        integrated_features = self.output_projection(combined_features)
        integrated_features = self.layer_norm(integrated_features)

        # アテンション重みの統合（可視化用）
        attention_weights = {
            'price_text': price_text_weights,
            'price_economic': price_economic_weights,
            'text_economic': text_economic_weights
        }

        return integrated_features, attention_weights

class AttentionFusion(nn.Module):
    """アテンションベースの特徴量フュージョン

    学習可能な重みで各モダリティの特徴量を融合。
    アテンション機構により動的に重みを調整。
    """

    def __init__(self, hidden_dim: int = 256, num_modalities: int = 3):
        super().__init__()

        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim

        # アテンション重み計算用の線形層
        self.attention_weights = nn.Linear(hidden_dim * num_modalities, num_modalities)
        self.softmax = nn.Softmax(dim=-1)

        # モダリティ別特徴量調整層
        self.modality_adapters = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_modalities)
        ])

    def forward(self, modality_features: torch.Tensor) -> torch.Tensor:
        """
        アテンションフュージョンの順伝播

        Args:
            modality_features: 各モダリティの特徴量 (batch_size, seq_len, hidden_dim * num_modalities)

        Returns:
            融合された特徴量 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = modality_features.shape

        # 各モダリティの特徴量を分割
        split_features = torch.split(modality_features, self.hidden_dim, dim=-1)
        # split_features: [tensor(batch_size, seq_len, hidden_dim)] * num_modalities

        # 各モダリティの適応
        adapted_features = []
        for i, features in enumerate(split_features):
            adapted = self.modality_adapters[i](features)
            adapted_features.append(adapted)

        # アテンション重みの計算
        concat_features = torch.cat(adapted_features, dim=-1)
        # (batch_size, seq_len, hidden_dim * num_modalities)

        attention_logits = self.attention_weights(concat_features)
        # (batch_size, seq_len, num_modalities)

        attention_weights = self.softmax(attention_logits)
        # (batch_size, seq_len, num_modalities)

        # 重み付き融合
        weighted_features = []
        for i, features in enumerate(adapted_features):
            weights = attention_weights[:, :, i].unsqueeze(-1)
            # (batch_size, seq_len, 1)
            weighted = features * weights
            weighted_features.append(weighted)

        fused_features = torch.sum(torch.stack(weighted_features, dim=0), dim=0)
        # (batch_size, seq_len, hidden_dim)

        return fused_features
