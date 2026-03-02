"""融合モジュール

モダリティ間および時間的特徴量融合を提供。
"""

__version__ = "1.0.0"

import torch  # type: ignore
import torch.nn as nn  # type: ignore

class TemporalIntegrationLayer(nn.Module):
    """時間的統合レイヤー

    BiLSTMとTransformerを組み合わせたハイブリッドアーキテクチャ。
    短期依存性（BiLSTM）と長期依存性（Transformer）を同時に学習。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # BiLSTM for temporal dependencies
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Transformer for long-range dependencies
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

        # 出力投影層
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        時間的統合の順伝播

        Args:
            x: 入力特徴量 (batch_size, seq_len, hidden_dim)
            attention_mask: アテンションマスク (batch_size, seq_len)

        Returns:
            時間的に統合された特徴量 (batch_size, seq_len, hidden_dim)
        """
        # BiLSTM処理
        lstm_out, _ = self.bilstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim)

        # Transformer処理
        # アテンションマスクの変換（Transformer用）
        if attention_mask is not None:
            # PyTorch Transformerはkey_padding_maskを使用（Trueがマスク）
            transformer_mask = attention_mask.bool()
        else:
            transformer_mask = None

        transformer_out = self.transformer(
            lstm_out, src_key_padding_mask=transformer_mask
        )
        # transformer_out: (batch_size, seq_len, hidden_dim)

        # 最終投影と正規化
        output = self.output_projection(transformer_out)
        output = self.layer_norm(output)

        return output

class ModalityFusion(nn.Module):
    """モダリティ融合レイヤー

    複数モダリティの特徴量を統合し、統一された表現を生成。
    """

    def __init__(
        self,
        num_modalities: int = 3,
        hidden_dim: int = 256,
        fusion_method: str = "attention",
    ):
        super().__init__()

        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method

        if fusion_method == "attention":
            # アテンションベース融合
            self.attention_fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_modalities),
                nn.Softmax(dim=-1),
            )
        elif fusion_method == "concat":
            # 単純結合
            self.concat_projection = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        elif fusion_method == "weighted_sum":
            # 重み付き和
            self.modality_weights = nn.Parameter(torch.ones(num_modalities))
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_features: torch.Tensor) -> torch.Tensor:
        """
        モダリティ融合の順伝播

        Args:
            modality_features: 各モダリティの特徴量
                             (batch_size, seq_len, hidden_dim * num_modalities)

        Returns:
            融合された特徴量 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = modality_features.shape

        if self.fusion_method == "attention":
            # アテンション重みの計算
            attention_weights = self.attention_fusion(modality_features)
            # (batch_size, seq_len, num_modalities)

            # 各モダリティの特徴量を分割
            split_features = torch.split(modality_features, self.hidden_dim, dim=-1)
            # split_features: [tensor(batch_size, seq_len, hidden_dim)] * num_modalities

            # 重み付き融合
            weighted_features = []
            for i, features in enumerate(split_features):
                weights = attention_weights[:, :, i].unsqueeze(-1)
                weighted = features * weights
                weighted_features.append(weighted)

            fused = torch.sum(torch.stack(weighted_features, dim=0), dim=0)

        elif self.fusion_method == "concat":
            # 単純結合
            fused = self.concat_projection(modality_features)

        elif self.fusion_method == "weighted_sum":
            # 重み付き和
            split_features = torch.split(modality_features, self.hidden_dim, dim=-1)
            weights = torch.softmax(self.modality_weights, dim=0)

            weighted_features = []
            for i, features in enumerate(split_features):
                weighted = features * weights[i]
                weighted_features.append(weighted)

            fused = torch.sum(torch.stack(weighted_features, dim=0), dim=0)
            fused = self.output_projection(fused)

        else:
            # デフォルト: concat
            fused = self.concat_projection(modality_features)

        # 最終正規化
        fused = self.layer_norm(fused)

        return fused

class MultiModalFeatureEncoder(nn.Module):
    """マルチモーダル特徴量エンコーダー

    各モダリティのエンコーダーと融合層を統合した完全な特徴量エンコーダー。
    """

    def __init__(
        self,
        price_dim: int = 156,
        text_dim: int = 768,
        economic_dim: int = 20,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 個別モダリティエンコーダー
        from ..encoders import EconomicEncoder, PriceEncoder, TextEncoder

        self.price_encoder = PriceEncoder(
            input_dim=price_dim,
            hidden_dims=[128, hidden_dim // 2],
            output_dim=hidden_dim,
        )

        self.text_encoder = TextEncoder(
            model_name="bert-base-uncased", output_dim=hidden_dim, fine_tune=True
        )

        self.economic_encoder = EconomicEncoder(
            input_dim=economic_dim,
            hidden_dims=[64, hidden_dim // 2],
            output_dim=hidden_dim,
        )

        # クロスモーダル・アテンション
        from ..attention import CrossModalAttention

        self.cross_attention = CrossModalAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
        )

        # 時間的統合
        self.temporal_integration = TemporalIntegrationLayer(
            hidden_dim=hidden_dim, num_layers=2, num_heads=num_heads, dropout=dropout
        )

        # 最終出力投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        price_data: torch.Tensor,
        text_data: torch.Tensor,
        economic_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        マルチモーダル特徴量エンコーディング

        Args:
            price_data: 価格データ (batch_size, seq_len, price_dim) または (batch_size, price_dim)
            text_data: テキストデータ (input_ids) (batch_size, seq_len)
            economic_data: 経済指標データ (batch_size, economic_dim)
            attention_mask: アテンションマスク (batch_size, seq_len)

        Returns:
            エンコードされた特徴量 (batch_size, hidden_dim)
        """
        # 各モダリティのエンコーディング
        price_features = self.price_encoder(price_data)  # (batch_size, hidden_dim)

        # テキスト特徴量のシーケンス化（簡易実装）
        batch_size = price_features.size(0)
        seq_len = text_data.size(1) if text_data.dim() > 1 else 1
        text_features = self.text_encoder(
            text_data, attention_mask
        )  # (batch_size, hidden_dim)

        economic_features = self.economic_encoder(
            economic_data
        )  # (batch_size, hidden_dim)

        # 特徴量のシーケンス化（時間軸の追加）
        # 実際の実装では、各タイムステップの特徴量が必要
        price_seq = price_features.unsqueeze(1).expand(-1, seq_len, -1)
        text_seq = text_features.unsqueeze(1).expand(-1, seq_len, -1)
        economic_seq = economic_features.unsqueeze(1).expand(-1, seq_len, -1)

        # クロスモーダル・アテンション
        attended_features = self.cross_attention(
            price_seq, text_seq, economic_seq, attention_mask
        )  # (batch_size, seq_len, hidden_dim)

        # 時間的統合
        temporal_features = self.temporal_integration(
            attended_features, attention_mask
        )  # (batch_size, seq_len, hidden_dim)

        # 最終出力（最後のタイムステップを使用）
        final_features = temporal_features[:, -1, :]  # (batch_size, hidden_dim)
        final_features = self.output_projection(final_features)

        return final_features
