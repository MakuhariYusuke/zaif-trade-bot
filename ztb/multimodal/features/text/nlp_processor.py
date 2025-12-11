"""
マルチモーダル学習 - 自然言語数値化手法設計
"""

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

try:
    # Transformers is an optional dependency used for heavy NLP features; tests
    # should not fail import if it's missing or incompatible with the torch
    # installed in CI. We import lazily and fallback to None to avoid import
    # errors during test collection.
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional runtime dep
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore

logger = logging.getLogger(__name__)


class TextSentimentEncoder(nn.Module):
    """
    自然言語を感情スコアに変換するエンコーダー
    複数の手法を組み合わせたハイブリッドアプローチ
    """

    def __init__(
        self,
        model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        embedding_dim: int = 768,
        output_dim: int = 3,
    ):  # [negative, neutral, positive]
        super().__init__()

        # BERTベースの感情分析モデル
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)

        # 感情スコア出力層
        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1),
        )

        # 感情強度エンコーダー（-1 to 1）
        self.intensity_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # -1 to 1
        )

    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        テキストを感情特徴量に変換

        Args:
            texts: 入力テキストのリスト

        Returns:
            感情特徴量辞書
            - sentiment_scores: [negative, neutral, positive] の確率分布
            - intensity: 感情強度 (-1: 非常にネガティブ, 1: 非常にポジティブ)
            - embeddings: BERT埋め込みベクトル
        """
        # トークナイズ
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # BERTエンコーディング
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]トークン

        # 感情分類
        sentiment_scores = self.sentiment_head(embeddings)

        # 感情強度
        intensity = self.intensity_encoder(embeddings)

        return {
            "sentiment_scores": sentiment_scores,
            "intensity": intensity,
            "embeddings": embeddings,
        }


class FinancialTextProcessor:
    """
    金融ニュース特化のテキスト処理クラス
    """

    def __init__(self):
        # 金融ドメイン特化の感情辞書
        self.financial_sentiment_dict = {
            # ポジティブワード
            "上昇": 0.8,
            "上昇トレンド": 0.9,
            "強気": 0.7,
            "買い": 0.6,
            "成長": 0.7,
            "回復": 0.6,
            "改善": 0.5,
            "好調": 0.6,
            "最高値": 0.8,
            "新高値": 0.9,
            "買い増し": 0.7,
            # ネガティブワード
            "下落": -0.8,
            "下落トレンド": -0.9,
            "弱気": -0.7,
            "売り": -0.6,
            "減少": -0.7,
            "悪化": -0.8,
            "不調": -0.6,
            "損失": -0.8,
            "最安値": -0.8,
            "新安値": -0.9,
            "売り越し": -0.7,
            # 中立的・文脈依存
            "安定": 0.1,
            "変動": 0.0,
            "調整": -0.2,
        }

    def extract_financial_sentiment(self, text: str) -> Dict[str, float]:
        """
        金融特化の感情分析

        Args:
            text: 分析対象テキスト

        Returns:
            感情スコア辞書
        """
        words = text.lower().split()
        sentiment_score = 0.0
        matched_words = []

        for word in words:
            if word in self.financial_sentiment_dict:
                sentiment_score += self.financial_sentiment_dict[word]
                matched_words.append(word)

        # 正規化（-1 to 1）
        normalized_score = np.tanh(sentiment_score / max(len(matched_words), 1))

        return {
            "sentiment_score": normalized_score,
            "matched_words": matched_words,
            "confidence": min(len(matched_words) / len(words), 1.0),
        }


class MultiModalFeatureIntegrator(nn.Module):
    """
    複数モダリティの特徴量統合クラス
    価格データ + テキスト感情 + 数値指標
    """

    def __init__(
        self,
        price_features_dim: int = 156,
        text_features_dim: int = 768,
        economic_features_dim: int = 20,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.price_features_dim = price_features_dim
        self.text_features_dim = text_features_dim
        self.economic_features_dim = economic_features_dim

        # 各モダリティのエンコーダー
        self.price_encoder = nn.Sequential(
            nn.Linear(price_features_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(text_features_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.economic_encoder = nn.Sequential(
            nn.Linear(economic_features_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # クロスモーダル・アテンション
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1
        )

        # 統合特徴量出力
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        price_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        economic_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        複数モダリティの特徴量統合

        Args:
            price_features: 価格ベース特徴量 [batch, seq_len, price_features_dim]
            text_embeddings: テキスト埋め込み [batch, seq_len, text_features_dim]
            economic_features: 経済指標特徴量 [batch, seq_len, economic_features_dim]

        Returns:
            統合された特徴量 [batch, seq_len, hidden_dim]
        """

        # 各モダリティのエンコーディング
        price_encoded = self.price_encoder(
            price_features
        )  # [batch, seq_len, hidden_dim]
        text_encoded = self.text_encoder(
            text_embeddings
        )  # [batch, seq_len, hidden_dim]
        econ_encoded = self.economic_encoder(
            economic_features
        )  # [batch, seq_len, hidden_dim]

        # クロスモーダル・アテンション（テキストをクエリとして価格と経済指標に注目）
        attn_output, _ = self.cross_attention(
            query=text_encoded.transpose(0, 1),  # [seq_len, batch, hidden_dim]
            key=torch.cat([price_encoded, econ_encoded], dim=-1).transpose(0, 1),
            value=torch.cat([price_encoded, econ_encoded], dim=-1).transpose(0, 1),
        )
        attn_output = attn_output.transpose(0, 1)  # [batch, seq_len, hidden_dim]

        # 特徴量融合
        combined = torch.cat([price_encoded, text_encoded, attn_output], dim=-1)
        integrated_features = self.fusion_layer(combined)

        return integrated_features


class SyntheticDataGenerator:
    """
    マルチモーダル学習用の合成データ生成器
    データが不足する場合の代替手段
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 市場状態パターン
        self.market_states = {
            "bull": {"sentiment": 0.7, "volatility": 0.3, "trend": 1},
            "bear": {"sentiment": -0.7, "volatility": 0.5, "trend": -1},
            "sideways": {"sentiment": 0.0, "volatility": 0.2, "trend": 0},
            "volatile": {"sentiment": -0.3, "volatility": 0.8, "trend": 0},
        }

    def generate_market_news(
        self, market_state: str, num_samples: int = 100
    ) -> List[Dict]:
        """
        市場状態に基づく合成ニュース生成

        Args:
            market_state: 市場状態 ('bull', 'bear', 'sideways', 'volatile')
            num_samples: 生成サンプル数

        Returns:
            合成ニュースデータのリスト
        """

        if market_state not in self.market_states:
            raise ValueError(f"Unknown market state: {market_state}")

        state_params = self.market_states[market_state]
        news_templates = {
            "bull": [
                "市場が上昇トレンドを継続、投資家心理改善",
                "経済指標が予想を上回り、強気相場が続く",
                "企業業績が好調で株価が最高値を更新",
            ],
            "bear": [
                "市場が下落トレンドにあり、投資家心理悪化",
                "経済指標が失望を誘い、弱気相場が継続",
                "企業業績の悪化で株価が最安値を記録",
            ],
            "sideways": [
                "市場が横ばい推移、投資家は様子見姿勢",
                "経済指標が安定し、方向感に欠ける相場",
                "企業業績が横ばいで株価に大きな変動なし",
            ],
            "volatile": [
                "市場が激しく変動、投資家心理不安定",
                "経済指標の変動が大きく、相場が乱高下",
                "地政学的リスクで株価が急落と急騰を繰り返す",
            ],
        }

        synthetic_data = []
        templates = news_templates[market_state]

        for i in range(num_samples):
            # テンプレート選択
            template = np.random.choice(templates)

            # 感情スコアにノイズを加える
            sentiment_noise = np.random.normal(0, 0.1)
            sentiment_score = state_params["sentiment"] + sentiment_noise

            # 信頼度
            confidence = np.random.uniform(0.7, 0.95)

            synthetic_data.append(
                {
                    "text": template,
                    "sentiment_score": np.clip(sentiment_score, -1, 1),
                    "confidence": confidence,
                    "market_state": market_state,
                    "timestamp": f"2025-10-{np.random.randint(1, 31):02d}",
                }
            )

        return synthetic_data

    def generate_economic_indicators(
        self, market_state: str, num_samples: int = 100
    ) -> np.ndarray:
        """
        市場状態に基づく経済指標生成

        Args:
            market_state: 市場状態
            num_samples: 生成サンプル数

        Returns:
            経済指標配列 [num_samples, num_indicators]
        """

        state_params = self.market_states[market_state]

        # 経済指標のベース値
        base_indicators = {
            "bull": [
                3.5,
                4.2,
                0.25,
                105.0,
                2.1,
            ],  # GDP, 失業率, 金利, 株価指数, インフレ
            "bear": [1.8, 6.1, 0.75, 85.0, 1.2],
            "sideways": [2.5, 5.0, 0.50, 95.0, 1.8],
            "volatile": [2.8, 5.5, 0.60, 90.0, 2.5],
        }

        indicators = []
        for i in range(num_samples):
            base = np.array(base_indicators[market_state])

            # ランダムノイズを加える
            noise = np.random.normal(0, 0.1, size=len(base))
            noisy_indicators = base * (1 + noise)

            # 市場状態に応じた追加変動
            volatility_factor = state_params["volatility"]
            trend_factor = state_params["trend"] * 0.1

            final_indicators = noisy_indicators * (
                1 + volatility_factor * np.random.normal(0, 0.2, size=len(base))
            )
            final_indicators = final_indicators * (1 + trend_factor)

            indicators.append(final_indicators)

        return np.array(indicators)


# 使用例とテスト
if __name__ == "__main__":
    # テキスト感情エンコーダーのテスト
    encoder = TextSentimentEncoder()

    test_texts = [
        "市場が大幅上昇、投資家心理が改善",
        "経済指標が悪化し株価が下落",
        "市場が安定して推移",
    ]

    results = encoder(test_texts)
    print("感情分析結果:")
    print(f"感情スコア: {results['sentiment_scores']}")
    print(f"感情強度: {results['intensity']}")

    # 合成データ生成のテスト
    generator = SyntheticDataGenerator()

    synthetic_news = generator.generate_market_news("bull", 5)
    print("\n合成ニュースデータ:")
    for news in synthetic_news:
        print(f"テキスト: {news['text']}")
        print(f"感情スコア: {news['sentiment_score']:.3f}")
        print(f"信頼度: {news['confidence']:.3f}")
        print("---")

    economic_data = generator.generate_economic_indicators("bull", 5)
    print(f"\n経済指標データ形状: {economic_data.shape}")
    print(f"サンプル指標: {economic_data[0]}")
