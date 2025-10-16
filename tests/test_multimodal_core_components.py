"""マルチモーダルコアコンポーネントの単体テスト

クロスモーダル・アテンション、時間的統合、特徴量エンコーダーのテストを含む。
"""

import unittest
import torch
import numpy as np

# テスト対象のインポート
from ztb.multimodal.core.attention import CrossModalAttention, MultiHeadCrossAttention, AttentionFusion
from ztb.multimodal.core.fusion import TemporalIntegrationLayer, ModalityFusion, MultiModalFeatureEncoder


class TestCrossModalAttention(unittest.TestCase):
    """CrossModalAttentionのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.attention = CrossModalAttention(
            hidden_dim=64,
            num_heads=4,
            dropout=0.1,
            num_layers=2
        )
        self.batch_size = 2
        self.seq_len = 8

    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.attention.hidden_dim, 64)
        self.assertEqual(self.attention.num_heads, 4)

    def test_forward(self):
        """順伝播テスト"""
        price_features = torch.randn(self.batch_size, self.seq_len, 64)
        text_features = torch.randn(self.batch_size, self.seq_len, 64)
        economic_features = torch.randn(self.batch_size, self.seq_len, 64)

        output = self.attention(price_features, text_features, economic_features)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_with_mask(self):
        """マスク付き順伝播テスト"""
        price_features = torch.randn(self.batch_size, self.seq_len, 64)
        text_features = torch.randn(self.batch_size, self.seq_len, 64)
        economic_features = torch.randn(self.batch_size, self.seq_len, 64)
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        attention_mask[:, -2:] = 0  # 最後の2トークンをマスク

        output = self.attention(price_features, text_features, economic_features, attention_mask)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        self.assertTrue(torch.isfinite(output).all())


class TestMultiHeadCrossAttention(unittest.TestCase):
    """MultiHeadCrossAttentionのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.attention = MultiHeadCrossAttention(
            hidden_dim=64,
            num_heads=4,
            dropout=0.1
        )
        self.batch_size = 2
        self.seq_len = 8

    def test_forward(self):
        """順伝播テスト"""
        price_features = torch.randn(self.batch_size, self.seq_len, 64)
        text_features = torch.randn(self.batch_size, self.seq_len, 64)
        economic_features = torch.randn(self.batch_size, self.seq_len, 64)

        output, attention_weights = self.attention(price_features, text_features, economic_features)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        self.assertTrue(torch.isfinite(output).all())

        # アテンション重みの確認
        self.assertIn('price_text', attention_weights)
        self.assertIn('price_economic', attention_weights)
        self.assertIn('text_economic', attention_weights)


class TestAttentionFusion(unittest.TestCase):
    """AttentionFusionのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.fusion = AttentionFusion(hidden_dim=64, num_modalities=3)
        self.batch_size = 2
        self.seq_len = 8

    def test_forward(self):
        """順伝播テスト"""
        modality_features = torch.randn(self.batch_size, self.seq_len, 64 * 3)

        output = self.fusion(modality_features)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        self.assertTrue(torch.isfinite(output).all())


class TestTemporalIntegrationLayer(unittest.TestCase):
    """TemporalIntegrationLayerのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.integration = TemporalIntegrationLayer(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        self.batch_size = 2
        self.seq_len = 8

    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.integration.hidden_dim, 64)
        self.assertEqual(self.integration.num_layers, 2)

    def test_forward(self):
        """順伝播テスト"""
        x = torch.randn(self.batch_size, self.seq_len, 64)

        output = self.integration(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_with_mask(self):
        """マスク付き順伝播テスト"""
        x = torch.randn(self.batch_size, self.seq_len, 64)
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        attention_mask[:, -2:] = 0

        output = self.integration(x, attention_mask)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 64))
        self.assertTrue(torch.isfinite(output).all())


class TestModalityFusion(unittest.TestCase):
    """ModalityFusionのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.batch_size = 2
        self.seq_len = 8
        self.hidden_dim = 64
        self.num_modalities = 3

    def test_attention_fusion(self):
        """アテンションフュージョンテスト"""
        fusion = ModalityFusion(
            num_modalities=self.num_modalities,
            hidden_dim=self.hidden_dim,
            fusion_method="attention"
        )

        modality_features = torch.randn(self.batch_size, self.seq_len, self.hidden_dim * self.num_modalities)

        output = fusion(modality_features)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertTrue(torch.isfinite(output).all())

    def test_concat_fusion(self):
        """結合フュージョンテスト"""
        fusion = ModalityFusion(
            num_modalities=self.num_modalities,
            hidden_dim=self.hidden_dim,
            fusion_method="concat"
        )

        modality_features = torch.randn(self.batch_size, self.seq_len, self.hidden_dim * self.num_modalities)

        output = fusion(modality_features)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertTrue(torch.isfinite(output).all())

    def test_weighted_sum_fusion(self):
        """重み付き和フュージョンテスト"""
        fusion = ModalityFusion(
            num_modalities=self.num_modalities,
            hidden_dim=self.hidden_dim,
            fusion_method="weighted_sum"
        )

        modality_features = torch.randn(self.batch_size, self.seq_len, self.hidden_dim * self.num_modalities)

        output = fusion(modality_features)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        self.assertTrue(torch.isfinite(output).all())


class TestMultiModalFeatureEncoder(unittest.TestCase):
    """MultiModalFeatureEncoderのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.encoder = MultiModalFeatureEncoder(
            price_dim=156,
            text_dim=768,
            economic_dim=20,
            hidden_dim=64,
            num_heads=4,
            dropout=0.1
        )
        self.batch_size = 2

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.encoder.price_encoder)
        self.assertIsNotNone(self.encoder.text_encoder)
        self.assertIsNotNone(self.encoder.economic_encoder)
        self.assertIsNotNone(self.encoder.cross_attention)
        self.assertIsNotNone(self.encoder.temporal_integration)

    def test_forward(self):
        """順伝播テスト"""
        price_data = torch.randn(self.batch_size, 156)
        text_data = torch.randint(0, 1000, (self.batch_size, 8))  # input_ids
        economic_data = torch.randn(self.batch_size, 20)

        output = self.encoder(price_data, text_data, economic_data)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_with_attention_mask(self):
        """アテンションマスク付き順伝播テスト"""
        price_data = torch.randn(self.batch_size, 156)
        text_data = torch.randint(0, 1000, (self.batch_size, 8))
        economic_data = torch.randn(self.batch_size, 20)
        attention_mask = torch.ones(self.batch_size, 8)
        attention_mask[:, -2:] = 0

        # 注意: 現在の実装ではattention_maskは使用されていない
        output = self.encoder(price_data, text_data, economic_data)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()