"""マルチモーダル学習モジュールの単体テスト

コアエンコーダーと設定管理クラスのテストを含む。
"""

import tempfile
import unittest
from pathlib import Path

import torch

from ztb.multimodal.config import MultimodalConfig

# テスト対象のインポート
from ztb.multimodal.core.encoders import EconomicEncoder, PriceEncoder, TextEncoder
from ztb.multimodal.training.trainers.multimodal_trainer import MultimodalSACTrainer


class TestPriceEncoder(unittest.TestCase):
    """PriceEncoderのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.encoder = PriceEncoder(input_dim=156, hidden_dims=[128, 64], output_dim=64)
        self.batch_size = 4
        self.seq_len = 10

    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.encoder.input_dim, 156)
        self.assertEqual(self.encoder.output_dim, 64)

    def test_forward_2d_input(self):
        """2D入力（batch_size, feature_dim）のテスト"""
        x = torch.randn(self.batch_size, 156)
        output = self.encoder(x)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_3d_input(self):
        """3D入力（batch_size, seq_len, feature_dim）のテスト"""
        x = torch.randn(self.batch_size, self.seq_len, 156)
        output = self.encoder(x)

        self.assertEqual(output.shape, (self.batch_size, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """勾配の流れテスト"""
        x = torch.randn(self.batch_size, 156)
        x.requires_grad_(True)

        output = self.encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestTextEncoder(unittest.TestCase):
    """TextEncoderのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.encoder = TextEncoder(
            model_name="bert-base-uncased", output_dim=768, fine_tune=True
        )
        self.batch_size = 2
        self.seq_len = 8

    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.encoder.model_name, "bert-base-uncased")
        self.assertEqual(self.encoder.output_dim, 768)
        self.assertTrue(self.encoder.fine_tune)

    def test_forward(self):
        """順伝播テスト"""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        output = self.encoder(input_ids)

        self.assertEqual(output.shape, (self.batch_size, 768))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_with_attention_mask(self):
        """アテンションマスク付き順伝播テスト"""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -2:] = 0  # 最後の2トークンをマスク

        output = self.encoder(input_ids, attention_mask)

        self.assertEqual(output.shape, (self.batch_size, 768))
        self.assertTrue(torch.isfinite(output).all())

    def test_fine_tune_parameter(self):
        """ファインチューニングパラメータテスト"""
        encoder_no_fine_tune = TextEncoder(fine_tune=False)

        # fine_tune=Falseの場合、BERTパラメータは固定されるべき
        for param in encoder_no_fine_tune.bert.parameters():
            self.assertFalse(param.requires_grad)


class TestEconomicEncoder(unittest.TestCase):
    """EconomicEncoderのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.encoder = EconomicEncoder(
            input_dim=20, hidden_dims=[64, 32], output_dim=32
        )
        self.batch_size = 3

    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.encoder.input_dim, 20)
        self.assertEqual(self.encoder.output_dim, 32)

    def test_forward(self):
        """順伝播テスト"""
        x = torch.randn(self.batch_size, 20)
        output = self.encoder(x)

        self.assertEqual(output.shape, (self.batch_size, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_gradient_flow(self):
        """勾配の流れテスト"""
        x = torch.randn(self.batch_size, 20)
        x.requires_grad_(True)

        output = self.encoder(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestMultimodalConfig(unittest.TestCase):
    """MultimodalConfigのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.config = MultimodalConfig()

    def test_default_initialization(self):
        """デフォルト初期化テスト"""
        self.assertEqual(self.config.version, "1.0.0")
        self.assertIsInstance(self.config.data, object)
        self.assertIsInstance(self.config.features, object)
        self.assertIsInstance(self.config.model, object)
        self.assertIsInstance(self.config.training, object)
        self.assertIsInstance(self.config.evaluation, object)
        self.assertIsInstance(self.config.hardware, object)
        self.assertIsInstance(self.config.api, object)

    def test_to_dict(self):
        """辞書変換テスト"""
        config_dict = self.config.to_dict()

        self.assertIn("version", config_dict)
        self.assertIn("data", config_dict)
        self.assertIn("features", config_dict)
        self.assertIn("model", config_dict)
        self.assertIn("training", config_dict)
        self.assertIn("evaluation", config_dict)
        self.assertIn("hardware", config_dict)
        self.assertIn("api", config_dict)

    def test_from_dict(self):
        """辞書からの設定作成テスト"""
        config_dict = {
            "version": "2.0.0",
            "data": {
                "symbols": ["EURUSD", "GBPUSD"],
                "timeframe": "5m",
                "lookback_days": 15,
            },
        }

        config = MultimodalConfig.from_dict(config_dict)

        self.assertEqual(config.version, "2.0.0")
        self.assertEqual(config.data.symbols, ["EURUSD", "GBPUSD"])
        self.assertEqual(config.data.timeframe, "5m")
        self.assertEqual(config.data.lookback_days, 15)

    def test_yaml_save_load(self):
        """YAML保存・読み込みテスト"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # 保存
            self.config.save_yaml(temp_path)

            # 読み込み
            loaded_config = MultimodalConfig.from_yaml(temp_path)

            # 比較
            self.assertEqual(self.config.version, loaded_config.version)
            self.assertEqual(self.config.data.symbols, loaded_config.data.symbols)

        finally:
            Path(temp_path).unlink()

    def test_env_var_expansion(self):
        """環境変数展開テスト"""
        import os

        # テスト環境変数を設定
        os.environ["TEST_NEWSAPI_KEY"] = "test_key_123"
        os.environ["TEST_FRED_KEY"] = "fred_key_456"

        config_dict = {
            "api": {
                "newsapi_key": "${TEST_NEWSAPI_KEY}",
                "fred_key": "${TEST_FRED_KEY}",
            }
        }

        config = MultimodalConfig.from_dict(config_dict)

        self.assertEqual(config.api.newsapi_key, "test_key_123")
        self.assertEqual(config.api.fred_key, "fred_key_456")

        # クリーンアップ
        del os.environ["TEST_NEWSAPI_KEY"]
        del os.environ["TEST_FRED_KEY"]


class TestMultimodalSACTrainer(unittest.TestCase):
    """MultimodalSACTrainerのテスト"""

    def setUp(self):
        # マルチモーダル設定
        self.multimodal_config = MultimodalConfig()
        self.multimodal_config.model.price_feature_dim = 156
        self.multimodal_config.features.embedding_dim = 768
        self.multimodal_config.model.economic_feature_dim = 10
        self.multimodal_config.model.action_dim = 3
        self.multimodal_config.model.attention_dim = 256
        self.multimodal_config.model.num_heads = 8

        # SAC設定
        self.sac_config = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        }

        # 環境設定
        self.env_config = {
            "observation_space": {"shape": (156,)},
            "action_space": {"n": 3},
        }

        self.trainer = MultimodalSACTrainer(
            multimodal_config=self.multimodal_config,
            sac_config=self.sac_config,
            env_config=self.env_config,
        )

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.trainer.multimodal_config, MultimodalConfig)
        self.assertEqual(self.trainer.sac_config, self.sac_config)
        self.assertEqual(self.trainer.env_config, self.env_config)
        self.assertIsNotNone(self.trainer.multimodal_agent)
        self.assertIsNone(self.trainer.data_loader)

    def test_multimodal_agent_configuration(self):
        """マルチモーダルエージェント設定テスト"""
        agent = self.trainer.multimodal_agent

        # 設定が正しく適用されていることを確認（feature_encoderを通じて）
        self.assertEqual(agent.feature_encoder.price_feature_dim, 156)
        self.assertEqual(agent.feature_encoder.text_embedding_dim, 768)
        self.assertEqual(agent.feature_encoder.economic_feature_dim, 10)


if __name__ == "__main__":
    unittest.main()
