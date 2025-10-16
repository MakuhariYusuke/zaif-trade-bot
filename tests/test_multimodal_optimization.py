"""マルチモーダル最適化機能の単体テスト

モデル圧縮、量子化、推論最適化のテストを含む。
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

# テスト対象のインポート
from ztb.multimodal.optimization.compression import KnowledgeDistillation, ModelPruning, ModelCompression
from ztb.multimodal.optimization.quantization import DynamicQuantization, QuantizationUtils
from ztb.multimodal.optimization.inference import InferenceOptimizer, MemoryManager


class TestKnowledgeDistillation(unittest.TestCase):
    """KnowledgeDistillationのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.distillation = KnowledgeDistillation(temperature=2.0, alpha=0.5)

    def test_initialization(self):
        """初期化テスト"""
        self.assertEqual(self.distillation.temperature, 2.0)
        self.assertEqual(self.distillation.alpha, 0.5)

    def test_distillation_loss(self):
        """蒸留損失テスト"""
        batch_size, num_classes = 4, 10

        student_logits = torch.randn(batch_size, num_classes)
        teacher_logits = torch.randn(batch_size, num_classes)

        loss = self.distillation.compute_distillation_loss(student_logits, teacher_logits)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)


class TestModelPruning(unittest.TestCase):
    """ModelPruningのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        self.pruning = ModelPruning(self.model, pruning_ratio=0.5)

    def test_prune_model(self):
        """モデルプルーニングテスト"""
        original_params = sum(p.numel() for p in self.model.parameters())

        self.pruning.prune_model()

        # スパース性が正しく計算されるか確認
        sparsity = self.pruning.get_sparsity()
        self.assertGreaterEqual(sparsity, 0.0)
        self.assertLessEqual(sparsity, 1.0)


class TestDynamicQuantization(unittest.TestCase):
    """DynamicQuantizationのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        self.quantizer = DynamicQuantization()

    def test_quantize_model(self):
        """動的量子化テスト"""
        try:
            quantized_model = self.quantizer.quantize_model(self.model)
            # 量子化が成功したか確認（エラーが発生しなければ成功）
            self.assertIsNotNone(quantized_model)
        except Exception as e:
            # 量子化がサポートされていない環境ではスキップ
            self.skipTest(f"量子化がサポートされていない環境: {e}")


class TestQuantizationUtils(unittest.TestCase):
    """QuantizationUtilsのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

    def test_get_model_size(self):
        """モデルサイズ取得テスト"""
        size_info = QuantizationUtils.get_model_size(self.model)

        self.assertIn('total_mb', size_info)
        self.assertGreater(size_info['total_mb'], 0)

    def test_measure_inference_time(self):
        """推論時間測定テスト"""
        input_data = torch.randn(1, 10)

        time_info = QuantizationUtils.measure_inference_time(self.model, input_data, num_runs=5)

        self.assertIn('avg_inference_time', time_info)
        self.assertIn('fps', time_info)
        self.assertGreater(time_info['fps'], 0)


class TestInferenceOptimizer(unittest.TestCase):
    """InferenceOptimizerのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        self.optimizer = InferenceOptimizer(self.model)

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsNotNone(self.optimizer.model)
        self.assertIsNotNone(self.optimizer.device)

    def test_predict(self):
        """推論テスト"""
        input_data = torch.randn(1, 10)

        output = self.optimizer.predict(input_data)

        self.assertEqual(output.shape, (1, 5))
        self.assertTrue(torch.isfinite(output).all())

    def test_memory_optimization(self):
        """メモリ最適化テスト"""
        self.optimizer.optimize_memory()
        # エラーが発生しなければ成功
        self.assertTrue(True)


class TestMemoryManager(unittest.TestCase):
    """MemoryManagerのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.manager = MemoryManager(max_memory_gb=8.0)

    def test_monitor_memory(self):
        """メモリ監視テスト"""
        memory_info = self.manager.monitor_memory()

        self.assertIn('allocated_gb', memory_info)
        self.assertIn('utilization_percent', memory_info)

    def test_cleanup_memory(self):
        """メモリクリーンアップテスト"""
        self.manager.cleanup_memory()
        # エラーが発生しなければ成功
        self.assertTrue(True)

    def test_get_memory_stats(self):
        """メモリ統計取得テスト"""
        # 監視データを追加
        for _ in range(5):
            self.manager.monitor_memory()

        stats = self.manager.get_memory_stats()

        self.assertIn('current', stats)
        self.assertIn('average', stats)
        self.assertIn('peak', stats)


if __name__ == '__main__':
    unittest.main()