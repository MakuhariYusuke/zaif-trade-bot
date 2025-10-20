"""
Unit tests for Online Learning Pipeline
インクリメンタル学習とストリーミングデータ処理のテスト
"""

import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from ztb.adaptation.online_learning.config import OnlineLearningConfig
from ztb.adaptation.online_learning.pipeline import (
    DriftDetector,
    OnlineLearningPipeline,
    ResourceMonitor,
)
from ztb.adaptation.online_learning.types import DataBatch, MemoryStrategy


class SimpleTestModel(nn.Module):
    """テスト用シンプルモデル"""

    def __init__(
        self, input_size: int = 10, hidden_size: int = 32, output_size: int = 1
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


class TestOnlineLearningPipeline(unittest.TestCase):
    """OnlineLearningPipelineのテスト"""

    def setUp(self):
        self.config = OnlineLearningConfig()
        self.config.batch_size = 16
        self.config.max_memory_samples = 100
        self.model = SimpleTestModel()
        self.pipeline = OnlineLearningPipeline(self.config, self.model)

    def tearDown(self):
        if (
            hasattr(self.pipeline, "streaming_thread")
            and self.pipeline.streaming_thread
        ):
            self.pipeline.stop_streaming()

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.pipeline.model, nn.Module)
        self.assertIsInstance(self.pipeline.optimizer, torch.optim.Optimizer)
        self.assertEqual(self.pipeline.learning_state.total_samples_processed, 0)
        self.assertEqual(len(self.pipeline.memory_buffer), 0)

    def test_single_update(self):
        """単一更新テスト"""
        # テストデータ作成
        batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32),
            targets=np.random.randn(16, 1).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="test_batch_1",
        )

        # モデル更新
        result = self.pipeline.update_model(batch)

        # 結果検証
        self.assertTrue(result.success)
        self.assertGreater(result.processing_time_ms, 0)
        self.assertEqual(
            result.parameter_updates, sum(p.numel() for p in self.model.parameters())
        )

        # 学習状態更新確認
        self.assertEqual(self.pipeline.learning_state.total_samples_processed, 1)
        self.assertEqual(len(self.pipeline.learning_state.loss_history), 1)

    def test_memory_management_sliding_window(self):
        """スライディングウィンドウメモリ管理テスト"""
        self.pipeline.config.memory_strategy = MemoryStrategy.SLIDING_WINDOW
        self.pipeline.config.max_memory_samples = 5

        # 複数バッチ追加
        for i in range(10):
            batch = DataBatch(
                features=np.random.randn(16, 10).astype(np.float32),
                targets=np.random.randn(16, 1).astype(np.float32),
                weights=None,
                timestamps=[datetime.now()] * 16,
                batch_id=f"batch_{i}",
            )
            self.pipeline.update_model(batch)

        # メモリバッファサイズ確認（最大5個）
        self.assertEqual(len(self.pipeline.memory_buffer), 5)

    def test_memory_management_importance_sampling(self):
        """重要度サンプリングメモリ管理テスト"""
        self.pipeline.config.memory_strategy = MemoryStrategy.IMPORTANCE_SAMPLING
        self.pipeline.config.max_memory_samples = 3

        # 複数バッチ追加
        for i in range(5):
            batch = DataBatch(
                features=np.random.randn(16, 10).astype(np.float32),
                targets=np.random.randn(16, 1).astype(np.float32),
                weights=None,
                timestamps=[datetime.now()] * 16,
                batch_id=f"batch_{i}",
            )
            self.pipeline.update_model(batch)

        # メモリバッファサイズ確認
        self.assertLessEqual(len(self.pipeline.memory_buffer), 3)

    def test_time_decay_memory(self):
        """時間減衰メモリ管理テスト"""
        self.pipeline.config.memory_strategy = MemoryStrategy.TIME_DECAY
        self.pipeline.config.max_memory_samples = 10

        # 過去のバッチ追加
        past_time = datetime.now() - timedelta(hours=2)
        batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32),
            targets=np.random.randn(16, 1).astype(np.float32),
            weights=np.ones(16),
            timestamps=[past_time] * 16,
            batch_id="old_batch",
        )
        self.pipeline.memory_buffer.append(batch)

        # 時間減衰適用
        self.pipeline._apply_time_decay()

        # 重みが減衰していることを確認
        self.assertLess(self.pipeline.memory_buffer[0].weights[0], 1.0)

    def test_checkpoint_creation(self):
        """チェックポイント作成テスト"""
        # 学習実行
        batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32),
            targets=np.random.randn(16, 1).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="checkpoint_test",
        )
        self.pipeline.update_model(batch)

        # チェックポイント作成
        self.pipeline._create_checkpoint()

        # チェックポイント確認
        checkpoints = self.pipeline.get_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        self.assertIn("1.0.0", checkpoints[0].version)

    def test_checkpoint_loading(self):
        """チェックポイント読み込みテスト"""
        # チェックポイント作成
        self.pipeline._create_checkpoint()
        checkpoints = self.pipeline.get_checkpoints()
        checkpoint = checkpoints[0]

        # 新しいモデルで読み込みテスト
        new_model = SimpleTestModel()
        new_pipeline = OnlineLearningPipeline(self.config, new_model)

        success = new_pipeline.load_checkpoint(checkpoint)
        self.assertTrue(success)
        self.assertEqual(new_pipeline.learning_state.model_version, checkpoint.version)

    def test_performance_monitoring(self):
        """パフォーマンス監視テスト"""
        # 複数更新実行
        for i in range(3):
            batch = DataBatch(
                features=np.random.randn(16, 10).astype(np.float32),
                targets=np.random.randn(16, 1).astype(np.float32),
                weights=None,
                timestamps=[datetime.now()] * 16,
                batch_id=f"perf_batch_{i}",
            )
            self.pipeline.update_model(batch)

        # パフォーマンスメトリクス取得
        metrics = self.pipeline.get_performance_metrics()
        self.assertEqual(len(metrics), 3)

        # メトリクス内容確認
        latest_metric = metrics[-1]
        self.assertIn("update_result", latest_metric)
        self.assertIn("resource_metrics", latest_metric)
        self.assertIn("learning_state", latest_metric)

    def test_gradient_clipping(self):
        """勾配クリッピングテスト"""
        self.pipeline.config.gradient_clipping = 0.1

        # 大きな勾配を発生させるデータ
        batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32) * 10,
            targets=np.random.randn(16, 1).astype(np.float32) * 10,
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="clip_test",
        )

        result = self.pipeline.update_model(batch)

        # 勾配ノルムがクリッピングされていることを確認
        self.assertLessEqual(
            result.gradient_norm, self.pipeline.config.gradient_clipping + 0.1
        )

    def test_streaming_processing(self):
        """ストリーミング処理テスト"""
        # ストリーミング設定を調整（小さいバッチサイズ）
        self.pipeline.config.streaming_config.batch_size = 4

        # ストリーミングデータジェネレータ
        def data_generator():
            for i in range(6):  # バッファサイズを超えるデータ
                yield DataBatch(
                    features=np.random.randn(4, 10).astype(np.float32),
                    targets=np.random.randn(4, 1).astype(np.float32),
                    weights=None,
                    timestamps=[datetime.now()] * 4,
                    batch_id=f"stream_batch_{i}",
                )
                time.sleep(0.01)  # 短い遅延

        # ストリーミング開始
        self.pipeline.start_streaming(data_generator())

        # 処理待機（バッファが複数回処理されるまで）
        time.sleep(0.2)

        # ストリーミング停止
        self.pipeline.stop_streaming()

        # 学習状態確認
        self.assertGreater(self.pipeline.learning_state.total_samples_processed, 0)


class TestDriftDetector(unittest.TestCase):
    """DriftDetectorのテスト"""

    def setUp(self):
        self.detector = DriftDetector(threshold=0.1)

    def test_initial_detection(self):
        """初期検知テスト"""
        batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32),
            targets=np.random.randn(16, 1).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="init_batch",
        )

        # 初期データではドリフト検知されない
        drift_detected, drift_type = self.detector.detect_drift(batch)
        self.assertFalse(drift_detected)
        self.assertEqual(drift_type, "none")

    def test_sudden_drift_detection(self):
        """突然ドリフト検知テスト"""
        # 初期データ
        initial_batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32),
            targets=np.random.randn(16, 1).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="init_batch",
        )
        self.detector.detect_drift(initial_batch)

        # 分布が大きく変化したデータ
        drift_batch = DataBatch(
            features=np.random.randn(16, 10).astype(np.float32) + 5.0,  # 平均値シフト
            targets=np.random.randn(16, 1).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="drift_batch",
        )

        drift_detected, drift_type = self.detector.detect_drift(drift_batch)
        self.assertTrue(drift_detected)
        self.assertEqual(drift_type, "sudden_drift")

    def test_no_drift_detection(self):
        """ドリフトなしテスト"""
        # 非常に緩い閾値でドリフト検知器を作成
        detector = DriftDetector(threshold=2.0)  # 非常に大きな閾値

        # 決定論的な初期データ（同じ値）
        np.random.seed(42)  # 再現性のためにシード設定
        initial_features = np.ones((16, 10)).astype(np.float32)  # すべて1のデータ
        initial_batch = DataBatch(
            features=initial_features,
            targets=np.ones((16, 1)).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="init_batch",
        )
        detector.detect_drift(initial_batch)

        # ほぼ同じデータ（ドリフト検知されないはず）
        np.random.seed(43)  # 異なるシードだが似たデータ
        similar_features = (
            np.ones((16, 10)).astype(np.float32) + 0.001
        )  # 非常に小さな変化
        similar_batch = DataBatch(
            features=similar_features,
            targets=np.ones((16, 1)).astype(np.float32),
            weights=None,
            timestamps=[datetime.now()] * 16,
            batch_id="similar_batch",
        )

        drift_detected, drift_type = detector.detect_drift(similar_batch)
        self.assertFalse(drift_detected)
        self.assertEqual(drift_type, "none")


class TestResourceMonitor(unittest.TestCase):
    """ResourceMonitorのテスト"""

    def setUp(self):
        self.monitor = ResourceMonitor()

    def test_metrics_collection(self):
        """メトリクス収集テスト"""
        metrics = self.monitor.get_metrics()

        # 基本的なメトリクス存在確認
        self.assertIsInstance(metrics.cpu_usage_percent, (int, float))
        self.assertIsInstance(metrics.memory_usage_mb, (int, float))
        self.assertIsInstance(metrics.disk_io_mb_per_sec, (int, float))
        self.assertIsInstance(metrics.network_io_mb_per_sec, (int, float))
        self.assertIsInstance(metrics.timestamp, datetime)

        # CPU使用率範囲確認
        self.assertGreaterEqual(metrics.cpu_usage_percent, 0.0)
        self.assertLessEqual(metrics.cpu_usage_percent, 100.0)

        # メモリ使用量確認
        self.assertGreater(metrics.memory_usage_mb, 0.0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 1024)  # 1GB
    def test_gpu_metrics(self, mock_memory, mock_cuda_available):
        """GPUメトリクステスト"""
        metrics = self.monitor.get_metrics()

        # GPUメモリが収集されていることを確認
        self.assertIsNotNone(metrics.gpu_memory_mb)
        if metrics.gpu_memory_mb is not None:
            self.assertGreater(metrics.gpu_memory_mb, 0.0)

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_gpu_metrics(self, mock_cuda_available):
        """GPUなし環境テスト"""
        metrics = self.monitor.get_metrics()

        # GPUメモリがNoneであることを確認
        self.assertIsNone(metrics.gpu_memory_mb)


class TestOnlineLearningSACTrainer(unittest.TestCase):
    """OnlineLearningSACTrainerのテスト"""

    def setUp(self):
        from ztb.adaptation.online_learning.trainer import OnlineLearningSACTrainer

        self.online_config = OnlineLearningConfig()
        self.online_config.batch_size = 16
        self.online_config.max_memory_samples = 100

        self.sac_config = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
        }

        self.env_config = {
            "observation_space": {"shape": (10,)},
            "action_space": {"n": 3},
        }

        self.trainer = OnlineLearningSACTrainer(
            online_config=self.online_config,
            sac_config=self.sac_config,
            env_config=self.env_config,
        )

    def tearDown(self):
        # オンライン学習がアクティブな場合は停止
        if self.trainer.is_online_learning_active:
            self.trainer.stop_online_learning()

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.trainer.online_config, OnlineLearningConfig)
        self.assertEqual(self.trainer.sac_config, self.sac_config)
        self.assertEqual(self.trainer.env_config, self.env_config)
        self.assertFalse(self.trainer.is_online_learning_active)
        self.assertIsNone(self.trainer.online_thread)
        self.assertIsNone(self.trainer.data_stream)


if __name__ == "__main__":
    unittest.main()
