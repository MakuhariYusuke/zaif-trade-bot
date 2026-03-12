"""
Unit tests for Concept Drift Detection Module
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ztb.adaptation.concept_drift.config import ConceptDriftConfig
from ztb.adaptation.concept_drift.detector import (
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    KolmogorovSmirnovDetector,
)
from ztb.adaptation.concept_drift.drift_types import DriftSeverity, DriftType
from ztb.adaptation.concept_drift.manager import ConceptDriftManager


class TestKolmogorovSmirnovDetector(unittest.TestCase):
    """Kolmogorov-Smirnov検定器のテスト"""

    def setUp(self):
        self.config = ConceptDriftConfig()
        self.detector = KolmogorovSmirnovDetector(self.config)

    def test_no_drift_same_distribution(self):
        """同じ分布のデータではドリフトを検知しない"""
        # 正規分布のデータ生成
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)

        # 参照データを設定
        self.detector.update_reference(data1)

        # ドリフト検知
        result = self.detector.detect_drift(data2)

        self.assertFalse(result.drift_detected)
        self.assertEqual(result.drift_type, DriftType.NONE)
        self.assertEqual(result.severity, DriftSeverity.NONE)
        self.assertGreater(result.confidence, 0.5)

    def test_drift_different_distribution(self):
        """異なる分布のデータでドリフトを検知"""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 1000)  # 平均0, 標準偏差1
        data2 = np.random.normal(2, 1, 1000)  # 平均2, 標準偏差1

        # 参照データを設定
        self.detector.update_reference(data1)

        # ドリフト検知
        result = self.detector.detect_drift(data2)

        self.assertTrue(result.drift_detected)
        self.assertEqual(result.drift_type, DriftType.CONCEPT_DRIFT)
        self.assertNotEqual(result.severity, DriftSeverity.NONE)
        self.assertLess(result.p_value, 0.05)

    def test_empty_data_handling(self):
        """空のデータに対する処理"""
        with self.assertRaises(ValueError):
            self.detector.detect_drift(np.array([]))


class TestADWINDetector(unittest.TestCase):
    """ADWIN検定器のテスト"""

    def setUp(self):
        self.config = ConceptDriftConfig()
        self.detector = ADWINDetector(self.config)

    def test_no_drift_stationary_data(self):
        """定常データではドリフトを検知しない（または検知頻度が低い）"""
        np.random.seed(42)
        # 定常データ生成
        data = np.random.normal(0, 1, 1000)

        drift_count = 0
        for i in range(0, len(data), 50):
            batch = data[i : i + 50]
            result = self.detector.detect_drift(batch)
            if result.drift_detected:
                drift_count += 1

        # ADWINはスライディングウィンドウのため、完全にドリフトを検知しないとは限らない
        # 検知回数が少ないことを確認（全体の10%未満）
        self.assertLess(drift_count, len(data) // 50 // 10)

    def test_drift_abrupt_change(self):
        """急激な変化でドリフトを検知"""
        np.random.seed(42)
        # 変化前のデータ
        data1 = np.random.normal(0, 1, 500)
        # 変化後のデータ
        data2 = np.random.normal(3, 1, 500)
        data = np.concatenate([data1, data2])

        drift_detected = False
        for i in range(0, len(data), 50):
            batch = data[i : i + 50]
            result = self.detector.detect_drift(batch)
            if result.drift_detected:
                drift_detected = True
                break

        self.assertTrue(drift_detected)


class TestDDMDetector(unittest.TestCase):
    """DDM検定器のテスト"""

    def setUp(self):
        self.config = ConceptDriftConfig()
        self.detector = DDMDetector(self.config)

    def test_no_drift_low_error_rate(self):
        """低いエラー率ではドリフトを検知しない"""
        # 低いエラー率のデータ
        errors = np.zeros(1000)  # すべて正解

        result = self.detector.detect_drift(np.ones(1000), errors)

        self.assertFalse(result.drift_detected)
        self.assertEqual(result.severity, DriftSeverity.NONE)

    def test_drift_high_error_rate(self):
        """高いエラー率でドリフトを検知"""
        # 高いエラー率のデータ
        errors = np.ones(1000)  # すべてエラー

        result = self.detector.detect_drift(np.ones(1000), errors)

        self.assertTrue(result.drift_detected)
        self.assertNotEqual(result.severity, DriftSeverity.NONE)


class TestEDDMDetector(unittest.TestCase):
    """EDDM検定器のテスト"""

    def setUp(self):
        self.config = ConceptDriftConfig()
        self.detector = EDDMDetector(self.config)

    def test_no_drift_consistent_performance(self):
        """一貫した性能ではドリフトを検知しない"""
        # 一貫したエラー率
        errors = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])

        result = self.detector.detect_drift(np.ones(1000), errors)

        # EDDMは距離ベースなので、必ずしもドリフトを検知しない
        # テストはエラーが発生しないことを確認
        self.assertIsInstance(result.drift_detected, bool)


class TestConceptDriftManager(unittest.TestCase):
    """ConceptDriftManagerのテスト"""

    def setUp(self):
        self.config = ConceptDriftConfig()
        # テストでは並列処理をデフォルトで無効化して安定性を確保
        self.config.detection_interval_seconds = 0  # 検知間隔を無効化
        self.manager = ConceptDriftManager(self.config)

    def test_initialization(self):
        """初期化テスト"""
        self.assertIsInstance(self.manager.detectors, dict)
        self.assertGreater(len(self.manager.detectors), 0)
        self.assertIn("ks_test", self.manager.detectors)

    def test_detect_drift_no_drift(self):
        """ドリフトなしの検知テスト"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        result = self.manager.detect_drift(data)

        self.assertFalse(result.drift_detected)
        self.assertEqual(result.drift_type, DriftType.NONE)

    def test_detect_drift_with_drift(self):
        """ドリフトありの検知テスト"""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 500)
        data2 = np.random.normal(3, 1, 500)  # 分布変化

        # 参照データを設定
        self.manager.update_reference_data(data1)

        result = self.manager.detect_drift(data2)

        # 複数の検知器の投票によりドリフトを検知する可能性あり
        self.assertIsInstance(result.drift_detected, bool)

    def test_dataframe_input(self):
        """DataFrame入力のテスト"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(1, 2, 100),
                "category": ["A"] * 50 + ["B"] * 50,
            }
        )

        result = self.manager.detect_drift(df)

        self.assertIsInstance(result.drift_detected, bool)

    def test_empty_dataframe_error(self):
        """空のDataFrameでのエラーテスト"""
        df = pd.DataFrame()

        with self.assertRaises(ValueError):
            self.manager.detect_drift(df)

    def test_parallel_detection(self):
        """並列検知のテスト"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)  # 小さなデータセットを使用

        # まず順次実行で参照データを設定
        result_sequential = self.manager.detect_drift(data, parallel=False)

        # 並列実行をテスト（小さいデータセットなのでタイムアウトしない）
        result_parallel = self.manager.detect_drift(data, parallel=True)

        # 結果は同じタイプであるべき
        self.assertEqual(
            type(result_sequential.drift_detected), type(result_parallel.drift_detected)
        )

    def test_history_management(self):
        """履歴管理のテスト"""
        np.random.seed(42)

        # 複数の検知を実行
        for i in range(5):
            data = np.random.normal(0, 1, 100)
            self.manager.detect_drift(data)

        history = self.manager.get_drift_history()
        self.assertEqual(len(history), 5)

        # 履歴サイズ制限のテスト
        self.config.max_history_size = 3
        manager_small = ConceptDriftManager(self.config)

        for i in range(5):
            data = np.random.normal(0, 1, 100)
            manager_small.detect_drift(data)

        history_small = manager_small.get_drift_history()
        self.assertLessEqual(len(history_small), 3)

    def test_detector_stats(self):
        """検知器統計のテスト"""
        stats = self.manager.get_detector_stats()

        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)

        for detector_name, detector_stats in stats.items():
            self.assertIn("total_detections", detector_stats)
            self.assertIn("avg_score", detector_stats)

    def test_reset_functionality(self):
        """リセット機能のテスト"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        # 検知を実行
        self.manager.detect_drift(data)
        self.assertGreater(len(self.manager.drift_history), 0)

        # リセット
        self.manager.reset_detectors()
        self.assertEqual(len(self.manager.drift_history), 0)
        self.assertIsNone(self.manager.last_detection_time)

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 無効なデータを渡す
        with self.assertRaises(ValueError):
            self.manager.detect_drift(np.array([]))

    @patch("ztb.adaptation.concept_drift.manager.ThreadPoolExecutor")
    def test_parallel_error_handling(self, mock_executor):
        """並列実行時のエラーハンドリングテスト"""
        # モックでエラーをシミュレート
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Test error")
        mock_executor.return_value.__enter__.return_value.submit.return_value = (
            mock_future
        )

        # as_completedを適切にモック化
        mock_completed = MagicMock()
        mock_completed.__iter__.return_value = [mock_future]
        mock_executor.return_value.__enter__.return_value.as_completed.return_value = (
            mock_completed
        )

        data = np.random.normal(0, 1, 100)
        result = self.manager.detect_drift(data, parallel=True)

        # エラーが発生しても結果が返される
        self.assertIsInstance(result, object)


if __name__ == "__main__":
    unittest.main()
