"""
データ処理機能のテスト

データ拡張、異常値処理、バリデーション機能のテスト。
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .data_augmentation import DataAugmentation
from .outlier_detection import OutlierDetector, OutlierHandler
from .data_validation import DataValidator, DataIntegrityChecker
from .data_processing_pipeline import DataProcessingPipeline, create_financial_data_pipeline


class TestDataAugmentation(unittest.TestCase):
    """DataAugmentationクラスのテスト。"""

    def setUp(self):
        """テスト前の準備。"""
        self.augmenter = DataAugmentation(random_seed=42)

        # サンプルデータ作成
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'price': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 100, 100),
            'timestamp': dates
        })

    def test_gaussian_noise(self):
        """ガウスノイズ拡張のテスト。"""
        original_std = self.sample_data['price'].std()

        augmented = self.augmenter._add_gaussian_noise(
            self.sample_data.copy(), std=0.01, columns=['price']
        )

        # ノイズが追加されていることを確認
        self.assertGreater(augmented['price'].std(), original_std)

    def test_time_warping(self):
        """時間軸ワーピングのテスト。"""
        augmented = self.augmenter._apply_time_warping(
            self.sample_data.copy(), sigma=0.2, columns=['price']
        )

        # データ形状が維持されていることを確認
        self.assertEqual(len(augmented), len(self.sample_data))
        self.assertTrue('price' in augmented.columns)

    def test_augmentation_pipeline(self):
        """拡張パイプラインのテスト。"""
        augmentations = [
            {"type": "gaussian_noise", "std": 0.01},
            {"type": "time_warping", "sigma": 0.1}
        ]

        augmented = self.augmenter.apply_augmentations(
            self.sample_data.copy(), augmentations, probability=1.0
        )

        # データが変更されていることを確認
        self.assertFalse(augmented['price'].equals(self.sample_data['price']))


class TestOutlierDetection(unittest.TestCase):
    """OutlierDetectorクラスのテスト。"""

    def setUp(self):
        """テスト前の準備。"""
        self.detector = OutlierDetector(random_seed=42)

        # 正常データと異常値を含むサンプルデータ
        np.random.seed(42)
        normal_data = np.random.normal(100, 5, 98)
        outliers = np.array([200, -50])  # 異常値
        all_data = np.concatenate([normal_data, outliers])

        self.sample_data = pd.DataFrame({
            'price': all_data,
            'volume': np.random.normal(1000, 100, 100)
        })

    def test_z_score_detection(self):
        """Z-score法による異常値検出のテスト。"""
        flags = self.detector._detect_z_score(
            self.sample_data, ['price'], threshold=2.0
        )

        # 異常値が検出されていることを確認
        self.assertTrue(flags['price'].sum() > 0)

    def test_iqr_detection(self):
        """IQR法による異常値検出のテスト。"""
        flags = self.detector._detect_iqr(
            self.sample_data, ['price'], multiplier=1.5
        )

        # 異常値が検出されていることを確認
        self.assertTrue(flags['price'].sum() > 0)

    def test_outlier_detection_pipeline(self):
        """異常値検出パイプラインのテスト。"""
        methods = [
            {"type": "z_score", "threshold": 2.0},
            {"type": "iqr", "multiplier": 1.5}
        ]

        result = self.detector.detect_outliers(
            self.sample_data, methods, columns=['price']
        )

        # 異常値フラグ列が追加されていることを確認
        self.assertTrue('price_is_outlier' in result.columns)


class TestOutlierHandler(unittest.TestCase):
    """OutlierHandlerクラスのテスト。"""

    def setUp(self):
        """テスト前の準備。"""
        self.handler = OutlierHandler()

        # 異常値を含むサンプルデータ
        data = pd.DataFrame({
            'price': [100, 101, 200, 102, 103, -50, 104, 105],
            'price_is_outlier': [False, False, True, False, False, True, False, False]
        })
        self.sample_data = data

    def test_remove_outliers(self):
        """異常値除去のテスト。"""
        processed = self.handler._remove_outliers(
            self.sample_data.copy(), ['price_is_outlier']
        )

        # 異常値が除去されていることを確認
        self.assertEqual(len(processed), 6)  # 元の8行から2行除去

    def test_interpolate_outliers(self):
        """異常値補間のテスト。"""
        processed = self.handler._interpolate_outliers(
            self.sample_data.copy(), ['price_is_outlier'], method='linear'
        )

        # 異常値が補間されていることを確認
        self.assertFalse(processed['price'].isnull().any())


class TestDataValidator(unittest.TestCase):
    """DataValidatorクラスのテスト。"""

    def setUp(self):
        """テスト前の準備。"""
        self.validator = DataValidator()

        # サンプルデータ
        self.sample_data = pd.DataFrame({
            'price': [100.5, 101.2, 99.8, None, 102.1],
            'volume': [1000, 1100, 900, 1050, 950],
            'timestamp': pd.date_range('2023-01-01', periods=5)
        })

        # サンプルスキーマ
        self.schema = {
            'price': {'type': 'float', 'range': [0, 200], 'not_null': True},
            'volume': {'type': 'int', 'range': [0, 2000], 'not_null': True},
            'timestamp': {'type': 'datetime', 'not_null': True}
        }

    def test_schema_validation(self):
        """スキーマバリデーションのテスト。"""
        result = self.validator._validate_schema(self.sample_data, self.schema)

        # Null値に関するエラーが検出されていることを確認
        self.assertTrue(len(result['errors']) > 0)

    def test_data_quality_metrics(self):
        """データ品質メトリクスのテスト。"""
        metrics = self.validator._calculate_quality_metrics(self.sample_data)

        # メトリクスが計算されていることを確認
        self.assertTrue(hasattr(metrics, 'completeness'))
        self.assertTrue(hasattr(metrics, 'accuracy'))
        self.assertTrue(0 <= metrics.completeness <= 1)


class TestDataIntegrityChecker(unittest.TestCase):
    """DataIntegrityCheckerクラスのテスト。"""

    def setUp(self):
        """テスト前の準備。"""
        self.checker = DataIntegrityChecker()

        # サンプルデータ
        self.sample_data = pd.DataFrame({
            'price': [100.5, 101.2, 99.8, 102.1, 103.0],
            'volume': [1000, 1100, 900, 1050, 950],
            'timestamp': pd.date_range('2023-01-01', periods=5)
        })

    def test_data_type_check(self):
        """データ型チェックのテスト。"""
        result = self.checker._check_data_types(self.sample_data)

        # エラーがないことを確認
        self.assertEqual(len(result['errors']), 0)

    def test_temporal_consistency(self):
        """時系列整合性チェックのテスト。"""
        result = self.checker._check_temporal_consistency(self.sample_data)

        # タイムスタンプが順序通りに並んでいることを確認
        self.assertEqual(len(result['errors']), 0)


class TestDataProcessingPipeline(unittest.TestCase):
    """DataProcessingPipelineクラスのテスト。"""

    def setUp(self):
        """テスト前の準備。"""
        self.pipeline = DataProcessingPipeline()

        # サンプルデータ
        dates = pd.date_range('2023-01-01', periods=50, freq='H')
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'price': np.random.normal(100, 5, 50),
            'volume': np.random.normal(1000, 100, 50),
            'timestamp': dates
        })

    def test_pipeline_execution(self):
        """パイプライン実行のテスト。"""
        result = self.pipeline.process_data(
            self.sample_data,
            steps=['validation', 'outlier_detection']
        )

        # 結果が返されていることを確認
        self.assertIsInstance(result, type(result))
        self.assertEqual(len(result.original_data), len(self.sample_data))

    def test_financial_pipeline_creation(self):
        """金融データ向けパイプライン作成のテスト。"""
        pipeline = create_financial_data_pipeline()

        # パイプラインが作成されていることを確認
        self.assertIsInstance(pipeline, DataProcessingPipeline)

    def test_config_update(self):
        """設定更新のテスト。"""
        custom_config = {
            "augmentation": {
                "enabled": False
            }
        }

        original_setting = self.pipeline.config["augmentation"]["enabled"]
        self.pipeline._update_config(custom_config)

        # 設定が更新されていることを確認
        self.assertNotEqual(self.pipeline.config["augmentation"]["enabled"], original_setting)


if __name__ == '__main__':
    unittest.main()