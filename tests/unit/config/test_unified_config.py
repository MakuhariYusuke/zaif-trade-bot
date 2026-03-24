"""
Unit tests for Unified Configuration System

統合設定管理システムの単体テスト
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from ztb.config.unified_config import ConfigFormat, UnifiedConfig, UnifiedConfigManager


class TestUnifiedConfig(unittest.TestCase):
    """UnifiedConfigクラスのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.sample_config = {
            "model_name": "test_model",
            "version": "1.0.0",
            "algorithm": "sac",
            "description": "Test configuration",
            "training": {
                "total_timesteps": 1000,
                "learning_rate": 0.001,
                "batch_size": 64,
                "buffer_size": 10000,
            },
            "features": {
                "basic_features": ["open", "high", "low", "close"],
                "technical_indicators": ["rsi", "macd"],
            },
            "reward_settings": {"base_profit_bonus_atr_coeff": 5.0},
            "ensemble_system": {"enabled": True},
            "market_regimes": {"bull_high_vol": {"correlation_target": 0.3}},
            "validation": {"enabled": True},
            "logging": {"tensorboard_log": "./logs"},
            "checkpoint": {"save_freq": 100},
        }

    def _make_temp_path(self, suffix: str) -> Path:
        """Create a temp file path and register cleanup."""
        fd, name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        path = Path(name)
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_unified_config_creation(self):
        """UnifiedConfigの作成テスト"""
        config = UnifiedConfig.from_dict(self.sample_config)

        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.version, "1.0.0")
        self.assertEqual(config.algorithm, "sac")
        self.assertEqual(config.description, "Test configuration")
        self.assertEqual(config.training["total_timesteps"], 1000)
        self.assertEqual(len(config.features), 2)
        self.assertEqual(config.get_feature_count(), 6)

    def test_unified_config_to_dict(self):
        """UnifiedConfigの辞書変換テスト"""
        config = UnifiedConfig.from_dict(self.sample_config)
        config_dict = config.to_dict()

        self.assertEqual(config_dict["model_name"], "test_model")
        self.assertEqual(config_dict["version"], "1.0.0")
        self.assertEqual(config_dict["algorithm"], "sac")
        self.assertIn("features", config_dict)
        self.assertIn("training", config_dict)

    def test_unified_config_validation_valid(self):
        """有効な設定の検証テスト"""
        config = UnifiedConfig.from_dict(self.sample_config)
        errors = config.validate()

        self.assertEqual(len(errors), 0)

    def test_unified_config_validation_invalid(self):
        """無効な設定の検証テスト"""
        invalid_config = self.sample_config.copy()
        invalid_config["model_name"] = ""  # 必須フィールドを空に
        invalid_config["algorithm"] = ""  # 必須フィールドを空に

        config = UnifiedConfig.from_dict(invalid_config)
        errors = config.validate()

        self.assertGreater(len(errors), 0)
        self.assertIn("Missing or empty required field", str(errors))

    def test_unified_config_save_load_json(self):
        """JSON形式での保存・読み込みテスト"""
        config = UnifiedConfig.from_dict(self.sample_config)
        temp_path = self._make_temp_path(".json")

        # 保存
        config.save(temp_path, ConfigFormat.JSON)

        # 読み込み
        loaded_config = UnifiedConfig.from_file(temp_path, ConfigFormat.JSON)

        # 検証
        self.assertEqual(loaded_config.model_name, config.model_name)
        self.assertEqual(loaded_config.version, config.version)
        self.assertEqual(
            loaded_config.get_feature_count(), config.get_feature_count()
        )

    def test_unified_config_save_load_yaml(self):
        """YAML形式での保存・読み込みテスト"""
        config = UnifiedConfig.from_dict(self.sample_config)
        temp_path = self._make_temp_path(".yaml")

        # 保存
        config.save(temp_path, ConfigFormat.YAML)

        # 読み込み
        loaded_config = UnifiedConfig.from_file(temp_path, ConfigFormat.YAML)

        # 検証
        self.assertEqual(loaded_config.model_name, config.model_name)
        self.assertEqual(loaded_config.version, config.version)

    def test_unified_config_auto_format_detection(self):
        """自動形式検知テスト"""
        config = UnifiedConfig.from_dict(self.sample_config)

        # JSONファイル
        temp_path = self._make_temp_path(".json")
        config.save(temp_path, ConfigFormat.JSON)
        loaded_config = UnifiedConfig.from_file(temp_path, ConfigFormat.AUTO)
        self.assertEqual(loaded_config.model_name, config.model_name)

        # YAMLファイル
        temp_path = self._make_temp_path(".yaml")
        config.save(temp_path, ConfigFormat.YAML)
        loaded_config = UnifiedConfig.from_file(temp_path, ConfigFormat.AUTO)
        self.assertEqual(loaded_config.model_name, config.model_name)

    def test_unified_config_file_not_found(self):
        """存在しないファイルの読み込みテスト"""
        with self.assertRaises(FileNotFoundError):
            UnifiedConfig.from_file(Path("nonexistent_file.json"))

    def test_unified_config_empty_features(self):
        """空の特徴量設定テスト"""
        config_data = self.sample_config.copy()
        config_data["features"] = {}

        config = UnifiedConfig.from_dict(config_data)
        self.assertEqual(config.get_feature_count(), 0)

        errors = config.validate()
        self.assertIn("No features configured", errors)


class TestUnifiedConfigManager(unittest.TestCase):
    """UnifiedConfigManagerクラスのテスト"""

    def setUp(self):
        """テスト前の準備"""
        self.manager = UnifiedConfigManager()
        self.sample_config = {
            "model_name": "test_model",
            "version": "1.0.0",
            "algorithm": "sac",
            "training": {
                "total_timesteps": 1000,
                "learning_rate": 0.001,
                "batch_size": 64,
                "buffer_size": 10000,
            },
            "features": {"basic_features": ["open", "high"]},
        }

    def _write_temp_json(self, data: dict[str, object]) -> Path:
        """Write JSON data to a temp path with automatic cleanup."""
        fd, name = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        path = Path(name)
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_config_manager_creation(self):
        """ConfigManagerの作成テスト"""
        self.assertIsInstance(self.manager, UnifiedConfigManager)
        self.assertEqual(len(self.manager.list_configs()), 0)

    def test_config_manager_load_config(self):
        """設定の読み込みテスト"""
        temp_path = self._write_temp_json(self.sample_config)

        config = self.manager.load_config("test_config", temp_path)

        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(len(self.manager.list_configs()), 1)
        self.assertIn("test_config", self.manager.list_configs())

        # 取得テスト
        retrieved_config = self.manager.get_config("test_config")
        self.assertEqual(retrieved_config.model_name, "test_model")

    def test_config_manager_get_nonexistent_config(self):
        """存在しない設定の取得テスト"""
        config = self.manager.get_config("nonexistent")
        self.assertIsNone(config)

    def test_config_manager_validate_all_configs(self):
        """全設定の検証テスト"""
        # 有効な設定を追加
        temp_path = self._write_temp_json(self.sample_config)
        self.manager.load_config("valid_config", temp_path)

        # 無効な設定を追加
        invalid_config = self.sample_config.copy()
        invalid_config["model_name"] = ""
        temp_path2 = self._write_temp_json(invalid_config)

        self.manager.load_config("invalid_config", temp_path2)

        # 検証
        results = self.manager.validate_all_configs()

        # 有効な設定はエラーがないためresultsに含まれない
        self.assertNotIn("valid_config", results)
        # 無効な設定のみresultsに含まれる
        self.assertIn("invalid_config", results)
        self.assertGreater(len(results["invalid_config"]), 0)  # 無効な設定


if __name__ == "__main__":
    unittest.main()
