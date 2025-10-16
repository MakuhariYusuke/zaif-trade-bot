"""
データ処理パイプラインの実装

データ拡張、異常値処理、バリデーションを統合した包括的な
データ処理パイプラインを提供。
"""

import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import json

from .data_augmentation import DataAugmentation
from .outlier_detection import OutlierDetector, OutlierHandler
from .data_validation import DataValidator, DataIntegrityChecker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """パイプライン実行結果を格納するデータクラス。"""
    original_data: pd.DataFrame
    processed_data: pd.DataFrame
    validation_results: List[Any]
    processing_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]


class DataProcessingPipeline:
    """
    データ処理パイプラインのメインクラス。

    データ拡張、異常値処理、バリデーションを統合し、
    エンドツーエンドのデータ処理ワークフローを提供。
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        DataProcessingPipelineを初期化。

        Args:
            config_path: パイプライン設定ファイルのパス
        """
        self.config = self._load_config(config_path) if config_path else self._get_default_config()

        # コンポーネントの初期化
        self.augmenter = DataAugmentation(random_seed=self.config.get("random_seed"))
        self.outlier_detector = OutlierDetector(random_seed=self.config.get("random_seed"))
        self.outlier_handler = OutlierHandler()
        self.validator = DataValidator()
        self.integrity_checker = DataIntegrityChecker()

        # 処理統計
        self.processing_stats = {
            "start_time": None,
            "end_time": None,
            "steps_completed": [],
            "errors_encountered": [],
            "warnings_generated": []
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み。"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    return json.load(f)
                else:
                    raise ValueError("Unsupported config file format")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得。"""
        return {
            "random_seed": 42,
            "augmentation": {
                "enabled": True,
                "probability": 0.8,
                "techniques": [
                    {"type": "gaussian_noise", "std": 0.01},
                    {"type": "time_warping", "sigma": 0.1}
                ]
            },
            "outlier_detection": {
                "enabled": True,
                "methods": [
                    {"type": "z_score", "threshold": 3.0},
                    {"type": "iqr", "multiplier": 1.5}
                ]
            },
            "outlier_handling": {
                "enabled": True,
                "method": "interpolate",
                "interpolation_method": "linear"
            },
            "validation": {
                "enabled": True,
                "schema": {},
                "strict_mode": False
            },
            "quality_checks": {
                "enabled": True,
                "min_completeness": 0.95,
                "min_accuracy": 0.90
            }
        }

    def process_data(
        self,
        data: pd.DataFrame,
        steps: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        データ処理パイプラインを実行。

        Args:
            data: 処理対象のデータ
            steps: 実行するステップのリスト（Noneの場合は全ステップ）
            custom_config: カスタム設定（ベース設定を上書き）

        Returns:
            処理結果

        Example:
            >>> pipeline = DataProcessingPipeline()
            >>> result = pipeline.process_data(
            ...     data=my_dataframe,
            ...     steps=["validation", "outlier_detection", "augmentation"]
            ... )
        """
        self.processing_stats["start_time"] = datetime.now()

        # 設定の更新
        if custom_config:
            self._update_config(custom_config)

        # 実行ステップの決定
        if steps is None:
            steps = ["validation", "outlier_detection", "outlier_handling", "augmentation"]

        processed_data = data.copy()
        validation_results = []
        quality_metrics = {}

        logger.info(f"Starting data processing pipeline with steps: {steps}")

        try:
            for step in steps:
                logger.info(f"Executing step: {step}")

                if step == "validation":
                    validation_result = self._run_validation(processed_data)
                    validation_results.append(validation_result)
                    quality_metrics.update(validation_result.metrics)

                    if not validation_result.is_valid and self.config["validation"].get("strict_mode", False):
                        raise ValueError(f"Validation failed: {validation_result.errors}")

                elif step == "outlier_detection":
                    processed_data = self._run_outlier_detection(processed_data)

                elif step == "outlier_handling":
                    processed_data = self._run_outlier_handling(processed_data)

                elif step == "augmentation":
                    processed_data = self._run_augmentation(processed_data)

                else:
                    logger.warning(f"Unknown step: {step}")
                    continue

                self.processing_stats["steps_completed"].append(step)

            # 最終品質チェック
            if self.config["quality_checks"]["enabled"]:
                final_quality = self._run_quality_checks(processed_data)
                quality_metrics.update(final_quality)

        except Exception as e:
            logger.error(f"Pipeline execution failed at step {step}: {e}")
            self.processing_stats["errors_encountered"].append(str(e))
            raise

        finally:
            self.processing_stats["end_time"] = datetime.now()

        # 処理時間を計算
        if self.processing_stats["start_time"] and self.processing_stats["end_time"]:
            duration = self.processing_stats["end_time"] - self.processing_stats["start_time"]
            self.processing_stats["duration_seconds"] = duration.total_seconds()

        return PipelineResult(
            original_data=data,
            processed_data=processed_data,
            validation_results=validation_results,
            processing_stats=self.processing_stats.copy(),
            quality_metrics=quality_metrics
        )

    def _run_validation(self, data: pd.DataFrame) -> Any:
        """バリデーションを実行。"""
        schema = self.config["validation"].get("schema", {})

        # 基本スキーマの自動生成（設定されていない場合）
        if not schema:
            schema = self._generate_basic_schema(data)

        validation_result = self.validator.validate_data(data, schema)

        # 整合性チェックも実行
        integrity_result = self.integrity_checker.check_integrity(data)

        # 結果の統合
        combined_result = type('CombinedResult', (), {})()
        combined_result.is_valid = validation_result.is_valid and integrity_result.is_valid
        combined_result.errors = validation_result.errors + integrity_result.errors
        combined_result.warnings = validation_result.warnings + integrity_result.warnings
        combined_result.metrics = {**validation_result.metrics, **integrity_result.metrics}
        combined_result.details = {**validation_result.details, **integrity_result.details}

        return combined_result

    def _run_outlier_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """異常値検出を実行。"""
        methods = self.config["outlier_detection"]["methods"]
        columns = self._get_numeric_columns(data)

        result_data = self.outlier_detector.detect_outliers(
            data, methods, columns=columns
        )

        logger.info(f"Outlier detection completed. Detected outliers in {len(columns)} columns")
        return result_data

    def _run_outlier_handling(self, data: pd.DataFrame) -> pd.DataFrame:
        """異常値処理を実行。"""
        handling_config = self.config["outlier_handling"]
        method = handling_config["method"]

        processed_data = self.outlier_handler.handle_outliers(
            data, method=method, **handling_config
        )

        logger.info(f"Outlier handling completed using method: {method}")
        return processed_data

    def _run_augmentation(self, data: pd.DataFrame) -> pd.DataFrame:
        """データ拡張を実行。"""
        aug_config = self.config["augmentation"]
        techniques = aug_config["techniques"]
        probability = aug_config.get("probability", 1.0)

        augmented_data = self.augmenter.apply_augmentations(
            data, techniques, probability=probability
        )

        logger.info(f"Data augmentation completed with {len(techniques)} techniques")
        return augmented_data

    def _run_quality_checks(self, data: pd.DataFrame) -> Dict[str, float]:
        """品質チェックを実行。"""
        quality_config = self.config["quality_checks"]

        # 基本的な品質メトリクス計算
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))

        # 正確性の簡易チェック
        numeric_cols = self._get_numeric_columns(data)
        accuracy_scores = []

        for col in numeric_cols:
            valid_data = data[col].dropna()
            if len(valid_data) > 0:
                # 負の値や極端な値の割合
                negative_ratio = (valid_data < 0).mean()
                extreme_ratio = ((valid_data < valid_data.quantile(0.01)) |
                               (valid_data > valid_data.quantile(0.99))).mean()
                accuracy_scores.append(1 - (negative_ratio + extreme_ratio) / 2)

        accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.5

        quality_metrics = {
            "final_completeness": completeness,
            "final_accuracy": accuracy
        }

        # 閾値チェック
        min_completeness = quality_config.get("min_completeness", 0.95)
        min_accuracy = quality_config.get("min_accuracy", 0.90)

        if completeness < min_completeness:
            logger.warning(f"Completeness {completeness:.2%} below threshold {min_completeness:.2%}")

        if accuracy < min_accuracy:
            logger.warning(f"Accuracy {accuracy:.2%} below threshold {min_accuracy:.2%}")

        return quality_metrics

    def _generate_basic_schema(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """基本的なスキーマを自動生成。"""
        schema = {}

        for col in data.columns:
            col_schema = {"not_null": False}  # デフォルトはNull許可

            # データ型に基づくスキーマ設定
            dtype = data[col].dtype

            if np.issubdtype(dtype, np.number):
                col_schema["type"] = "float" if np.issubdtype(dtype, np.floating) else "int"

                # 数値列の範囲推定
                if len(data[col].dropna()) > 0:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    col_schema["range"] = [min_val, max_val]

            elif dtype == 'object':
                col_schema["type"] = "string"

            elif dtype == 'datetime64[ns]':
                col_schema["type"] = "datetime"
                col_schema["not_null"] = True  # タイムスタンプは必須

            # 列名に基づく特別設定
            col_lower = col.lower()
            if 'price' in col_lower:
                col_schema["range"] = [0, float('inf')]  # 価格は正の値
            elif 'volume' in col_lower:
                col_schema["range"] = [0, float('inf')]  # 出来高は正の値

            schema[col] = col_schema

        return schema

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """数値列を取得。"""
        return data.select_dtypes(include=[np.number]).columns.tolist()

    def _update_config(self, custom_config: Dict[str, Any]):
        """設定を更新。"""
        def update_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = update_dict(base[key], value)
                else:
                    base[key] = value
            return base

        self.config = update_dict(self.config, custom_config)

    def save_config(self, config_path: str):
        """設定をファイルに保存。"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif config_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError("Unsupported config file format")

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """パイプラインの現在の状態を取得。"""
        return {
            "config": self.config,
            "processing_stats": self.processing_stats,
            "components_initialized": {
                "augmenter": self.augmenter is not None,
                "outlier_detector": self.outlier_detector is not None,
                "outlier_handler": self.outlier_handler is not None,
                "validator": self.validator is not None,
                "integrity_checker": self.integrity_checker is not None
            }
        }


def create_financial_data_pipeline(
    augmentation_techniques: Optional[List[Dict[str, Any]]] = None,
    outlier_methods: Optional[List[Dict[str, Any]]] = None,
    validation_schema: Optional[Dict[str, Dict[str, Any]]] = None
) -> DataProcessingPipeline:
    """
    金融データ向けのデータ処理パイプラインを作成。

    Args:
        augmentation_techniques: データ拡張手法
        outlier_methods: 異常値検出手法
        validation_schema: バリデーションスキーマ

    Returns:
        設定済みのパイプライン
    """
    # デフォルト設定
    config = {
        "random_seed": 42,
        "augmentation": {
            "enabled": True,
            "probability": 0.7,
            "techniques": augmentation_techniques or [
                {"type": "gaussian_noise", "std": 0.005},
                {"type": "time_warping", "sigma": 0.05},
                {"type": "feature_mixing", "mix_ratio": 0.05}
            ]
        },
        "outlier_detection": {
            "enabled": True,
            "methods": outlier_methods or [
                {"type": "z_score", "threshold": 2.5},
                {"type": "iqr", "multiplier": 1.5},
                {"type": "isolation_forest", "contamination": 0.05}
            ]
        },
        "outlier_handling": {
            "enabled": True,
            "method": "interpolate",
            "interpolation_method": "linear"
        },
        "validation": {
            "enabled": True,
            "schema": validation_schema or {},
            "strict_mode": False
        },
        "quality_checks": {
            "enabled": True,
            "min_completeness": 0.95,
            "min_accuracy": 0.90
        }
    }

    pipeline = DataProcessingPipeline()
    pipeline._update_config(config)

    return pipeline