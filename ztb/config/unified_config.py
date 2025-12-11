"""
Unified Configuration System

統合設定管理システム
すべての設定ファイルを統一的に管理し、型安全性を確保
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConfigFormat(Enum):
    """設定ファイル形式"""

    JSON = "json"
    YAML = "yaml"
    AUTO = "auto"


class ConfigType(Enum):
    """設定タイプ"""

    TRAINING = "training"
    FEATURES = "features"
    ENVIRONMENT = "environment"
    MODEL = "model"
    EVALUATION = "evaluation"


@dataclass
class UnifiedConfig:
    """
    統合設定クラス

    すべての設定を統一的に管理し、型安全性を確保
    """

    # 基本情報
    model_name: str
    version: str
    algorithm: str
    description: str = ""

    # トレーニング設定
    training: Dict[str, Any] = field(default_factory=dict)

    # 環境設定
    environment: Dict[str, Any] = field(default_factory=dict)

    # 特徴量設定
    features: Dict[str, List[str]] = field(default_factory=dict)

    # 報酬設定
    reward_settings: Dict[str, Any] = field(default_factory=dict)

    # アンサンブル設定
    ensemble_system: Dict[str, Any] = field(default_factory=dict)

    # 市場レジーム設定
    market_regimes: Dict[str, Any] = field(default_factory=dict)

    # 検証設定
    validation: Dict[str, Any] = field(default_factory=dict)

    # ログ設定
    logging: Dict[str, Any] = field(default_factory=dict)

    # チェックポイント設定
    checkpoint: Dict[str, Any] = field(default_factory=dict)

    # メタデータ
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        config_path: Union[str, Path],
        format_type: ConfigFormat = ConfigFormat.AUTO,
    ) -> "UnifiedConfig":
        """
        設定ファイルを読み込んでUnifiedConfigを作成

        Args:
            config_path: 設定ファイルのパス
            format_type: ファイル形式

        Returns:
            UnifiedConfigインスタンス

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValueError: ファイル形式がサポートされていない場合
            json.JSONDecodeError: JSONファイルのパースエラーの場合
            yaml.YAMLError: YAMLファイルのパースエラーの場合
            Exception: その他の予期しないエラーの場合
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # 形式の自動判定
        if format_type == ConfigFormat.AUTO:
            if path.suffix.lower() == ".json":
                format_type = ConfigFormat.JSON
            elif path.suffix.lower() in [".yaml", ".yml"]:
                format_type = ConfigFormat.YAML
            else:
                raise ValueError(
                    f"Unsupported file extension: {path.suffix}. Supported: .json, .yaml, .yml"
                )

        # ファイル読み込み
        try:
            with open(path, "r", encoding="utf-8") as f:
                if format_type == ConfigFormat.JSON:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Invalid JSON format in {config_path}: {e}", e.doc, e.pos
                        )
                else:  # YAML
                    try:
                        data = yaml.safe_load(f)
                        if data is None:
                            raise ValueError(
                                f"YAML file is empty or contains only comments: {config_path}"
                            )
                    except yaml.YAMLError as e:
                        raise yaml.YAMLError(
                            f"Invalid YAML format in {config_path}: {e}"
                        )
        except (IOError, OSError) as e:
            raise IOError(f"Failed to read config file {config_path}: {e}")

        try:
            return cls.from_dict(data)
        except Exception as e:
            raise ValueError(
                f"Failed to create UnifiedConfig from data in {config_path}: {e}"
            ) from e

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedConfig":
        """
        辞書からUnifiedConfigを作成

        Args:
            data: 設定データの辞書

        Returns:
            UnifiedConfigインスタンス

        Raises:
            TypeError: dataが辞書でない場合
            ValueError: 必須フィールドが欠けている場合
        """
        if not isinstance(data, dict):
            raise TypeError(f"Config data must be a dictionary, got {type(data)}")

        # 必須フィールドのチェック
        required_fields = ["model_name", "version", "algorithm"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields in config: {missing_fields}")

        # 特徴量設定の処理
        features = data.get("features", {})
        if isinstance(features, dict):
            # ネストされた特徴量設定をフラット化
            processed_features = {}
            for category, feature_list in features.items():
                if isinstance(feature_list, list):
                    processed_features[category] = feature_list
                elif isinstance(feature_list, str):
                    processed_features[category] = [feature_list]
                elif isinstance(feature_list, dict):
                    # 辞書形式の特徴量設定はそのまま保持
                    processed_features[category] = feature_list
                else:
                    # 数値などの他の型も許容（target_dimensionsなど）
                    processed_features[category] = feature_list
        else:
            processed_features = {}

        try:
            return cls(
                model_name=str(data.get("model_name", "unknown")).strip(),
                version=str(data.get("version", "unknown")).strip(),
                algorithm=str(data.get("algorithm", "unknown")).strip().lower(),
                description=str(data.get("description", "")).strip(),
                training=data.get("training", {}),
                environment=data.get("environment", {}),
                features=processed_features,
                reward_settings=data.get("reward_settings", {}),
                ensemble_system=data.get("ensemble_system", {}),
                market_regimes=data.get("market_regimes", {}),
                validation=data.get("validation", {}),
                logging=data.get("logging", {}),
                checkpoint=data.get("checkpoint", {}),
                metadata=data.get("_metadata", {}),
            )
        except Exception as e:
            raise ValueError(f"Failed to create UnifiedConfig from data: {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "algorithm": self.algorithm,
            "description": self.description,
            "training": self.training,
            "environment": self.environment,
            "features": self.features,
            "reward_settings": self.reward_settings,
            "ensemble_system": self.ensemble_system,
            "market_regimes": self.market_regimes,
            "validation": self.validation,
            "logging": self.logging,
            "checkpoint": self.checkpoint,
            "_metadata": self.metadata,
        }

    def save(
        self,
        config_path: Union[str, Path],
        format_type: ConfigFormat = ConfigFormat.JSON,
    ) -> None:
        """設定をファイルに保存"""
        path = Path(config_path)
        data = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            if format_type == ConfigFormat.JSON:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def get_feature_count(self) -> int:
        """特徴量の総数を返す"""
        return sum(len(feature_list) for feature_list in self.features.values())

    def validate(self) -> List[str]:
        """
        設定の妥当性を検証

        Returns:
            エラーメッセージのリスト。空の場合は妥当
        """
        errors = []

        # 必須フィールドのチェック
        required_fields = ["model_name", "version", "algorithm"]
        for field in required_fields:
            value = getattr(self, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"Missing or empty required field: {field}")

        # モデル名の形式チェック
        if (
            self.model_name
            and not self.model_name.replace("_", "").replace("-", "").isalnum()
        ):
            errors.append(
                f"Invalid model_name format: {self.model_name}. Use only alphanumeric, underscore, and hyphen"
            )

        # バージョンの形式チェック
        if self.version and not self._is_valid_version(self.version):
            errors.append(
                f"Invalid version format: {self.version}. Use semantic versioning (e.g., 1.0.0)"
            )

        # アルゴリズムのチェック
        supported_algorithms = ["sac", "ppo", "ddpg", "td3"]
        if self.algorithm and self.algorithm.lower() not in supported_algorithms:
            errors.append(
                f"Unsupported algorithm: {self.algorithm}. Supported: {supported_algorithms}"
            )

        # 特徴量のチェック
        if not self.features:
            errors.append("No features configured")
        else:
            # 各特徴量タイプのチェック
            for feature_type, feature_list in self.features.items():
                if not isinstance(feature_list, list):
                    errors.append(f"Features for {feature_type} must be a list")
                elif not feature_list:
                    errors.append(f"Empty feature list for {feature_type}")
                else:
                    # 特徴量名の形式チェック
                    for feature in feature_list:
                        if not isinstance(feature, str) or not feature.strip():
                            errors.append(
                                f"Invalid feature name in {feature_type}: {feature}"
                            )

        # トレーニング設定のチェック
        if not self.training:
            errors.append("No training configuration")
        else:
            # 必須のトレーニングパラメータチェック
            required_training_fields = ["learning_rate", "batch_size", "buffer_size"]
            for field in required_training_fields:
                if field not in self.training:
                    errors.append(f"Missing training parameter: {field}")
                elif not isinstance(self.training[field], (int, float)):
                    errors.append(f"Training parameter {field} must be numeric")

        # 報酬設定のチェック
        if self.reward_settings:
            # base_action_penaltyのチェック
            if "base_action_penalty" in self.reward_settings:
                penalty = self.reward_settings["base_action_penalty"]
                if not isinstance(penalty, (int, float)) or penalty < 0:
                    errors.append(
                        f"base_action_penalty must be non-negative number, got: {penalty}"
                    )

            # action_bonusesのチェック
            if "action_bonuses" in self.reward_settings:
                bonuses = self.reward_settings["action_bonuses"]
                if not isinstance(bonuses, dict):
                    errors.append("action_bonuses must be a dictionary")
                else:
                    for action, bonus in bonuses.items():
                        if not isinstance(bonus, (int, float)):
                            errors.append(
                                f"action_bonuses[{action}] must be numeric, got: {bonus}"
                            )

        return errors

    def _is_valid_version(self, version: str) -> bool:
        """バージョンの形式を検証"""
        import re

        # セマンティックバージョニングの形式 (例: 1.0.0, 2.1.3-alpha)
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        return bool(re.match(pattern, version))


class UnifiedConfigManager:
    """
    統合設定マネージャー

    複数の設定ソースを統合的に管理
    """

    def __init__(self):
        self.configs: Dict[str, UnifiedConfig] = {}
        self.logger = get_logger(__name__)

    def load_config(self, name: str, config_path: Union[str, Path]) -> UnifiedConfig:
        """設定を読み込み"""
        config = UnifiedConfig.from_file(config_path)
        self.configs[name] = config
        self.logger.info(f"Loaded config '{name}' from {config_path}")
        return config

    def get_config(self, name: str) -> Optional[UnifiedConfig]:
        """設定を取得"""
        return self.configs.get(name)

    def list_configs(self) -> List[str]:
        """設定名の一覧を取得"""
        return list(self.configs.keys())

    def validate_all_configs(self) -> Dict[str, List[str]]:
        """すべての設定を検証"""
        results = {}
        for name, config in self.configs.items():
            errors = config.validate()
            if errors:
                results[name] = errors
        return results
