"""
Unified Configuration System

統合設定管理システム
すべての設定ファイルを統一的に管理し、型安全性を確保
"""

from typing import Any, Dict, List, Optional, Union
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from ztb.utils.logging_utils import get_logger
from ztb.utils.errors import safe_operation

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
    def from_file(cls, config_path: Union[str, Path],
                  format_type: ConfigFormat = ConfigFormat.AUTO) -> 'UnifiedConfig':
        """
        設定ファイルを読み込んでUnifiedConfigを作成

        Args:
            config_path: 設定ファイルのパス
            format_type: ファイル形式

        Returns:
            UnifiedConfigインスタンス
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # 形式の自動判定
        if format_type == ConfigFormat.AUTO:
            if path.suffix.lower() == '.json':
                format_type = ConfigFormat.JSON
            elif path.suffix.lower() in ['.yaml', '.yml']:
                format_type = ConfigFormat.YAML
            else:
                raise ValueError(f"Unsupported file extension: {path.suffix}")

        # ファイル読み込み
        with open(path, 'r', encoding='utf-8') as f:
            if format_type == ConfigFormat.JSON:
                data = json.load(f)
            else:
                data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedConfig':
        """辞書からUnifiedConfigを作成"""
        # 特徴量設定の処理
        features = data.get('features', {})
        if isinstance(features, dict):
            # ネストされた特徴量設定をフラット化
            processed_features = {}
            for category, feature_list in features.items():
                if isinstance(feature_list, list):
                    processed_features[category] = feature_list
                else:
                    processed_features[category] = [feature_list]
        else:
            processed_features = {}

        return cls(
            model_name=data.get('model_name', 'unknown'),
            version=data.get('version', 'unknown'),
            algorithm=data.get('algorithm', 'unknown'),
            description=data.get('description', ''),
            training=data.get('training', {}),
            features=processed_features,
            reward_settings=data.get('reward_settings', {}),
            ensemble_system=data.get('ensemble_system', {}),
            market_regimes=data.get('market_regimes', {}),
            validation=data.get('validation', {}),
            logging=data.get('logging', {}),
            checkpoint=data.get('checkpoint', {}),
            metadata=data.get('_metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'algorithm': self.algorithm,
            'description': self.description,
            'training': self.training,
            'features': self.features,
            'reward_settings': self.reward_settings,
            'ensemble_system': self.ensemble_system,
            'market_regimes': self.market_regimes,
            'validation': self.validation,
            'logging': self.logging,
            'checkpoint': self.checkpoint,
            '_metadata': self.metadata
        }

    def save(self, config_path: Union[str, Path],
             format_type: ConfigFormat = ConfigFormat.JSON) -> None:
        """設定をファイルに保存"""
        path = Path(config_path)
        data = self.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            if format_type == ConfigFormat.JSON:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def get_feature_count(self) -> int:
        """特徴量の総数を返す"""
        return sum(len(feature_list) for feature_list in self.features.values())

    def validate(self) -> List[str]:
        """設定の妥当性を検証"""
        errors = []

        # 必須フィールドのチェック
        required_fields = ['model_name', 'version', 'algorithm']
        for field in required_fields:
            if not getattr(self, field):
                errors.append(f"Missing required field: {field}")

        # 特徴量のチェック
        if not self.features:
            errors.append("No features configured")

        # トレーニング設定のチェック
        if not self.training:
            errors.append("No training configuration")

        return errors


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