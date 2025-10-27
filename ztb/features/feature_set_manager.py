"""
特徴量セットの統一管理システム
設定ファイルベースで特徴量を動的に管理し、簡単に増減できるようにする
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ztb.utils.config_loader import ConfigLoader
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureSetType(Enum):
    """特徴量セットの種類"""

    CURATED = "curated"
    FULL = "full"
    MINIMAL = "minimal"
    CUSTOM = "custom"


@dataclass
class FeatureSetConfig:
    """特徴量セットの設定"""

    name: str
    description: str
    features: List[str]
    enabled: bool = True
    version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None


class FeatureSetManager:
    """
    特徴量セットの統一管理マネージャー

    特徴量セットをYAML設定ファイルで管理し、
    動的な追加・削除・有効化・無効化をサポート
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("configs/features/feature_sets.yaml")
        self.feature_sets: Dict[str, FeatureSetConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """設定ファイルを読み込む"""
        if not self.config_path.exists():
            logger.warning(
                f"Feature sets config not found: {self.config_path}, creating default"
            )
            self._create_default_config()
            return

        try:
            config_data = ConfigLoader.load(self.config_path)
            self._parse_config(config_data)
            logger.info(
                f"Loaded {len(self.feature_sets)} feature sets from {self.config_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load feature sets config: {e}")
            self._create_default_config()

    def _parse_config(self, config_data: Dict[str, Any]) -> None:
        """設定データをパース"""
        for set_name, set_config in config_data.get("feature_sets", {}).items():
            if isinstance(set_config, dict):
                self.feature_sets[set_name] = FeatureSetConfig(
                    name=set_name,
                    description=set_config.get("description", ""),
                    features=set_config.get("features", []),
                    enabled=set_config.get("enabled", True),
                    version=set_config.get("version", "1.0"),
                    metadata=set_config.get("metadata", {}),
                )

    def _create_default_config(self) -> None:
        """デフォルト設定ファイルを作成"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # デフォルトの特徴量セット
        default_features = self._get_default_features()

        default_config = {
            "version": "1.0",
            "description": "特徴量セットの統一管理設定",
            "feature_sets": {
                "curated": {
                    "description": "質的に改善された特徴量セット（78個）",
                    "features": default_features,
                    "enabled": True,
                    "version": "1.0",
                    "metadata": {"category": "production", "recommended": True},
                },
                "minimal": {
                    "description": "最小限の特徴量セット（20個）",
                    "features": default_features[:20],
                    "enabled": True,
                    "version": "1.0",
                    "metadata": {"category": "testing", "recommended": False},
                },
                "full": {
                    "description": "全特徴量セット（curatedと同じ）",
                    "features": default_features,
                    "enabled": True,
                    "version": "1.0",
                    "metadata": {"category": "experimental", "recommended": False},
                },
            },
        }

        # YAMLファイルに保存
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created default feature sets config: {self.config_path}")

        # 設定を再読み込み
        self._load_config()

    def _get_default_features(self) -> List[str]:
        """デフォルトの特徴量リストを取得（curated_features.pyから）"""
        # 既存のCURATED_FEATURESをインポート
        try:
            from ztb.features.curated_features import CURATED_FEATURES

            return CURATED_FEATURES
        except ImportError:
            logger.warning("Could not import CURATED_FEATURES, using basic features")
            return [
                "close",
                "open",
                "high",
                "low",
                "volume",
                "RSI",
                "MACD",
                "BB_Position",
                "ATR",
                "ADX",
            ]

    def get_feature_set(self, name: str) -> List[str]:
        """指定された名前の特徴量セットを取得"""
        if name not in self.feature_sets:
            logger.warning(f"Feature set '{name}' not found, using 'curated'")
            name = "curated"

        feature_set = self.feature_sets[name]
        if not feature_set.enabled:
            logger.warning(f"Feature set '{name}' is disabled, using 'curated'")
            curated_set = self.feature_sets.get("curated")
            if curated_set and curated_set.enabled:
                return curated_set.features
            else:
                return self._get_default_features()

        logger.info(
            f"Using feature set '{name}' with {len(feature_set.features)} features"
        )
        return feature_set.features

    def add_feature_set(
        self,
        name: str,
        features: List[str],
        description: str = "",
        enabled: bool = True,
    ) -> bool:
        """新しい特徴量セットを追加"""
        if name in self.feature_sets:
            logger.warning(f"Feature set '{name}' already exists")
            return False

        self.feature_sets[name] = FeatureSetConfig(
            name=name,
            description=description,
            features=features,
            enabled=enabled,
            version="1.0",
        )

        self._save_config()
        logger.info(f"Added feature set '{name}' with {len(features)} features")
        return True

    def update_feature_set(
        self,
        name: str,
        features: Optional[List[str]] = None,
        description: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """特徴量セットを更新"""
        if name not in self.feature_sets:
            logger.warning(f"Feature set '{name}' not found")
            return False

        feature_set = self.feature_sets[name]

        if features is not None:
            feature_set.features = features
        if description is not None:
            feature_set.description = description
        if enabled is not None:
            feature_set.enabled = enabled

        self._save_config()
        logger.info(f"Updated feature set '{name}'")
        return True

    def remove_feature_set(self, name: str) -> bool:
        """特徴量セットを削除"""
        if name not in self.feature_sets:
            logger.warning(f"Feature set '{name}' not found")
            return False

        if name in ["curated", "minimal", "full"]:
            logger.warning(f"Cannot remove built-in feature set '{name}'")
            return False

        del self.feature_sets[name]
        self._save_config()
        logger.info(f"Removed feature set '{name}'")
        return True

    def add_features(self, set_name: str, features: List[str]) -> bool:
        """特徴量セットに特徴量を追加"""
        if set_name not in self.feature_sets:
            logger.warning(f"Feature set '{set_name}' not found")
            return False

        feature_set = self.feature_sets[set_name]
        # 重複を避ける
        existing_features = set(feature_set.features)
        new_features = [f for f in features if f not in existing_features]

        if not new_features:
            logger.info(f"All features already exist in set '{set_name}'")
            return True

        feature_set.features.extend(new_features)
        self._save_config()
        logger.info(f"Added {len(new_features)} features to set '{set_name}'")
        return True

    def remove_features(self, set_name: str, features: List[str]) -> bool:
        """特徴量セットから特徴量を削除"""
        if set_name not in self.feature_sets:
            logger.warning(f"Feature set '{set_name}' not found")
            return False

        feature_set = self.feature_sets[set_name]
        original_count = len(feature_set.features)

        # 指定された特徴量を削除
        features_to_remove = set(features)
        feature_set.features = [
            f for f in feature_set.features if f not in features_to_remove
        ]

        removed_count = original_count - len(feature_set.features)
        if removed_count == 0:
            logger.info(f"No features were removed from set '{set_name}'")
            return True

        self._save_config()
        logger.info(f"Removed {removed_count} features from set '{set_name}'")
        return True

    def list_feature_sets(self) -> Dict[str, Dict[str, Any]]:
        """利用可能な特徴量セットの一覧を取得"""
        return {
            name: {
                "description": fs.description,
                "feature_count": len(fs.features),
                "enabled": fs.enabled,
                "version": fs.version,
            }
            for name, fs in self.feature_sets.items()
        }

    def get_feature_count(self, name: str) -> int:
        """指定された特徴量セットの特徴量数を取得"""
        feature_set = self.feature_sets.get(name)
        return len(feature_set.features) if feature_set else 0

    def _save_config(self) -> None:
        """設定をファイルに保存"""
        config_data = {
            "version": "1.0",
            "description": "特徴量セットの統一管理設定",
            "feature_sets": {},
        }

        for name, feature_set in self.feature_sets.items():
            config_data["feature_sets"][name] = {
                "description": feature_set.description,
                "features": feature_set.features,
                "enabled": feature_set.enabled,
                "version": feature_set.version,
                "metadata": feature_set.metadata or {},
            }

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)


# グローバルインスタンス
_feature_manager: Optional[FeatureSetManager] = None


def get_feature_manager() -> FeatureSetManager:
    """特徴量マネージャーのインスタンスを取得"""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureSetManager()
    return _feature_manager


def get_feature_set(name: str = "curated") -> List[str]:
    """特徴量セットを取得（後方互換性のための関数）"""
    manager = get_feature_manager()
    return manager.get_feature_set(name)


def get_features_to_remove(feature_set_name: str = "curated") -> List[str]:
    """削除すべき特徴量を取得（後方互換性のための関数）"""
    # この関数は現在は使用されないが、後方互換のために残す
    return []
