"""
Feature Management System - モデルごとの特徴量スキーマ管理

このモジュールは、各モデルに紐付いた特徴量スキーマを管理し、
訓練・推論時の特徴量不一致問題を解決します。

主な機能:
1. モデルごとの特徴量スキーマ保存
2. 推論時の自動スキーマ検出
3. 環境の動的特徴量設定
4. スキーマバージョン管理

使用例:
    # 訓練時
    manager = FeatureSchemaManager(model_name="v384_curated_60")
    manager.save_schema(features=feature_list, config=training_config)

    # 推論時
    manager = FeatureSchemaManager(model_name="v384_curated_60")
    schema = manager.load_schema()
    env = create_env_with_schema(schema)
"""

import asyncio
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ztb.io.json_io import read_json, write_json
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class FeatureSchema:
    """特徴量スキーマのデータクラス"""

    features: list[str]
    config: dict[str, Any]
    scaler_data: dict[str, NDArray[np.floating[Any]]] | None = None

    def __post_init__(self) -> None:
        """バリデーション"""
        if not self.features:
            raise ValueError("features cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "features": self.features,
            "config": self.config,
            "scaler_data": self.scaler_data,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSchema":
        """辞書から復元"""
        return cls(**data)

@dataclass
class FeatureSchemaMetadata:
    """特徴量スキーマのメタデータ"""

    model_name: str
    num_features: int
    feature_names: list[str]
    schema_hash: str
    created_at: str
    training_config: dict[str, Any]
    curated_features_spec: str | None = None
    feature_filtering_enabled: bool = False
    feature_filter_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSchemaMetadata":
        """辞書から復元"""
        return cls(**data)

class FeatureSchemaManager:
    """
    モデルごとの特徴量スキーマを管理するクラス

    ディレクトリ構造:
        models/
        ├── model_name.zip
        └── schemas/
            └── model_name/
                ├── features_schema.json
                ├── scaler.npz
                └── metadata.json
    """

    DEFAULT_MODELS_DIR = Path("models")

    def __init__(
        self,
        model_name: str,
        models_dir: Path | None = None,
        schemas_dir: Path | None = None,
    ):
        """
        Args:
            model_name: モデル名（例: "v384_curated_60"）
            models_dir: モデルディレクトリ
            schemas_dir: スキーマ保存ディレクトリ（Noneの場合はmodels_dir/schemas）
        """
        super().__init__()
        self.model_name = model_name
        self.models_dir = (
            models_dir if models_dir is not None else self.DEFAULT_MODELS_DIR
        )
        self.schemas_dir = schemas_dir or (self.models_dir / "schemas")
        self.model_schema_dir = self.schemas_dir / model_name

        # ディレクトリ作成
        self.model_schema_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FeatureSchemaManager initialized for model: {model_name}")
        logger.info(f"Schema directory: {self.model_schema_dir}")

    def save_schema(
        self,
        features: list[str],
        config: dict[str, Any],
        scaler_data: dict[str, NDArray[np.floating[Any]]] | None = None,
    ) -> str:
        """
        モデルの特徴量スキーマを保存

        Args:
            features: 特徴量名のリスト
            config: 訓練設定
            scaler_data: 正規化パラメータ（mean, std）

        Returns:
            スキーマのハッシュ値
        """
        # ハッシュ計算
        schema_hash = self._compute_hash(features)

        # メタデータ作成
        metadata = FeatureSchemaMetadata(
            model_name=self.model_name,
            num_features=len(features),
            feature_names=features,
            schema_hash=schema_hash,
            created_at=datetime.now().isoformat(),
            training_config=config,
            curated_features_spec=config.get("curated_features_list"),
            feature_filtering_enabled=config.get("enable_feature_filtering", False),
            feature_filter_mode=config.get("feature_filter_mode"),
        )

        # 保存
        self._save_features_schema(features, schema_hash)
        self._save_metadata(metadata)

        if scaler_data is not None:
            self._save_scaler(scaler_data, schema_hash)

        logger.info(f"✅ Saved schema for {self.model_name}")
        logger.info(f"   Features: {len(features)}")
        logger.info(f"   Hash: {schema_hash}")

        return schema_hash

    def load_schema(self) -> FeatureSchemaMetadata:
        """
        モデルの特徴量スキーマを読み込み

        Returns:
            スキーマメタデータ
        """
        metadata_path = self.model_schema_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Schema metadata not found for model {self.model_name}. "
                f"Expected at: {metadata_path}"
            )

        data = read_json(metadata_path)

        metadata = FeatureSchemaMetadata.from_dict(data)

        logger.info(f"📖 Loaded schema for {self.model_name}")
        logger.info(f"   Features: {metadata.num_features}")
        logger.info(f"   Hash: {metadata.schema_hash}")

        return metadata

    def load_scaler(self) -> dict[str, NDArray[np.floating[Any]]] | None:
        """正規化パラメータを読み込み"""
        scaler_path = self.model_schema_dir / "scaler.npz"

        if not scaler_path.exists():
            logger.warning(f"Scaler not found: {scaler_path}")
            return None

        data = np.load(scaler_path)
        return {
            "mean": data["mean"],
            "std": data["std"],
        }

    def get_feature_list(self) -> list[str]:
        """特徴量リストを取得"""
        metadata = self.load_schema()
        return metadata.feature_names

    def get_num_features(self) -> int:
        """特徴量数を取得"""
        metadata = self.load_schema()
        return metadata.num_features

    async def load_schema_async(self) -> FeatureSchemaMetadata | None:
        """特徴量スキーマを非同期で読み込み"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_schema)

    async def save_schema_async(
        self,
        features: list[str],
        config: dict[str, Any],
        scaler_data: dict[str, NDArray[np.floating[Any]]] | None = None,
    ) -> str:
        """モデルの特徴量スキーマを非同期で保存"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.save_schema, features, config, scaler_data
        )

    def verify_compatibility(self, other_model_name: str) -> bool:
        """
        別のモデルとの互換性をチェック

        Args:
            other_model_name: 比較対象のモデル名

        Returns:
            互換性があればTrue
        """
        try:
            other_manager = FeatureSchemaManager(other_model_name, self.models_dir)
            other_metadata = other_manager.load_schema()
            my_metadata = self.load_schema()

            compatible = (
                my_metadata.num_features == other_metadata.num_features
                and my_metadata.feature_names == other_metadata.feature_names
            )

            if compatible:
                logger.info(
                    f"✅ {self.model_name} and {other_model_name} are compatible"
                )
            else:
                logger.warning(
                    f"⚠️  {self.model_name} and {other_model_name} are NOT compatible"
                )
                logger.warning(
                    f"   {self.model_name}: {my_metadata.num_features} features"
                )
                logger.warning(
                    f"   {other_model_name}: {other_metadata.num_features} features"
                )

            return compatible

        except FileNotFoundError:
            logger.error(
                f"Cannot verify compatibility: {other_model_name} schema not found"
            )
            return False

    def _compute_hash(self, features: list[str]) -> str:
        """特徴量リストのハッシュを計算"""
        features_str = ",".join(sorted(features))
        return hashlib.sha256(features_str.encode()).hexdigest()[:16]

    def _save_features_schema(self, features: list[str], schema_hash: str) -> None:
        """features_schema.jsonを保存"""
        schema_path = self.model_schema_dir / "features_schema.json"
        write_json(schema_path, features, indent=2, ensure_ascii=False)
        logger.debug(f"Saved features schema: {schema_path}")

    def _save_metadata(self, metadata: FeatureSchemaMetadata) -> None:
        """metadata.jsonを保存"""
        metadata_path = self.model_schema_dir / "metadata.json"
        write_json(metadata_path, metadata.to_dict(), indent=2, ensure_ascii=False)
        logger.debug(f"Saved metadata: {metadata_path}")

    def _save_scaler(
        self, scaler_data: dict[str, NDArray[np.floating[Any]]], schema_hash: str
    ) -> None:
        """scaler.npzを保存"""
        scaler_path = self.model_schema_dir / "scaler.npz"
        np.savez(scaler_path, mean=scaler_data["mean"], std=scaler_data["std"])
        logger.debug(f"Saved scaler: {scaler_path}")

    @staticmethod
    def list_all_schemas(models_dir: Path | None = None) -> list[str]:
        """利用可能なすべてのスキーマをリスト"""
        if models_dir is None:
            models_dir = FeatureSchemaManager.DEFAULT_MODELS_DIR
        schemas_dir = models_dir / "schemas"
        if not schemas_dir.exists():
            return []

        schemas = []
        for model_dir in schemas_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "metadata.json").exists():
                schemas.append(model_dir.name)

        return sorted(schemas)

    @staticmethod
    def print_schema_summary(models_dir: Path | None = None) -> None:
        """全スキーマのサマリーを表示"""
        if models_dir is None:
            models_dir = FeatureSchemaManager.DEFAULT_MODELS_DIR
        schemas = FeatureSchemaManager.list_all_schemas(models_dir)

        if not schemas:
            logger.info("No schemas found")
            return

        logger.info("=" * 80)
        logger.info("Available Feature Schemas")
        logger.info("=" * 80)

        for schema_name in schemas:
            manager = FeatureSchemaManager(schema_name, models_dir)
            try:
                metadata = manager.load_schema()
                logger.info(f"\n📦 {schema_name}")
                logger.info(f"   Features: {metadata.num_features}")
                logger.info(f"   Hash: {metadata.schema_hash}")
                logger.info(f"   Created: {metadata.created_at}")
                if metadata.curated_features_spec:
                    logger.info(f"   Curated: {metadata.curated_features_spec}")
            except Exception as e:
                logger.error(f"   Error loading: {e}")

        logger.info("=" * 80)

def migrate_legacy_schema(
    model_name: str,
    legacy_schema_path: Path | None = None,
    legacy_scaler_path: Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    レガシーのグローバルスキーマをモデル固有スキーマに移行

    Args:
        model_name: モデル名
        legacy_schema_path: 旧features_schema.jsonのパス
        legacy_scaler_path: 旧scaler.npzのパス
        config: 訓練設定（可能な範囲で）
    """
    logger.info(f"Migrating legacy schema to {model_name}...")

    # set default paths if not provided
    if legacy_schema_path is None:
        legacy_schema_path = (
            FeatureSchemaManager.DEFAULT_MODELS_DIR / "features_schema.json"
        )
    if legacy_scaler_path is None:
        legacy_scaler_path = FeatureSchemaManager.DEFAULT_MODELS_DIR / "scaler.npz"

    # レガシースキーマ読み込み
    if not legacy_schema_path.exists():
        raise FileNotFoundError(f"Legacy schema not found: {legacy_schema_path}")

    schema_data = read_json(legacy_schema_path)

    # features_schema.jsonからcolumnsを抽出
    if isinstance(schema_data, dict) and "columns" in schema_data:
        features = schema_data["columns"]
    elif isinstance(schema_data, list):
        features = schema_data
    else:
        raise ValueError(f"Invalid schema format in {legacy_schema_path}")

    # スケーラー読み込み
    scaler_data = None
    if legacy_scaler_path.exists():
        data = np.load(legacy_scaler_path)
        scaler_data = {
            "mean": data["mean"],
            "std": data["std"],
        }

    # 新システムで保存
    manager = FeatureSchemaManager(model_name)
    manager.save_schema(features=features, config=config or {}, scaler_data=scaler_data)

    logger.info(f"✅ Migration completed for {model_name}")
