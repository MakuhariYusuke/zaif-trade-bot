"""
Feature registry and manager for trading features.
特徴量レジストリとマネージャー
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union
import pandas as pd
import networkx as nx
from .base import Feature, ComputableFeature, CommonPreprocessor


class FeatureManager:
    """特徴量マネージャー"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.features: Dict[str, Union[Feature, ComputableFeature]] = {}
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """features.yaml を読み込み"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}

    def register(self, feature: Union[Feature, ComputableFeature]):
        """特徴量を登録"""
        self.features[feature.name] = feature

    def get_enabled_features(self, wave: Optional[int] = None) -> List[str]:
        """有効な特徴量を取得"""
        enabled = []
        for name, config in self.config.get('features', {}).items():
            if config.get('enabled', False) and not config.get('harmful', False):
                if wave is None or config.get('wave', 1) == wave:
                    enabled.append(name)
        return enabled

    def compute_features(self, df: pd.DataFrame, wave: Optional[int] = None) -> pd.DataFrame:
        """特徴量を計算（DAGで依存解決）"""
        # 共通前処理
        df = CommonPreprocessor.preprocess(df)

        # 有効な特徴量を取得
        if wave is None:
            # wave=Noneの場合、configのenabledに基づく
            enabled_features = [name for name, config in self.config.get('features', {}).items() if config.get('enabled', False)]
        else:
            enabled_features = self.get_enabled_features(wave)
        if not enabled_features:
            return df

        # DAG構築
        dag = nx.DiGraph()
        for name in enabled_features:
            if name not in self.features:
                continue
            feature = self.features[name]
            dag.add_node(name)
            for dep in feature.deps:
                dag.add_edge(dep, name)

        # トポロジカルソートで計算順序決定
        try:
            order = list(nx.topological_sort(dag))
        except nx.NetworkXError:
            raise ValueError("Circular dependency detected in features")

        # 計算実行
        computed: Set[str] = set()
        for name in order:
            if name in self.features:
                feature = self.features[name]
                # 依存がすべて計算済みかチェック
                if all(dep in computed or dep in df.columns for dep in feature.deps):
                    params = self.config.get('features', {}).get(name, {}).get('params', {})
                    new_df = feature.compute(df, **params)
                    df = pd.concat([df, new_df], axis=1)
                    computed.add(name)

        return df

    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """特徴量情報を取得"""
        return self.config.get('features', {}).get(name)


# グローバルマネージャーインスタンス
_manager: Optional[FeatureManager] = None

DEFAULT_FEATURE_CONFIG_PATH: str = "config/features.yaml"

def get_feature_manager(config_path: str = DEFAULT_FEATURE_CONFIG_PATH) -> FeatureManager:
    """グローバル特徴量マネージャーを取得"""
    global _manager
    if _manager is None:
        _manager = FeatureManager(config_path)
    return _manager