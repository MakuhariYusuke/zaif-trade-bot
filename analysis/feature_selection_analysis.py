#!/usr/bin/env python3
"""
Feature Selection and Overfitting Prevention Analysis
特徴量選択と過学習防止のための分析スクリプト

SELL bias対策で追加された特徴量の影響を評価し、
過学習リスクを軽減するための特徴量削減を実施
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """
    特徴量選択と過学習防止クラス
    """

    def __init__(self, config_path: str = "configs/features/feature_sets.yaml"):
        self.config_path = Path(config_path)
        self.feature_sets = self._load_feature_sets()

    def _load_feature_sets(self) -> Dict:
        """特徴量セット設定を読み込み"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def analyze_feature_correlations(
        self, data: pd.DataFrame, threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """
        特徴量間の相関分析を行い、高相関特徴量を特定

        Args:
            data: 特徴量データ
            threshold: 相関係数の閾値

        Returns:
            相関グループの辞書
        """
        # 数値データのみを対象
        numeric_data = data.select_dtypes(include=[np.number])

        # 相関係数行列
        corr_matrix = numeric_data.corr().abs()

        # 高相関ペアの特定
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )

        # 相関グループの作成
        correlation_groups = {}
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 not in correlation_groups:
                correlation_groups[feat1] = []
            if feat2 not in correlation_groups:
                correlation_groups[feat2] = []

            correlation_groups[feat1].append(feat2)
            correlation_groups[feat2].append(feat1)

        logger.info(
            f"Found {len(high_corr_pairs)} high correlation pairs (>{threshold})"
        )
        return correlation_groups

    def select_features_by_importance(
        self, X: pd.DataFrame, y: pd.Series, top_k: int = 50
    ) -> List[str]:
        """
        ランダムフォレストによる特徴量重要度ランキング

        Args:
            X: 特徴量データ
            y: ターゲット（アクションラベル）
            top_k: 選択する特徴量数

        Returns:
            選択された特徴量名のリスト
        """
        # ランダムフォレストで特徴量重要度を計算
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # 重要度ランキング
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        # 上位特徴量の選択
        selected_features = feature_importance.head(top_k)["feature"].tolist()

        logger.info(f"Selected top {top_k} features by importance")
        logger.info(f"Top 5 features: {selected_features[:5]}")

        return selected_features

    def apply_pca_reduction(
        self, X: pd.DataFrame, variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, PCA]:
        """
        PCAによる次元削減

        Args:
            X: 特徴量データ
            variance_threshold: 説明される分散の割合

        Returns:
            PCA適用後のデータとPCAオブジェクト
        """
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_scaled)

        # DataFrameに変換
        pca_columns = [f"PC_{i+1}" for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA: {X.shape[1]} features -> {X_pca.shape[1]} components")
        logger.info(f"Explained variance: {explained_variance:.3f}")

        return X_pca_df, pca

    def create_reduced_feature_set(
        self,
        original_features: List[str],
        selected_features: List[str],
        output_path: str = "configs/features/reduced_feature_set.yaml",
    ) -> None:
        """
        削減された特徴量セットを作成

        Args:
            original_features: 元の特徴量リスト
            selected_features: 選択された特徴量リスト
            output_path: 出力ファイルパス
        """
        reduced_config = {
            "description": "過学習防止のための削減特徴量セット",
            "feature_sets": {
                "reduced": {
                    "description": f"重要度ベースで選択された{len(selected_features)}個の特徴量",
                    "enabled": True,
                    "features": selected_features,
                },
                "original_curated": {
                    "description": f"元の{len(original_features)}個の特徴量セット",
                    "enabled": False,
                    "features": original_features,
                },
            },
        }

        # ファイル出力
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(reduced_config, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"Reduced feature set saved to {output_path}")

    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: str = "analysis/feature_importance.png",
    ) -> None:
        """
        特徴量重要度の可視化

        Args:
            X: 特徴量データ
            y: ターゲット
            save_path: 保存パス
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # 重要度の上位20個をプロット
        feature_importance = (
            pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance, x="importance", y="feature")
        plt.title("Top 20 Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved to {save_path}")


def main():
    """メイン実行関数"""
    selector = FeatureSelector()

    # サンプルデータの生成（実際のデータに置き換え）
    np.random.seed(42)
    n_samples = 1000
    n_features = 78  # curatedセットの特徴量数

    # 特徴量データのシミュレーション
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # アクションラベルのシミュレーション（SELL biasを考慮）
    y = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])  # HOLD, BUY, SELL

    logger.info("Starting feature selection analysis...")

    # 1. 相関分析
    correlation_groups = selector.analyze_feature_correlations(X)

    # 2. 特徴量重要度分析
    selected_features = selector.select_features_by_importance(X, y, top_k=50)

    # 3. PCA分析
    X_pca, pca = selector.apply_pca_reduction(X)

    # 4. 削減特徴量セットの作成
    original_features = X.columns.tolist()
    selector.create_reduced_feature_set(original_features, selected_features)

    # 5. 可視化
    selector.plot_feature_importance(X, y)

    logger.info("Feature selection analysis completed")
    logger.info(f"Original features: {len(original_features)}")
    logger.info(f"Selected features: {len(selected_features)}")
    logger.info(f"PCA components: {X_pca.shape[1]}")


if __name__ == "__main__":
    main()
