"""
特徴量相関分析システム
既存の相関分析機能を活用して特徴量間の相関を調査・可視化
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer

logger = logging.getLogger(__name__)


class FeatureCorrelationAnalyzer:
    """
    特徴量相関分析クラス
    既存の相関分析機能を拡張して特徴量間の関係を分析
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        初期化

        Args:
            data_path: データファイルのパス（オプション）
        """
        self.data_path = data_path or "results/sac_v434_1_results.json"
        try:
            self.analyzer = BacktestAnalyzer(self.data_path)
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"BacktestAnalyzer初期化に失敗、分析機能を無効化: {e}")
            self.analyzer = None
        self.correlation_data = {}
        self.feature_data = {}

    def load_feature_data(self) -> bool:
        """
        特徴量データを読み込み

        Returns:
            読み込み成功フラグ
        """
        try:
            if self.analyzer is None:
                logger.warning(
                    "BacktestAnalyzerが初期化されていないため、特徴量データを読み込めません"
                )
                return False

            if not Path(self.data_path).exists():
                logger.warning(f"データファイルが見つかりません: {self.data_path}")
                return False

            # 既存の分析データを活用
            self.analyzer.load_data(self.data_path)
            self.correlation_data = self.analyzer.analyze_correlation_and_dependencies()

            logger.info(f"特徴量データを読み込みました: {self.data_path}")
            return True

        except Exception as e:
            logger.error(f"特徴量データの読み込みに失敗: {e}")
            return False

    def analyze_feature_correlations(
        self, feature_matrix: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        特徴量間の相関分析

        Args:
            feature_matrix: 特徴量行列 [samples, features]
            feature_names: 特徴量名リスト

        Returns:
            相関分析結果
        """
        try:
            # 標準化
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)

            # ピアソン相関
            pearson_corr = np.corrcoef(normalized_features.T)

            # スピアマン相関（非線形関係）
            spearman_corr = np.zeros_like(pearson_corr)
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    if i != j:
                        corr, _ = stats.spearmanr(
                            normalized_features[:, i], normalized_features[:, j]
                        )
                        spearman_corr[i, j] = corr

            # 相互情報量（非線形依存）
            mi_matrix = np.zeros((len(feature_names), len(feature_names)))
            for i in range(len(feature_names)):
                for j in range(len(feature_names)):
                    if i != j:
                        mi = mutual_info_regression(
                            normalized_features[:, [i]], normalized_features[:, j]
                        )[0]
                        mi_matrix[i, j] = mi

            # 相関の強度分析
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    pearson_val = abs(pearson_corr[i, j])
                    spearman_val = abs(spearman_corr[i, j])
                    mi_val = mi_matrix[i, j]

                    if pearson_val > 0.7 or spearman_val > 0.7 or mi_val > 0.5:
                        high_corr_pairs.append(
                            {
                                "feature1": feature_names[i],
                                "feature2": feature_names[j],
                                "pearson": pearson_corr[i, j],
                                "spearman": spearman_corr[i, j],
                                "mutual_info": mi_val,
                            }
                        )

            # 階層的クラスタリング
            linkage_matrix = linkage(pearson_corr, method="ward")

            return {
                "pearson_correlation": pearson_corr,
                "spearman_correlation": spearman_corr,
                "mutual_information": mi_matrix,
                "high_correlation_pairs": high_corr_pairs,
                "linkage_matrix": linkage_matrix,
                "feature_names": feature_names,
            }

        except Exception as e:
            logger.error(f"特徴量相関分析に失敗: {e}")
            return {}

    def create_correlation_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: str = "reports/feature_correlation_report.txt",
    ) -> None:
        """
        相関分析レポート作成

        Args:
            analysis_results: 分析結果
            output_path: 出力ファイルパス
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=== 特徴量相関分析レポート ===\n\n")

                # 高相関ペアのレポート
                high_corr_pairs = analysis_results.get("high_correlation_pairs", [])
                if high_corr_pairs:
                    f.write("高相関特徴量ペア:\n")
                    for pair in sorted(
                        high_corr_pairs, key=lambda x: abs(x["pearson"]), reverse=True
                    )[:10]:
                        f.write(f"  {pair['feature1']} ↔ {pair['feature2']}\n")
                        f.write(f"    Pearson: {pair['pearson']:.3f}\n")
                        f.write(f"    Spearman: {pair['spearman']:.3f}\n")
                        f.write(f"    Mutual Info: {pair['mutual_info']:.3f}\n")
                        f.write("\n")
                else:
                    f.write("高相関特徴量ペアは見つかりませんでした。\n")

                f.write("\n")

                # 特徴量の統計情報
                feature_names = analysis_results.get("feature_names", [])
                pearson_corr = analysis_results.get("pearson_correlation", np.array([]))

                if len(pearson_corr) > 0:
                    f.write("特徴量ごとの平均相関:\n")
                    for i, name in enumerate(feature_names):
                        avg_corr = np.mean(np.abs(pearson_corr[i, :]))
                        f.write(f"  {name}: {avg_corr:.3f}\n")

            logger.info(f"相関分析レポートを作成しました: {output_path}")

        except Exception as e:
            logger.error(f"レポート作成に失敗: {e}")

    def visualize_correlations(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str = "reports/correlation_plots",
    ) -> None:
        """
        相関関係の可視化

        Args:
            analysis_results: 分析結果
            output_dir: 出力ディレクトリ
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            feature_names = analysis_results.get("feature_names", [])
            pearson_corr = analysis_results.get("pearson_correlation", np.array([]))

            if len(pearson_corr) == 0 or len(feature_names) == 0:
                logger.warning("可視化データが不足しています")
                return

            # 相関ヒートマップ
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
            sns.heatmap(
                pearson_corr,
                mask=mask,
                annot=False,
                cmap="coolwarm",
                xticklabels=feature_names,
                yticklabels=feature_names,
                center=0,
                square=True,
            )
            plt.title("特徴量ピアソン相関ヒートマップ")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/pearson_correlation_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # デンドログラム
            linkage_matrix = analysis_results.get("linkage_matrix")
            if linkage_matrix is not None:
                plt.figure(figsize=(10, 8))
                dendrogram(linkage_matrix, labels=feature_names, orientation="right")
                plt.title("特徴量階層的クラスタリング")
                plt.xlabel("ユークリッド距離")
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/feature_dendrogram.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

            logger.info(f"相関可視化グラフを作成しました: {output_dir}")

        except Exception as e:
            logger.error(f"可視化に失敗: {e}")

    def analyze_feature_importance_correlation(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        target_returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        特徴量とリターンの相関分析（特徴量重要度）

        Args:
            feature_matrix: 特徴量行列
            feature_names: 特徴量名リスト
            target_returns: 目的変数（リターン）

        Returns:
            重要度分析結果
        """
        try:
            importance_results = {}

            # 各特徴量とリターンの相関
            for i, name in enumerate(feature_names):
                feature_values = feature_matrix[:, i]

                # ピアソン相関
                pearson_corr, pearson_p = stats.pearsonr(feature_values, target_returns)

                # スピアマン相関
                spearman_corr, spearman_p = stats.spearmanr(
                    feature_values, target_returns
                )

                # 相互情報量
                mi = mutual_info_regression(
                    feature_values.reshape(-1, 1), target_returns
                )[0]

                importance_results[name] = {
                    "pearson_correlation": pearson_corr,
                    "pearson_p_value": pearson_p,
                    "spearman_correlation": spearman_corr,
                    "spearman_p_value": spearman_p,
                    "mutual_information": mi,
                }

            # 重要度ランキング
            sorted_features = sorted(
                importance_results.items(),
                key=lambda x: abs(x[1]["pearson_correlation"]),
                reverse=True,
            )

            return {
                "feature_importance": importance_results,
                "importance_ranking": sorted_features,
            }

        except Exception as e:
            logger.error(f"特徴量重要度分析に失敗: {e}")
            return {}


def create_sample_feature_analysis():
    """
    サンプル特徴量分析の実行例
    """
    analyzer = FeatureCorrelationAnalyzer()

    # サンプル特徴量データ（実際のデータに置き換え）
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # ランダム特徴量生成（実際には実データを使用）
    feature_matrix = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # ターゲットリターン生成（特徴量との相関を付与）
    target_returns = (
        feature_matrix[:, 0] * 0.5
        + feature_matrix[:, 1] * 0.3
        + np.random.randn(n_samples) * 0.2
    )

    # 相関分析
    correlation_results = analyzer.analyze_feature_correlations(
        feature_matrix, feature_names
    )

    # 重要度分析
    importance_results = analyzer.analyze_feature_importance_correlation(
        feature_matrix, feature_names, target_returns
    )

    # レポート作成
    analyzer.create_correlation_report(correlation_results)

    # 可視化
    analyzer.visualize_correlations(correlation_results)

    print("特徴量相関分析が完了しました")
    print(f"高相関ペア数: {len(correlation_results.get('high_correlation_pairs', []))}")

    # 重要度ランキング表示
    ranking = importance_results.get("importance_ranking", [])
    print("\n特徴量重要度ランキング（上位5位）:")
    for i, (name, metrics) in enumerate(ranking[:5]):
        print(f"  {i+1}. {name}: {metrics['importance']:.3f}")


if __name__ == "__main__":
    create_sample_feature_analysis()
