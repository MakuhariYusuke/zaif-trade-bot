#!/usr/bin/env python3
"""
Enhanced Feature Analysis and Selection System

このスクリプトは既存のfeature_analysis.pyを拡張し、以下を提供:
1. Harmful特徴量の自動判定（NaN率、分散、外れ値）
2. 相関ベースのインテリジェント特徴量選択
3. 重要度ベースの特徴量ランキング
4. メモリ最適化版の特徴量セット提案

使用方法:
    python analyze_feature_selection.py --data ml-dataset-enhanced-balanced.csv --target-features 60

出力:
    - reports/feature_selection_YYYYMMDD_HHMMSS.json: 詳細分析レポート
    - reports/recommended_features.txt: 推奨特徴量リスト
    - reports/harmful_features.txt: 削除すべき有害特徴量
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Import existing utilities
from ztb.preprocessing.feature_correlation_filter import FeatureCorrelationProcessor
from ztb.io.data_loader import DataLoader

class EnhancedFeatureAnalyzer:
    """拡張特徴量分析システム"""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        target_column: str = "win",
        data_path: str | None = None,
    ):
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = DataLoader.load_csv_optimized(data_path)
        else:
            raise ValueError("Either df or data_path must be provided")

        self.target_column = target_column
        self.features = self._identify_features()
        self.analysis_results: dict[str, Any] = {}

    def identify_harmful_features(
        self,
        nan_threshold: float = 0.10,  # 10% NaN率以上
        variance_threshold: float = 1e-10,  # 分散がほぼゼロ
        outlier_threshold: float = 0.30,  # 30% 外れ値以上
        correlation_threshold: float = 0.95,  # 過度な相関
        zero_value_threshold: float = 0.80,  # 80% ゼロ値以上
    ) -> dict[str, dict[str, Any]]:
        """
        拡張Harmful特徴量判定システム

        Harmfulの定義 (拡張版):
        1. NaN率が高い (>10%)
        2. 分散がほぼゼロ (定数特徴量)
        3. 外れ値が異常に多い (>30%)
        4. ゼロ値が過度に多い (>80%)
        5. 他の特徴量と過度に相関 (>95%)
        6. SAC v427特徴量特化判定 (市場適応性の欠如)
        """
        harmful_features = {}

        # 特徴量の基本統計を事前計算
        feature_stats = self._calculate_feature_statistics()

        for feature in self.features:
            data = self.df[feature]
            issues = []
            severity_score = 0

            # 1. NaN率チェック
            nan_rate = data.isnull().sum() / len(data)
            if nan_rate > nan_threshold:
                issues.append(f"high_nan_rate:{nan_rate:.2%}")
                severity_score += 3

            # 2. 分散チェック
            data_clean = data.dropna()
            if len(data_clean) > 0:
                variance = data_clean.var()
                if variance < variance_threshold:
                    issues.append(f"low_variance:{variance:.2e}")
                    severity_score += 3

            # 3. 外れ値チェック（IQR法 + Z-score）
            if len(data_clean) > 0:
                # IQR法
                Q1 = data_clean.quantile(0.25)
                Q3 = data_clean.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_rate = (
                        (data_clean < lower_bound) | (data_clean > upper_bound)
                    ).sum() / len(data_clean)
                    if outlier_rate > outlier_threshold:
                        issues.append(f"high_outlier_rate:{outlier_rate:.2%}")
                        severity_score += 2

                # Z-score法 (追加)
                z_scores = abs((data_clean - data_clean.mean()) / data_clean.std())
                extreme_outlier_rate = (z_scores > 3).sum() / len(data_clean)
                if extreme_outlier_rate > 0.05:  # 5% 極端外れ値
                    issues.append(f"extreme_outliers:{extreme_outlier_rate:.2%}")
                    severity_score += 1

            # 4. ゼロ値チェック
            if len(data_clean) > 0:
                zero_rate = (data_clean == 0).sum() / len(data_clean)
                if zero_rate > zero_value_threshold:
                    issues.append(f"excessive_zeros:{zero_rate:.2%}")
                    severity_score += 2

            # 5. 相関チェック (他の特徴量との過度な相関)
            if feature in feature_stats:
                high_corr_count = sum(
                    1
                    for corr in feature_stats[feature]["correlations"].values()
                    if abs(corr) > correlation_threshold
                )
                if high_corr_count > 0:
                    issues.append(f"over_correlated:{high_corr_count}_features")
                    severity_score += 1

            # 6. SAC v427特徴量特化判定
            sac_issues = self._check_sac_v427_specific_issues(feature, data_clean)
            if sac_issues:
                issues.extend(sac_issues)
                severity_score += len(sac_issues)

            # 判定結果
            if issues:
                severity = (
                    "critical"
                    if severity_score >= 5
                    else "moderate"
                    if severity_score >= 3
                    else "minor"
                )

                harmful_features[feature] = {
                    "issues": issues,
                    "nan_rate": float(nan_rate),
                    "variance": float(variance) if len(data_clean) > 0 else 0.0,
                    "zero_rate": float(zero_rate) if len(data_clean) > 0 else 0.0,
                    "severity": severity,
                    "severity_score": severity_score,
                    "recommendation": self._generate_removal_recommendation(
                        issues, severity
                    ),
                }

        return harmful_features

    def _calculate_feature_statistics(self) -> dict[str, dict[str, Any]]:
        """特徴量の統計情報を計算（相関など）"""
        stats = {}

        # 相関行列を計算（メモリ効率的に）
        numeric_features = [
            f
            for f in self.features
            if self.df[f].dtype in ["float64", "float32", "int64", "int32"]
        ]
        if len(numeric_features) > 1:
            corr_matrix = self.df[numeric_features].corr()

            for feature in numeric_features:
                if feature in corr_matrix.columns:
                    correlations = corr_matrix[feature].drop(feature).to_dict()
                    stats[feature] = {
                        "correlations": correlations,
                        "mean_corr": abs(corr_matrix[feature].drop(feature)).mean(),
                        "max_corr": abs(corr_matrix[feature].drop(feature)).max(),
                    }

        return stats

    def _check_sac_v427_specific_issues(
        self, feature: str, data: pd.Series
    ) -> list[str]:
        """SAC v427特徴量特化の判定"""
        issues = []

        # 市場レジーム特徴量のチェック
        if "regime" in feature.lower():
            # レジーム特徴量は通常0-1の範囲であるべき
            if data.min() < 0 or data.max() > 1:
                issues.append("regime_out_of_bounds")

        # 相関特徴量のチェック
        if "correlation" in feature.lower():
            # 相関は通常-1から1の範囲
            if data.min() < -1.1 or data.max() > 1.1:  # 許容誤差
                issues.append("correlation_out_of_bounds")

        # 技術指標のチェック
        if any(
            indicator in feature.lower()
            for indicator in ["rsi", "macd", "bb", "sma", "ema"]
        ):
            # 技術指標の異常値チェック
            if data.std() > data.abs().mean() * 10:  # 標準偏差が平均の10倍以上
                issues.append("technical_indicator_unstable")

        # ノイズチェック（高頻度変動）
        if len(data) > 100:
            # 連続した値の変化が激しい場合
            diff_std = data.diff().std()
            if diff_std > data.std() * 2:
                issues.append("excessive_noise")

        return issues

    def _generate_removal_recommendation(self, issues: list[str], severity: str) -> str:
        """削除推奨理由を生成"""
        if severity == "critical":
            return "即時削除推奨 - 学習に悪影響を及ぼす可能性が高い"
        elif severity == "moderate":
            return "検討推奨 - 品質改善のために削除を検討"
        else:
            return "注意 - 状況に応じて削除を検討"

    def select_by_correlation(
        self,
        correlation_threshold: float = 0.90,
        importance_dict: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """
        相関ベースのインテリジェント特徴量選択

        高相関ペアがある場合、重要度が低い方を削除
        重要度情報がない場合は、分散が大きい方を保持
        """
        processor = FeatureCorrelationProcessor()
        corr_matrix = processor.analyze_correlations(self.df, self.features)

        removed_features = []
        kept_features = set(self.features)
        correlation_pairs = []

        for i in range(len(self.features)):
            for j in range(i + 1, len(self.features)):
        for i, feature in enumerate(self.features):
            importance_dict[feature] = {
                "importance_mean": float(perm_importance.importances_mean[i]),
                "importance_std": float(perm_importance.importances_std[i]),
            }

        return importance_dict

    def suggest_optimal_features(
        self, target_count: int = 60, remove_harmful: bool = True
    ) -> dict[str, Any]:
        """
        最適な特徴量セットを提案

        手順:
        1. Harmful特徴量を除外
        2. 高相関特徴量を削減
        3. 重要度でランキング
        4. 目標数まで選択
        """
        print(f"📊 Starting feature selection (target: {target_count} features)")

        # Step 1: Harmful特徴量の識別
        print("1️⃣ Identifying harmful features...")
        harmful_features = self.identify_harmful_features()
        print(f"   Found {len(harmful_features)} harmful features")

        # Step 2: 重要度計算
        print("2️⃣ Calculating feature importance...")
        importance_dict = self.calculate_feature_importance()
        print(f"   Calculated importance for {len(importance_dict)} features")

        # Step 3: 相関ベース削減
        print("3️⃣ Removing highly correlated features...")
        kept_after_correlation, removed_by_correlation = self.select_by_correlation(
            correlation_threshold=0.90, importance_dict=importance_dict
        )
        print(f"   Removed {len(removed_by_correlation)} correlated features")

        # Step 4: Harmfulを除外
        if remove_harmful:
            print("4️⃣ Removing harmful features...")
            kept_after_harmful = [
                f for f in kept_after_correlation if f not in harmful_features
            ]
            print(
                f"   Removed {len(kept_after_correlation) - len(kept_after_harmful)} harmful features"
            )
        else:
            kept_after_harmful = kept_after_correlation

        # Step 5: 重要度でソート
        print("5️⃣ Ranking by importance...")
        feature_scores = {
            f: importance_dict[f]["importance_mean"] for f in kept_after_harmful
        }
        sorted_features = sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Step 6: 目標数まで選択
        print(f"6️⃣ Selecting top {target_count} features...")
        if len(sorted_features) > target_count:
            selected_features = [f for f, _ in sorted_features[:target_count]]
            rejected_features = [f for f, _ in sorted_features[target_count:]]
        else:
            selected_features = [f for f, _ in sorted_features]
            rejected_features = []

        print("🔍 Performing detailed feature quality analysis...")

        # Harmful特徴量の識別
        harmful_features = self.identify_harmful_features()

        # 品質スコアの計算
        quality_scores = self._calculate_quality_scores()

        # 特徴量の分類
        categories = self._categorize_features_by_quality(
            harmful_features, quality_scores
        )

        # SAC v427特化の分析
        sac_analysis = self._analyze_sac_v427_feature_quality()

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(self.features),
            "harmful_features": harmful_features,
            "quality_scores": quality_scores,
            "categories": categories,
            "sac_v427_analysis": sac_analysis,
            "recommendations": self._generate_quality_recommendations(
                harmful_features, categories
            ),
        }
            "regime_features": [],
            "correlation_features": [],
            "ensemble_features": [],
            "technical_features": [],
            "market_features": [],
        }

        for feature in self.features:
            feature_lower = feature.lower()

            if "regime" in feature_lower:
                analysis["regime_features"].append(feature)
            elif "correlation" in feature_lower or "beta" in feature_lower:
                analysis["correlation_features"].append(feature)
            elif "ensemble" in feature_lower:
                analysis["ensemble_features"].append(feature)
            elif any(
                ind in feature_lower for ind in ["rsi", "macd", "bb", "sma", "ema"]
            ):
                analysis["technical_features"].append(feature)
            elif any(
                mkt in feature_lower
                for mkt in ["volume", "microstructure", "volatility"]
            ):
                analysis["market_features"].append(feature)

        # カテゴリごとの統計
        category_stats = {}
        for category, features in analysis.items():
            if features:
                scores = [
                    self._calculate_quality_scores().get(f, 0.0) for f in features
                ]
                category_stats[category] = {
                    "count": len(features),
                    "avg_quality": sum(scores) / len(scores),
                    "excellent_count": sum(1 for s in scores if s >= 0.8),
                    "poor_count": sum(1 for s in scores if s < 0.4),
                }

        analysis["category_stats"] = category_stats
        return analysis

    def _generate_quality_recommendations(
        self, harmful_features: dict, categories: dict
    ) -> list[str]:
        """品質分析に基づく推奨事項を生成"""
        recommendations = []

        # Harmful特徴量の推奨
        if harmful_features:
            critical_count = sum(
                1 for h in harmful_features.values() if h["severity"] == "critical"
            )
            recommendations.append(
                f"⚠️  {len(harmful_features)}個のharmful特徴量を検出（うち{critical_count}個がcritical）"
            )
            recommendations.append("   → 即時削除を推奨")

        # 品質カテゴリの推奨
        poor_count = len(categories["poor"])
        if poor_count > 0:
            recommendations.append(f"📉 {poor_count}個のpoor品質特徴量を検出")
            recommendations.append("   → 品質改善または削除を検討")

        # SAC v427特化の推奨
        sac_analysis = self._analyze_sac_v427_feature_quality()
        for category, stats in sac_analysis.get("category_stats", {}).items():
            if stats["poor_count"] > stats["count"] * 0.3:  # 30%以上がpoor
                recommendations.append(f"🔧 {category}カテゴリの品質改善が必要")

        return recommendations

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="拡張特徴量分析システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python analyze_feature_selection.py --data ml-dataset-enhanced-balanced.csv --target-features 60
  python analyze_feature_selection.py --data data.csv --analyze-quality --output-dir reports/
        """,
    )

    parser.add_argument(
        "--data", type=str, required=True, help="分析対象のデータファイルパス"
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="win",
        help="ターゲット列名 (デフォルト: win)",
    )

    parser.add_argument(
        "--target-features", type=int, default=60, help="目標特徴量数 (デフォルト: 60)"
    )

    parser.add_argument(
        "--analyze-quality", action="store_true", help="詳細品質分析を実行"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="出力ディレクトリ (デフォルト: reports)",
    )

    parser.add_argument(
        "--nan-threshold", type=float, default=0.10, help="NaN率閾値 (デフォルト: 0.10)"
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="相関閾値 (デフォルト: 0.90)",
    )

    args = parser.parse_args()

    try:
        # データ読み込み
        print(f"📊 Loading data from {args.data}...")
        df = DataLoader.load_csv_optimized(args.data)

        # アナライザー初期化
        analyzer = EnhancedFeatureAnalyzer(df=df, target_column=args.target_column)

        print(f"✅ Loaded {len(df)} rows, {len(analyzer.features)} features")

        # 出力ディレクトリ作成
        os.makedirs(args.output_dir, exist_ok=True)

        if args.analyze_quality:
            # 詳細品質分析
            print("\n🔍 Running detailed quality analysis...")
            quality_report = analyzer.analyze_feature_quality_detailed()

            # 品質レポート保存
            quality_file = os.path.join(
                args.output_dir,
                f"feature_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(quality_file, "w", encoding="utf-8") as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False)

            print(f"📄 Quality analysis saved to: {quality_file}")

            # 品質サマリー表示
            print("\n📊 Quality Analysis Summary:")
            print(f"   Total features: {quality_report['total_features']}")
            print(f"   Harmful features: {len(quality_report['harmful_features'])}")
            print(
                f"   Excellent quality: {len(quality_report['categories']['excellent'])}"
            )
            print(f"   Good quality: {len(quality_report['categories']['good'])}")
            print(f"   Fair quality: {len(quality_report['categories']['fair'])}")
            print(f"   Poor quality: {len(quality_report['categories']['poor'])}")

            # 推奨事項表示
            print("\n💡 Recommendations:")
            for rec in quality_report["recommendations"]:
                print(f"   {rec}")

        else:
            # 従来の特徴量選択
            print(f"\n🎯 Selecting optimal {args.target_features} features...")

            selected_features, removal_log = analyzer.select_optimal_features(
                target_count=args.target_features,
                nan_threshold=args.nan_threshold,
                correlation_threshold=args.correlation_threshold,
            )

            # 結果保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                args.output_dir, f"feature_selection_{timestamp}.json"
            )

            result = {
                "timestamp": timestamp,
                "original_features": len(analyzer.features),
                "selected_features": len(selected_features),
                "target_count": args.target_features,
                "selected_feature_list": selected_features,
                "removal_log": removal_log,
                "parameters": {
                    "nan_threshold": args.nan_threshold,
                    "correlation_threshold": args.correlation_threshold,
                },
            }

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print("✅ Feature selection completed!")
            print(
                f"   Selected: {len(selected_features)}/{len(analyzer.features)} features"
            )
            print(f"   Results saved to: {result_file}")

            # 特徴量リスト保存
            feature_list_file = os.path.join(
                args.output_dir, "recommended_features.txt"
            )
            with open(feature_list_file, "w", encoding="utf-8") as f:
                f.write("# Recommended Features\n")
                f.write(f"# Generated: {timestamp}\n")
                f.write(
                    f"# Selected: {len(selected_features)}/{len(analyzer.features)}\n\n"
                )
                for feature in selected_features:
                    f.write(f"{feature}\n")

            print(f"   Feature list saved to: {feature_list_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
