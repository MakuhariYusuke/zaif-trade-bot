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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Import existing utilities
from ztb.preprocessing.feature_correlation_filter import FeatureCorrelationProcessor
from ztb.utils.data_utils import load_csv_data


class EnhancedFeatureAnalyzer:
    """拡張特徴量分析システム"""

    def __init__(self, df: pd.DataFrame, target_column: str = "win"):
        self.df = df
        self.target_column = target_column
        self.features = self._identify_features()
        self.analysis_results: Dict[str, Any] = {}

    def _identify_features(self) -> List[str]:
        """特徴量列を識別"""
        exclude_cols = [
            "ts",
            "pair",
            "side",
            "pnl",
            "win",
            "source",
            "timestamp",
            "date",
            "datetime",
        ]
        features = [
            col
            for col in self.df.columns
            if col not in exclude_cols
            and self.df[col].dtype in ["float64", "int64", "float32", "int32"]
        ]
        return features

    def identify_harmful_features(
        self,
        nan_threshold: float = 0.10,  # 10% NaN率以上
        variance_threshold: float = 1e-10,  # 分散がほぼゼロ
        outlier_threshold: float = 0.30,  # 30% 外れ値以上
    ) -> Dict[str, Dict[str, Any]]:
        """
        Harmful特徴量を判定

        Harmfulの定義:
        1. NaN率が高い (>10%)
        2. 分散がほぼゼロ (定数特徴量)
        3. 外れ値が異常に多い (>30%)
        """
        harmful_features = {}

        for feature in self.features:
            data = self.df[feature]
            issues = []

            # 1. NaN率チェック
            nan_rate = data.isnull().sum() / len(data)
            if nan_rate > nan_threshold:
                issues.append(f"high_nan_rate:{nan_rate:.2%}")

            # 2. 分散チェック
            data_clean = data.dropna()
            if len(data_clean) > 0:
                variance = data_clean.var()
                if variance < variance_threshold:
                    issues.append(f"low_variance:{variance:.2e}")

            # 3. 外れ値チェック（IQR法）
            if len(data_clean) > 0:
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

            if issues:
                harmful_features[feature] = {
                    "issues": issues,
                    "nan_rate": float(nan_rate),
                    "variance": float(variance) if len(data_clean) > 0 else 0.0,
                    "severity": "critical" if len(issues) >= 2 else "moderate",
                }

        return harmful_features

    def select_by_correlation(
        self,
        correlation_threshold: float = 0.90,
        importance_dict: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
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
                feat1 = self.features[i]
                feat2 = self.features[j]

                if feat1 not in kept_features or feat2 not in kept_features:
                    continue

                corr_value = corr_matrix.loc[feat1, feat2]
                if (
                    isinstance(corr_value, (int, float))
                    and abs(corr_value) > correlation_threshold
                ):
                    # 高相関ペア発見
                    correlation_pairs.append(
                        {
                            "feature1": feat1,
                            "feature2": feat2,
                            "correlation": float(corr_value),
                        }
                    )

                    # どちらを削除するか決定
                    if importance_dict:
                        # 重要度ベース
                        imp1 = importance_dict.get(feat1, {}).get("importance_mean", 0)
                        imp2 = importance_dict.get(feat2, {}).get("importance_mean", 0)
                        to_remove = feat1 if imp1 < imp2 else feat2
                        to_keep = feat2 if to_remove == feat1 else feat1
                        reason = "lower_importance"
                    else:
                        # 分散ベース
                        var1 = self.df[feat1].var()
                        var2 = self.df[feat2].var()
                        to_remove = feat1 if var1 < var2 else feat2
                        to_keep = feat2 if to_remove == feat1 else feat1
                        reason = "lower_variance"

                    if to_remove in kept_features:
                        kept_features.remove(to_remove)
                        removed_features.append(
                            {
                                "feature": to_remove,
                                "reason": reason,
                                "correlated_with": to_keep,
                                "correlation": float(corr_value),
                            }
                        )

        return list(kept_features), removed_features

    def calculate_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Permutation Importanceを計算"""
        X = self.df[self.features].fillna(0)
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )

        importance_dict = {}
        for i, feature in enumerate(self.features):
            importance_dict[feature] = {
                "importance_mean": float(perm_importance.importances_mean[i]),
                "importance_std": float(perm_importance.importances_std[i]),
            }

        return importance_dict

    def suggest_optimal_features(
        self, target_count: int = 60, remove_harmful: bool = True
    ) -> Dict[str, Any]:
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

        print(f"✅ Selected {len(selected_features)} features")

        return {
            "selected_features": selected_features,
            "rejected_features": rejected_features,
            "harmful_features": harmful_features,
            "removed_by_correlation": removed_by_correlation,
            "importance_dict": importance_dict,
            "sorted_features": sorted_features,
            "summary": {
                "original_count": len(self.features),
                "harmful_count": len(harmful_features),
                "correlation_removed_count": len(removed_by_correlation),
                "final_count": len(selected_features),
                "reduction_rate": (1 - len(selected_features) / len(self.features))
                * 100,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Feature Selection Analysis")
    parser.add_argument(
        "--data",
        type=str,
        default="ml-dataset-enhanced-balanced.csv",
        help="Path to dataset",
    )
    parser.add_argument(
        "--target-features",
        type=int,
        default=60,
        help="Target number of features to select",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports", help="Output directory for reports"
    )

    args = parser.parse_args()

    # Load data
    print(f"📂 Loading data from {args.data}...")
    df = load_csv_data(args.data)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Create analyzer
    analyzer = EnhancedFeatureAnalyzer(df)
    print(f"   Identified {len(analyzer.features)} feature columns")

    # Run analysis
    results = analyzer.suggest_optimal_features(
        target_count=args.target_features, remove_harmful=True
    )

    # Print summary
    print("\n" + "=" * 80)
    print("📋 FEATURE SELECTION SUMMARY")
    print("=" * 80)
    print(f"Original features:     {results['summary']['original_count']}")
    print(f"Harmful features:      {results['summary']['harmful_count']}")
    print(f"Correlation removed:   {results['summary']['correlation_removed_count']}")
    print(f"Final selected:        {results['summary']['final_count']}")
    print(f"Reduction rate:        {results['summary']['reduction_rate']:.1f}%")
    print("=" * 80)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    report_path = os.path.join(args.output_dir, f"feature_selection_{timestamp}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed report saved to: {report_path}")

    # Selected features list
    selected_path = os.path.join(args.output_dir, "recommended_features.txt")
    with open(selected_path, "w", encoding="utf-8") as f:
        f.write("# Recommended Features (ranked by importance)\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total: {len(results['selected_features'])} features\n\n")
        for i, feat in enumerate(results["selected_features"], 1):
            imp = results["importance_dict"][feat]["importance_mean"]
            f.write(f"{i:3d}. {feat:40s} (importance: {imp:.6f})\n")
    print(f"📝 Recommended features saved to: {selected_path}")

    # Harmful features list
    if results["harmful_features"]:
        harmful_path = os.path.join(args.output_dir, "harmful_features.txt")
        with open(harmful_path, "w", encoding="utf-8") as f:
            f.write("# Harmful Features (should be removed)\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(results['harmful_features'])} features\n\n")
            for feat, info in results["harmful_features"].items():
                f.write(f"- {feat}\n")
                f.write(f"  Issues: {', '.join(info['issues'])}\n")
                f.write(f"  Severity: {info['severity']}\n\n")
        print(f"⚠️  Harmful features saved to: {harmful_path}")

    print("\n✅ Feature selection analysis complete!")


if __name__ == "__main__":
    main()
