#!/usr/bin/env python3
"""
拡張特徴量分析実行スクリプト

既存のEnhancedFeatureAnalyzerを拡張し、SAC v427特徴量のharmful判定を行う
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_enhanced_feature_analysis(
    data_path: str = None,
    target_column: str = "future_return",
    output_dir: str = "docs/features/analysis/results",
):
    """
    拡張特徴量分析を実行

    Args:
        data_path: データファイルパス（Noneの場合はサンプルデータ生成）
        target_column: ターゲット列名
        output_dir: 出力ディレクトリ
    """
    print("🚀 Starting Enhanced Feature Analysis with Harmful Detection")
    print("=" * 60)

    try:
        from ztb.analysis.specialized.features.analyze_feature_selection import (
            EnhancedFeatureAnalyzer,
        )

        # データ準備
        if data_path and os.path.exists(data_path):
            print(f"📊 Loading data from: {data_path}")
            df = pd.read_csv(data_path)
        else:
            print("📊 Generating sample SAC v427 feature data...")
            df = generate_sample_sac_features()
            data_path = "sample_sac_v427_features.csv"

        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")

        # アナライザー初期化
        analyzer = EnhancedFeatureAnalyzer(df=df, target_column=target_column)

        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # 1. 詳細品質分析実行
        print("\n🔍 Phase 1: Detailed Quality Analysis")
        quality_report = analyzer.analyze_feature_quality_detailed()

        # 2. Harmful特徴量の詳細分析
        print("\n⚠️  Phase 2: Harmful Feature Detection")
        harmful_features = analyzer.identify_harmful_features()

        # 3. SAC v427特化分析
        print("\n🎯 Phase 3: SAC v427 Specific Analysis")
        sac_analysis = analyzer._analyze_sac_v427_feature_quality()

        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 総合レポート
        comprehensive_report = {
            "timestamp": timestamp,
            "data_source": data_path,
            "analysis_type": "enhanced_sac_v427_feature_analysis",
            "summary": {
                "total_features": len(analyzer.features),
                "harmful_features_count": len(harmful_features),
                "excellent_quality_count": len(
                    quality_report["categories"]["excellent"]
                ),
                "poor_quality_count": len(quality_report["categories"]["poor"]),
            },
            "quality_analysis": quality_report,
            "harmful_features": harmful_features,
            "sac_v427_analysis": sac_analysis,
            "recommendations": generate_actionable_recommendations(
                quality_report, harmful_features, sac_analysis
            ),
        }

        # JSON保存
        report_file = os.path.join(
            output_dir, f"sac_v427_feature_analysis_{timestamp}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

        # テキストサマリー保存
        summary_file = os.path.join(output_dir, f"analysis_summary_{timestamp}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(generate_text_summary(comprehensive_report))

        # Harmful特徴量リスト保存
        if harmful_features:
            harmful_file = os.path.join(output_dir, f"harmful_features_{timestamp}.txt")
            with open(harmful_file, "w", encoding="utf-8") as f:
                f.write("# Harmful Features Requiring Removal\n")
                f.write(f"# Generated: {timestamp}\n")
                f.write(f"# Total harmful: {len(harmful_features)}\n\n")
                for feature, details in harmful_features.items():
                    f.write(f"{feature}\n")
                    f.write(f"  Severity: {details['severity']}\n")
                    f.write(f"  Issues: {', '.join(details['issues'])}\n")
                    f.write(f"  Recommendation: {details['recommendation']}\n\n")

        print("\n✅ Analysis Complete!")
        print(f"📄 Comprehensive report: {report_file}")
        print(f"📋 Summary: {summary_file}")
        if harmful_features:
            print(f"⚠️  Harmful features list: {harmful_file}")

        # コンソールに主要結果を表示
        display_key_findings(comprehensive_report)

        return comprehensive_report

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_sample_sac_features(n_samples: int = 1000) -> pd.DataFrame:
    """SAC v427特徴量のサンプルデータを生成"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="1H")

    # 基本価格データ生成
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, n_samples)))
    high = close * (1 + np.random.uniform(0, 0.01, n_samples))
    low = close * (1 - np.random.uniform(0, 0.01, n_samples))
    open_price = close + np.random.normal(0, close * 0.005, n_samples)
    volume = np.random.uniform(1000, 10000, n_samples)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    # SAC v427特徴量をシミュレート
    returns = df["close"].pct_change()

    # 市場レジーム特徴量
    vol_20 = returns.rolling(20).std()
    vol_50 = returns.rolling(50).std()
    df["volatility_regime"] = (vol_20 > vol_50).astype(int)

    sma_20 = df["close"].rolling(20).mean()
    sma_50 = df["close"].rolling(50).mean()
    df["trend_regime"] = (sma_20 > sma_50).astype(int)

    # 相関特徴量
    for lag in [1, 5, 10]:
        lagged_returns = returns.shift(lag)
        df[f"price_correlation_lag_{lag}"] = returns.rolling(50).corr(lagged_returns)

    # 技術指標
    df["rsi_14"] = 50 + np.random.normal(0, 10, n_samples)  # 簡易RSI
    df["macd"] = np.random.normal(0, 0.01, n_samples)
    df["bb_position"] = np.random.normal(0, 0.2, n_samples)

    # アンサンブル特徴量
    df["ensemble_confidence_bull"] = np.random.uniform(0.3, 0.9, n_samples)
    df["ensemble_pred_buy"] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

    # ターゲット生成
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1

    # NaNを追加してharmful判定をテスト
    df = df.reset_index(drop=True)  # indexをリセット
    harmful_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
    df.loc[harmful_indices, "rsi_14"] = np.nan

    # 定数特徴量を追加
    df["constant_feature"] = 1.0

    # 過度に相関する特徴量を追加
    df["highly_correlated_feature"] = df["close"] * 0.99 + np.random.normal(
        0, 0.01, len(df)
    )

    return df


def generate_actionable_recommendations(
    quality_report: dict, harmful_features: dict, sac_analysis: dict
) -> list:
    """実行可能な推奨事項を生成"""
    recommendations = []

    # Harmful特徴量の推奨
    if harmful_features:
        critical = [
            f for f, d in harmful_features.items() if d["severity"] == "critical"
        ]
        moderate = [
            f for f, d in harmful_features.items() if d["severity"] == "moderate"
        ]

        recommendations.append(
            {
                "priority": "high",
                "action": "remove_harmful_features",
                "description": f"Remove {len(harmful_features)} harmful features ({len(critical)} critical, {len(moderate)} moderate)",
                "features": list(harmful_features.keys()),
                "impact": "Improve model stability and reduce overfitting",
            }
        )

    # 品質改善の推奨
    poor_features = quality_report["categories"]["poor"]
    if len(poor_features) > 0:
        recommendations.append(
            {
                "priority": "medium",
                "action": "review_poor_quality_features",
                "description": f"Review {len(poor_features)} poor quality features",
                "features": poor_features,
                "impact": "Enhance feature reliability",
            }
        )

    # SAC v427特化の推奨
    category_stats = sac_analysis.get("category_stats", {})
    for category, stats in category_stats.items():
        if stats["poor_count"] > stats["count"] * 0.3:
            recommendations.append(
                {
                    "priority": "medium",
                    "action": f"improve_{category}_quality",
                    "description": f"Improve quality of {category} features ({stats['poor_count']}/{stats['count']} are poor)",
                    "impact": f"Enhance {category} feature effectiveness",
                }
            )

    return recommendations


def generate_text_summary(report: dict) -> str:
    """テキスト形式のサマリーを生成"""
    summary = []
    summary.append("SAC v427 Enhanced Feature Analysis Summary")
    summary.append("=" * 50)
    summary.append(f"Generated: {report['timestamp']}")
    summary.append(f"Data Source: {report['data_source']}")
    summary.append("")

    summary.append("📊 OVERVIEW")
    summary.append(f"Total Features: {report['summary']['total_features']}")
    summary.append(f"Harmful Features: {report['summary']['harmful_features_count']}")
    summary.append(f"Excellent Quality: {report['summary']['excellent_quality_count']}")
    summary.append(f"Poor Quality: {report['summary']['poor_quality_count']}")
    summary.append("")

    if report["harmful_features"]:
        summary.append("⚠️  HARMFUL FEATURES")
        for feature, details in list(report["harmful_features"].items())[:10]:  # Top 10
            summary.append(f"• {feature} ({details['severity']})")
            summary.append(f"  Issues: {', '.join(details['issues'])}")
        if len(report["harmful_features"]) > 10:
            summary.append(f"... and {len(report['harmful_features']) - 10} more")
        summary.append("")

    summary.append("💡 RECOMMENDATIONS")
    for rec in report["recommendations"]:
        summary.append(f"• {rec['description']}")
        summary.append(f"  Impact: {rec['impact']}")
    summary.append("")

    return "\n".join(summary)


def display_key_findings(report: dict):
    """主要な発見をコンソールに表示"""
    print("\n🔑 KEY FINDINGS")
    print("-" * 30)

    summary = report["summary"]
    print(f"📈 Total Features Analyzed: {summary['total_features']}")
    print(f"⚠️  Harmful Features Found: {summary['harmful_features_count']}")
    print(f"⭐ Excellent Quality: {summary['excellent_quality_count']}")
    print(f"📉 Poor Quality: {summary['poor_quality_count']}")

    if report["harmful_features"]:
        print("\n🚨 Top Harmful Features:")
        harmful_list = list(report["harmful_features"].items())[:5]
        for feature, details in harmful_list:
            print(
                f"   • {feature} ({details['severity']}) - {', '.join(details['issues'][:2])}"
            )

    print("\n🎯 Next Steps:")
    for rec in report["recommendations"][:3]:
        print(f"   • {rec['description']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="拡張SAC v427特徴量分析")
    parser.add_argument("--data", help="データファイルパス")
    parser.add_argument(
        "--target-column", default="future_return", help="ターゲット列名"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/features/analysis/results",
        help="出力ディレクトリ",
    )

    args = parser.parse_args()

    run_enhanced_feature_analysis(
        data_path=args.data,
        target_column=args.target_column,
        output_dir=args.output_dir,
    )
