#!/usr/bin/env python3
"""
SAC v434.1 詳細バックテスト分析スクリプト
"""

from ztb.io.json_io import read_json

from ztb.analysis.comparative.analyze_backtest import BacktestAnalyzer


def main():
    # バックテスト結果を読み込み
    results = read_json("backtest_results_sac_model_20251022_041525.json")

    print("=== SAC v434.1 詳細バックテスト分析 ===")
    print(f"モデル: {results['model_name']}")
    print(f"エピソード数: {results['episodes']}")
    print(f"平均リターン: {results['avg_return']:.2f}%")
    print(f"総取引回数: {results['total_trades']}")
    print()

    # BacktestAnalyzerで詳細分析
    try:
        analyzer = BacktestAnalyzer("backtest_results_sac_model_20251022_041525.json")
        analyzer.analyze()
    except Exception as e:
        print(f"BacktestAnalyzer実行エラー: {e}")
        print("基本的な分析のみを行います...")

        # 基本的な分析
        analyze_basic_results(results)


def analyze_basic_results(results):
    """基本的な結果分析"""
    print("=== 基本分析結果 ===")

    # 収益性分析
    print("収益性分析:")
    print(f"  平均リターン: {results['avg_return']:.2f}%")
    print(f"  最高リターン: {results['best_return']:.2f}%")
    print(f"  最低リターン: {results['worst_return']:.2f}%")

    # 取引パターン分析
    print("\n取引パターン分析:")
    print(f"  総取引回数: {results['total_trades']:,}")
    print(f"  エピソードあたり取引: {results['trades_per_episode']:.1f}")
    print(
        f"  取引頻度: {results['trades_per_episode']/5000*100:.1f}% (データポイントあたり)"
    )

    # 安定性分析
    print("\n安定性分析:")
    print(f"  リターンの標準偏差: {results['std_return']:.2f}%")
    print(f"  報酬の標準偏差: {results['std_reward']:.2f}")

    # 潜在的な問題点の特定
    print("\n=== 潜在的な問題点と改善提案 ===")

    if results["avg_return"] == 0.0 and results["std_return"] == 0.0:
        print("🚨 問題: 全てのエピソードで収益が0%")
        print("   → モデルが取引を実行しているが、利益が出ていない")
        print("   → 報酬関数の設計や学習戦略の見直しが必要")

    if results["trades_per_episode"] > 4000:
        print("🚨 問題: 過度な取引頻度")
        print("   → エピソードあたり4,621回の取引は非現実的")
        print("   → 取引コストが収益を圧迫している可能性")

    if results["std_reward"] == 0.0:
        print("🚨 問題: 完全に決定論的な行動")
        print("   → 全てのエピソードで全く同じ結果")
        print("   → 確率的探索が不十分または学習が収束しすぎ")

    print("\n=== v434.2 改善提案 ===")
    print("1. 報酬関数の見直し:")
    print("   - 取引コストをより厳しくペナルティ化")
    print("   - 利益実現のインセンティブを強化")
    print("   - 過度な取引を抑制する仕組みの導入")

    print("2. 学習戦略の改善:")
    print("   - カリキュラム学習の段階的難易度調整")
    print("   - 市場レジーム適応の強化")
    print("   - アンサンブル学習の多様性向上")

    print("3. 特徴量エンジニアリング:")
    print("   - 156個の特徴量を効果的に活用")
    print("   - 市場レジーム固有の特徴量生成")
    print("   - 特徴量の重要度分析と選択")

    print("4. 連続行動の最適化:")
    print("   - SACの連続行動空間の有効活用")
    print("   - ポジションサイズの動的調整")
    print("   - リスク管理の統合")


if __name__ == "__main__":
    main()
