#!/usr/bin/env python3
"""
SAC v428 Phase 2 Analysis Test Script
"""
import json
from ztb.analysis.analyze_backtest import BacktestAnalyzer

def main():
    print("🧪 SAC v428 Phase 2 Analysis Test")
    print("=" * 60)

    # バックテスト結果とトレーニングレポートを読み込み
    backtest_path = "results/sac_v428_mock_backtest.json"
    report_path = "reports/training_report_sac_sac_v428_position_optimized_20251018_215151.json"
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            training_report = json.load(f)

        print('✅ トレーニングレポート読み込み完了')
        print(f'モデル: {training_report.get("model_name", "unknown")}')
        print(f'タイムステップ: {training_report.get("total_timesteps", 0)}')
        print(f'アクション分布: {training_report.get("action_distribution", {})}')

        # BacktestAnalyzerで分析を実行
        analyzer = BacktestAnalyzer(backtest_path, report_path)

        print('\n=== 包括的なレポート生成 ===')
        comprehensive_report = analyzer.generate_comprehensive_report()
        print(comprehensive_report)

        # レポートをファイルに保存
        output_path = "reports/sac_v428_phase2_comprehensive_analysis.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)

        print(f"\n✅ 分析レポート保存完了: {output_path}")

    except FileNotFoundError:
        print(f"❌ トレーニングレポートが見つかりません: {report_path}")
    except Exception as e:
        print(f"❌ 分析実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()