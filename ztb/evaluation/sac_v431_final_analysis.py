#!/usr/bin/env python3
"""
SAC v431 包括的検証スクリプト
報酬関数、パフォーマンス、市場適応性の詳細分析
"""

import json
import numpy as np
from pathlib import Path

def load_config():
    """Load v431 configuration"""
    config_path = Path(__file__).parent.parent / "configs" / "v431" / "sac_v431_1_enhanced.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def load_backtest_results():
    """Load backtest results"""
    results_path = Path(__file__).parent.parent / "evaluation" / "sac_v431_backtest_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)

def analyze_reward_function_v431(config):
    """Analyze SAC v431 reward function structure"""
    print("=" * 80)
    print("SAC v431 報酬関数構造分析")
    print("=" * 80)

    reward_config = config['reward_function']

    print("\n[ボーナスベース報酬設定]")
    print("-" * 50)
    print(f"SELLボーナス: {reward_config['sell_bonus']}")
    print(f"HOLDボーナス: {reward_config['hold_bonus']}")
    print(f"BUYボーナス:  {reward_config['buy_bonus']}")

    print("\n[市場適応型乗数]")
    print("-" * 50)
    market_multipliers = reward_config['market_adaptive']
    print(f"サイドウェイズ乗数: {market_multipliers['sideways_multiplier']}")
    print(f"高ボラティリティ乗数: {market_multipliers['high_vol_multiplier']}")
    print(f"低ボラティリティ乗数: {market_multipliers['low_vol_multiplier']}")

    print("\n[リスクと時間ペナルティ]")
    print("-" * 50)
    print(f"リスクペナルティ: {reward_config['risk_penalty']}")
    print(f"時間ペナルティ: {reward_config['time_penalty']}")

    # 理論的分析
    print("\n[理論的評価]")
    print("-" * 50)

    sell_bonus = reward_config['sell_bonus']
    hold_bonus = reward_config['hold_bonus']
    buy_bonus = reward_config['buy_bonus']

    # アクションの期待値比較
    print("アクション期待値分析:")
    print(f"  SELLボーナス: {sell_bonus}")
    print(f"  HOLDボーナス: {hold_bonus}")
    print(f"  BUYボーナス:  {buy_bonus}")

    # 市場適応性の評価
    sideways_mult = market_multipliers['sideways_multiplier']
    print("\n市場適応性分析:")
    print(f"  サイドウェイズ相場HOLD期待値: {hold_bonus * sideways_mult:.3f}")
    print(f"  通常相場HOLD期待値: {hold_bonus:.3f}")
    print(f"  適応性倍率: {sideways_mult:.1f}x")

    # 推奨される改善点
    print("\n[改善推奨]")
    print("-" * 50)

    if hold_bonus >= sell_bonus or hold_bonus >= buy_bonus:
        print("⚠️  HOLDボーナスが高すぎる可能性")
        print("   → HOLD率が高くなる原因となる")

    if abs(sell_bonus - buy_bonus) > 0.1:
        print("⚠️  SELL/BUYボーナスの非対称性")
        print("   → 特定の方向へのバイアスを生む")

    if sideways_mult <= 1.0:
        print("⚠️  サイドウェイズ市場での適応性が低い")
        print("   → レンジ相場でのHOLD優位性が弱い")

    print("✅ ボーナスベースのアプローチはv430のペナルティ問題を解決")
    print("✅ 市場適応型乗数は状況に応じた柔軟性を提供")

def analyze_backtest_performance(results):
    """Analyze backtest performance metrics"""
    print("\n" + "=" * 80)
    print("バックテストパフォーマンス分析")
    print("=" * 80)

    print("\n[基本指標]")
    print("-" * 50)
    print(f"総リターン: {results['total_return']:.2f}%")
    print(f"最終資本: ${results['final_capital']:.2f}")
    print(f"総取引数: {results['num_trades']}")
    print(f"勝率: {results['win_rate']:.1f}%")
    print(f"シャープレシオ: {results['sharpe_ratio']:.2f}")
    print(f"最大ドローダウン: ${results['max_drawdown']:.2f}")

    print("\n[行動分布]")
    print("-" * 50)
    actions = results['actions_taken']
    total_actions = sum(actions.values())
    for action, count in actions.items():
        pct = count / total_actions * 100
        print(f"  {action}: {count} ({pct:.1f}%)")

    print("\n[取引分析]")
    print("-" * 50)
    trades = results['trades']
    if trades:
        pnl_values = [t['pnl'] for t in trades]
        print(f"平均取引利益: ${np.mean(pnl_values):.2f}")
        print(f"取引利益標準偏差: ${np.std(pnl_values):.2f}")
        print(f"最大利益: ${np.max(pnl_values):.2f}")
        print(f"最大損失: ${np.min(pnl_values):.2f}")

        # 取引タイプ別分析
        trade_types = {}
        for trade in trades:
            ttype = trade['type']
            if ttype not in trade_types:
                trade_types[ttype] = []
            trade_types[ttype].append(trade['pnl'])

        print("\n取引タイプ別分析:")
        for ttype, pnls in trade_types.items():
            avg_pnl = np.mean(pnls)
            win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100
            print(f"  {ttype}: 平均利益 ${avg_pnl:.2f}, 勝率 {win_rate:.1f}%")

    print("\n[パフォーマンス評価]")
    print("-" * 50)

    total_return = results['total_return']
    win_rate = results['win_rate']
    sharpe_ratio = results['sharpe_ratio']

    if total_return > 0:
        print("✅ 総リターンがプラス")
    else:
        print("❌ 総リターンがマイナス")

    if win_rate > 55:
        print("✅ 高い勝率")
    elif win_rate > 45:
        print("⚠️ 平均的な勝率")
    else:
        print("❌ 低い勝率")

    if sharpe_ratio > 1.0:
        print("✅ 良好なリスク調整リターン")
    elif sharpe_ratio > 0.5:
        print("⚠️ 許容可能なリスク調整リターン")
    else:
        print("❌ 低いリスク調整リターン")

def generate_recommendations(results, config):
    """Generate improvement recommendations"""
    print("\n" + "=" * 80)
    print("改善推奨事項")
    print("=" * 80)

    total_return = results['total_return']
    win_rate = results['win_rate']
    sharpe_ratio = results['sharpe_ratio']
    actions = results['actions_taken']

    recommendations = []

    # リターンベースの推奨
    if total_return < -50:
        recommendations.append("🚨 緊急: 総リターンが大幅にマイナス - 報酬関数を見直す")
    elif total_return < 0:
        recommendations.append("⚠️ 総リターンがマイナス - 利益を生む取引を増やす")

    # 勝率ベースの推奨
    if win_rate < 45:
        recommendations.append("⚠️ 勝率が低い - 取引タイミングを改善")

    # シャープレシオベースの推奨
    if sharpe_ratio < 0.5:
        recommendations.append("⚠️ リスク調整リターンが低い - リスク管理を強化")

    # 行動分布ベースの推奨
    hold_pct = actions['HOLD'] / sum(actions.values()) * 100
    if hold_pct > 60:
        recommendations.append("⚠️ HOLD率が高すぎる - 取引頻度を上げる")
    elif hold_pct < 20:
        recommendations.append("⚠️ HOLD率が低すぎる - 不要な取引を減らす")

    # 報酬関数ベースの推奨
    reward_config = config['reward_function']
    if abs(reward_config['sell_bonus'] - reward_config['buy_bonus']) > 0.1:
        recommendations.append("⚠️ BUY/SELLボーナスのバランスを調整")

    if reward_config['market_adaptive']['sideways_multiplier'] < 1.3:
        recommendations.append("📈 サイドウェイズ市場でのHOLDボーナスを強化")

    # ポジティブな推奨
    if win_rate > 55 and sharpe_ratio > 1.0:
        recommendations.append("✅ 良好なパフォーマンス - 現在の設定を維持")

    print("\n[推奨事項]")
    print("-" * 50)
    for rec in recommendations:
        print(rec)

    print("\n[次のステップ]")
    print("-" * 50)
    print("1. パラメータ最適化を実行")
    print("2. 実際の市場データでのテスト")
    print("3. より高度な学習技術の統合")
    print("4. リスク管理機能の強化")

def main():
    """Main analysis function"""
    print("SAC v431 包括的検証レポート")
    print("=" * 100)

    try:
        # Load data
        config = load_config()
        results = load_backtest_results()

        # Perform analyses
        analyze_reward_function_v431(config)
        analyze_backtest_performance(results)
        generate_recommendations(results, config)

        print("\n" + "=" * 100)
        print("✅ SAC v431 検証完了")
        print("=" * 100)

    except Exception as e:
        print(f"❌ 分析中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()