#!/usr/bin/env python3
"""
Enhanced Backtest Analysis for SAC v446
ドル/円変換、BTC増加量計算、現実世界期間計算を含む詳細分析
"""
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def analyze_backtest_results():
    """バックテスト結果の詳細分析"""

    # 結果読み込み
    with open('backtest_results_sac_v446.json', 'r') as f:
        data = json.load(f)

    # データ読み込み
    df = pd.read_csv('data/btc_jpy_real_dataset.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("=" * 60)
    print("🎯 SAC v446 バックテスト詳細分析")
    print("=" * 60)

    # 基本情報
    print("
📊 基本情報:"    print(f"モデル: {data['model_name']}")
    print(f"設定ファイル: {data['config_path']}")
    print(f"総ステップ数: {data['total_steps']:,}")

    # 期間計算
    total_steps = data['total_steps']
    # データは1時間間隔なので、1ステップ = 1時間
    total_hours = total_steps
    total_days = total_hours / 24
    total_weeks = total_days / 7
    total_months = total_days / 30

    print("
⏰ 期間分析:"    print(f"総時間: {total_hours:.0f}時間")
    print(f"総日数: {total_days:.1f}日")
    print(f"総週間: {total_weeks:.1f}週間")
    print(f"総月数: {total_months:.1f}ヶ月")

    # 価格データ分析
    prices = data['price_history']
    if prices:
        price_start = prices[0]
        price_end = prices[-1]
        price_change_pct = ((price_end - price_start) / price_start) * 100

        print("
💰 市場価格分析:"        print(f"開始価格: ¥{price_start:,.0f}")
        print(f"終了価格: ¥{price_end:,.0f}")
        print(f"価格変動: {price_change_pct:+.2f}%")

    # ポートフォリオ分析
    initial_portfolio = data['initial_portfolio']
    final_portfolio = data['final_portfolio']
    total_return_pct = data['total_return_pct']
    total_reward = data['total_reward']

    print("
💼 ポートフォリオ分析:"    print(f"初期残高: ¥{initial_portfolio:,.0f}")
    print(f"最終残高: ¥{final_portfolio:,.0f}")
    print(f"総リターン: {total_return_pct:.2f}%")
    print(f"総利益: ¥{final_portfolio - initial_portfolio:,.0f}")
    print(f"総報酬: {total_reward:,.2f}")

    # BTC残高分析（仮想環境なので推定）
    btc_holdings = data.get('btc_holdings', [])
    if btc_holdings:
        btc_start = btc_holdings[0] if btc_holdings else 0
        btc_end = btc_holdings[-1] if btc_holdings else 0
        btc_change = btc_end - btc_start

        print("
₿ BTC残高分析:"        print(f"BTC初期残高: {btc_start:.6f} BTC")
        print(f"BTC最終残高: {btc_end:.6f} BTC")
        print(f"BTC増加量: {btc_change:+.6f} BTC")

        # BTC価値計算（最終価格ベース）
        if prices and btc_end > 0:
            btc_value_jpy = btc_end * price_end
            print(f"BTC現在価値: ¥{btc_value_jpy:,.0f}")

    # アクション分布修正
    actions = data.get('actions', [])
    if actions:
        # アクションの正しい解釈: -1=SELL, 0=HOLD, 1=BUY
        sell_count = sum(1 for a in actions if a == -1)
        hold_count = sum(1 for a in actions if a == 0)
        buy_count = sum(1 for a in actions if a == 1)
        total_actions = len(actions)

        print("
🎮 アクション分布 (修正版):"        print(f"BUY (購入): {buy_count}回 ({buy_count/total_actions*100:.1f}%)")
        print(f"HOLD (保持): {hold_count}回 ({hold_count/total_actions*100:.1f}%)")
        print(f"SELL (売却): {sell_count}回 ({sell_count/total_actions*100:.1f}%)")

        # アクション分布のバランスチェック
        if sell_count > buy_count * 2:
            print("⚠️  SELLバイアスが強い（学習時と一致しない可能性）")
        elif abs(buy_count - sell_count) < total_actions * 0.1:
            print("✅ BUY/SELLバランス良好")
        else:
            print("⚠️  BUY/SELLバランスに偏りあり")

    # ポートフォリオ履歴分析
    portfolio_history = data['portfolio_history']
    if portfolio_history:
        portfolio_array = np.array(portfolio_history)
        max_portfolio = np.max(portfolio_array)
        min_portfolio = np.min(portfolio_array)
        volatility = np.std(portfolio_array)

        print("
📈 ポートフォリオ履歴分析:"        print(f"最大残高: ¥{max_portfolio:,.0f}")
        print(f"最小残高: ¥{min_portfolio:,.0f}")
        print(f"残高変動性: ¥{volatility:,.0f}")

        # ドローダウン計算
        peak = portfolio_array[0]
        max_drawdown = 0
        for value in portfolio_array:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        print(f"最大ドローダウン: {max_drawdown:.2f}%")

    # 収益性評価
    print("
🏆 収益性評価:"    if total_return_pct > 50:
        print("🌟 優良: 50%以上のリターン")
    elif total_return_pct > 20:
        print("✅ 良好: 20%以上のリターン")
    elif total_return_pct > 0:
        print("⚠️ 微益: プラスのリターン")
    else:
        print("❌ 損失: マイナスのリターン")

    # 時間あたりリターン
    hourly_return = total_return_pct / total_hours if total_hours > 0 else 0
    daily_return = total_return_pct / total_days if total_days > 0 else 0

    print("
⏱️ 時間あたりリターン:"    print(f"時間あたり: {hourly_return:.4f}%")
    print(f"日あたり: {daily_return:.4f}%")

    # 問題点の指摘
    print("
🔍 潜在的な問題点:"    issues = []

    if total_reward < -10000:
        issues.append("総報酬が極端に低い（報酬関数に問題の可能性）")

    if sell_count > buy_count * 3:
        issues.append("SELLアクションが異常に多い（学習時と乖離）")

    if total_return_pct > price_change_pct * 2:
        issues.append("市場価格変動を大きく上回るリターン（非現実的）")

    if not issues:
        issues.append("重大な問題は検出されませんでした")

    for issue in issues:
        print(f"• {issue}")

    print("=" * 60)

if __name__ == "__main__":
    analyze_backtest_results()
