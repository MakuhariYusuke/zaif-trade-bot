#!/usr/bin/env python3
"""
v456 バックテスト結果分析スクリプト

backtest_results/v456/ の結果を詳細分析
"""

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


def analyze_backtest_results():
    """バックテスト結果を分析"""
    
    results_dir = Path("backtest_results/v456")
    
    if not results_dir.exists():
        print(f"❌ 結果ディレクトリが見つかりません: {results_dir}")
        return
    
    # 1. ポートフォリオ履歴を読み込み
    portfolio_csv = results_dir / "portfolio_history.csv"
    if portfolio_csv.exists():
        df_portfolio = pd.read_csv(portfolio_csv)
        print("\n📈 ポートフォリオ価値推移:")
        print(f"  初期値: ¥{df_portfolio['portfolio_value'].iloc[0]:,.0f}")
        print(f"  最終値: ¥{df_portfolio['portfolio_value'].iloc[-1]:,.0f}")
        print(f"  最大値: ¥{df_portfolio['portfolio_value'].max():,.0f}")
        print(f"  最小値: ¥{df_portfolio['portfolio_value'].min():,.0f}")
        print(f"  ステップ数: {len(df_portfolio)}")
        
        # リターン計算
        returns = df_portfolio['portfolio_value'].pct_change().dropna()
        print(f"\n📊 リターン統計:")
        print(f"  平均日次リターン: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
        print(f"  標準偏差: {returns.std():.6f} ({returns.std()*100:.4f}%)")
        print(f"  シャープレシオ (1分足): {(returns.mean()/returns.std())*np.sqrt(525600):.4f}")
    
    # 2. 統計情報を読み込み
    stats_json = results_dir / "stats.json"
    if stats_json.exists():
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        
        print("\n💰 取引統計:")
        print(f"  総ステップ数: {stats['total_steps']}")
        print(f"  総取引数: {stats['total_trades']}")
        print(f"  ロング: {stats['long_trades']}")
        print(f"  ショート: {stats['short_trades']}")
        print(f"  勝ち: {stats['winning_trades']} ({(stats['winning_trades']/max(1,stats['total_trades']))*100:.1f}%)")
        print(f"  負け: {stats['losing_trades']} ({(stats['losing_trades']/max(1,stats['total_trades']))*100:.1f}%)")
        
        print(f"\n📊 リスク指標:")
        print(f"  純利益: ¥{stats['net_pnl']:,.0f}")
        print(f"  最大ドローダウン: ¥{stats['max_drawdown']:,.0f} ({stats['max_drawdown_percent']*100:.2f}%)")
        print(f"  シャープレシオ: {stats['sharpe_ratio']:.4f}")
        
        print(f"\n📈 アクション分布:")
        for action, count in sorted(stats['action_distribution'].items()):
            pct = (count / stats['total_steps'] * 100) if stats['total_steps'] > 0 else 0
            print(f"  {action}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    analyze_backtest_results()
