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
    portfolio_csv = results_dir / "backtest_portfolio.csv"
    if not portfolio_csv.exists():
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
    stats_json = results_dir / "backtest_metrics.json"
    stats = None
    if stats_json.exists():
        with open(stats_json, "r", encoding="utf-8") as f:
            stats_payload = json.load(f)
        stats = stats_payload.get("metrics", stats_payload)
    else:
        stats_json = results_dir / "stats.json"
        if stats_json.exists():
            with open(stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
        
    if stats:
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

    # 3. 取引履歴詳細分析 (trades.json がある場合)
    trades_json = results_dir / "trades.json"
    trades_csv = results_dir / "backtest_trades.csv"
    if trades_csv.exists() or trades_json.exists():
        try:
            df_trades = None
            if trades_csv.exists():
                df_trades = pd.read_csv(trades_csv)
                trades = df_trades.to_dict("records")
            else:
                with open(trades_json, "r", encoding="utf-8") as f:
                    trades = json.load(f)
            
            if not trades:
                print("\n⚠️ 取引履歴が空です。")
                return

            if df_trades is None:
                df_trades = pd.DataFrame(trades)
            
            pnl_key = "pnl" if "pnl" in df_trades.columns else "gross_pnl" if "gross_pnl" in df_trades.columns else None
            if pnl_key is None:
                print("\n⚠️ trades.json に PnL 列が見つかりません。")
                return

            if "net_pnl" in df_trades.columns:
                df_trades["net_pnl"] = df_trades["net_pnl"].astype(float)
            elif "fee" in df_trades.columns or "slippage" in df_trades.columns:
                df_trades["fee"] = df_trades.get("fee", 0.0)
                df_trades["slippage"] = df_trades.get("slippage", 0.0)
                df_trades["cost"] = df_trades["fee"] + df_trades["slippage"]
                df_trades["net_pnl"] = df_trades[pnl_key] - df_trades["cost"]
            else:
                # 手数料推定 (往復 0.1% と仮定: Entry 0.05% + Exit 0.05%)
                FEE_RATE = 0.001
                df_trades["cost"] = (df_trades["entry"] * df_trades["size"] * FEE_RATE) + \
                                    (df_trades["exit"] * df_trades["size"] * FEE_RATE)
                df_trades["net_pnl"] = df_trades[pnl_key] - df_trades["cost"]
            
            gross_profit = df_trades[df_trades["net_pnl"] > 0]["net_pnl"].sum()
            gross_loss = df_trades[df_trades["net_pnl"] <= 0]["net_pnl"].sum()
            
            # Profit Factor
            pf = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
            
            # Win/Loss Stats
            winning_trades = df_trades[df_trades["net_pnl"] > 0]
            losing_trades = df_trades[df_trades["net_pnl"] <= 0]
            
            avg_win = winning_trades["net_pnl"].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades["net_pnl"].mean() if not losing_trades.empty else 0
            
            win_rate_net = len(winning_trades) / len(df_trades)
            loss_rate_net = len(losing_trades) / len(df_trades)
            
            # Expectancy
            expectancy = (avg_win * win_rate_net) + (avg_loss * loss_rate_net)
            
            print("\n🔍 詳細取引分析:")
            if "cost" in df_trades.columns:
                print(f"  総取引コスト: ¥{df_trades['cost'].sum():,.0f}")
            print(f"  Net PnL(推定):      ¥{df_trades['net_pnl'].sum():,.0f}")
            print(f"  Profit Factor:     {pf:.2f}")
            print(f"  平均利益 (Avg Win): ¥{avg_win:,.0f}")
            print(f"  平均損失 (Avg Loss): ¥{avg_loss:,.0f}")
            print(f"  勝率 (Net PnLベース): {win_rate_net*100:.1f}%")
            print(f"  期待値 (Expectancy): ¥{expectancy:,.0f} / trade")
            
        except Exception as e:
            print(f"\n❌ 詳細分析エラー: {e}")


if __name__ == "__main__":
    analyze_backtest_results()


if __name__ == "__main__":
    analyze_backtest_results()
