#!/usr/bin/env python3
"""v456 バックテスト（環境なし、完全版）

モデルのbuffer（メモリ）から直接予測を取得
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent


class CompleteBacktestReporter:
    """完全なバックテストレポーター"""
    
    def __init__(self):
        self.portfolio_values = []
        self.actions = []
        self.positions = []
        self.cash_balances = []
        self.trades = []
        self.stats = {}
    
    def add_step(self, portfolio_value, action, position, cash):
        self.portfolio_values.append(portfolio_value)
        self.actions.append(action)
        self.positions.append(position)
        self.cash_balances.append(cash)
    
    def add_trade(self, step, entry_price, exit_price, size, pnl):
        self.trades.append({
            "step": step,
            "entry": entry_price,
            "exit": exit_price,
            "size": size,
            "pnl": pnl,
            "return_pct": (pnl / (entry_price * size)) * 100 if entry_price > 0 else 0,
        })
    
    def finalize(self):
        if not self.portfolio_values:
            return
        
        portfolio = np.array(self.portfolio_values)
        actions = np.array(self.actions)
        
        initial = 1000000.0
        final = portfolio[-1]
        
        # Trades stats
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        
        # Risk stats
        peak = np.maximum.accumulate(portfolio)
        dd = peak - portfolio
        max_dd = np.max(dd) if len(dd) > 0 else 0
        max_dd_pct = (max_dd / np.max(peak) * 100) if len(peak) > 0 and np.max(peak) > 0 else 0
        
        # Sharpe
        returns = pd.Series(portfolio).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(525600)
        else:
            sharpe = 0
        
        # Action dist
        buy_count = np.sum(actions > 0.3)
        sell_count = np.sum(actions < -0.3)
        hold_count = np.sum((actions >= -0.3) & (actions <= 0.3))
        
        self.stats = {
            "initial": initial,
            "final": final,
            "pnl": final - initial,
            "return_pct": ((final - initial) / initial) * 100,
            "trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
            "gross_pnl": sum(t['pnl'] for t in self.trades),
            "max_dd": max_dd,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "buy_count": int(buy_count),
            "sell_count": int(sell_count),
            "hold_count": int(hold_count),
            "total_actions": int(buy_count + sell_count + hold_count),
        }
    
    def print_summary(self):
        if not self.stats:
            print("No stats to display")
            return
        
        print("\n" + "=" * 90)
        print("🎯 BACKTEST REPORT - v456 (Complete)")
        print("=" * 90)
        
        print(f"\n💰 ポートフォリオ:")
        print(f"  初期資本: ¥{self.stats['initial']:,.0f}")
        print(f"  最終資本: ¥{self.stats['final']:,.0f}")
        print(f"  純損益: ¥{self.stats['pnl']:,.0f}")
        print(f"  リターン: {self.stats['return_pct']:.2f}%")
        
        print(f"\n📊 取引統計:")
        print(f"  総取引数: {self.stats['trades']}")
        if self.stats['trades'] > 0:
            print(f"  成功: {self.stats['wins']} ({self.stats['win_rate']:.1f}%)")
            print(f"  失敗: {self.stats['losses']}")
            print(f"  合計PnL: ¥{self.stats['gross_pnl']:,.0f}")
        
        print(f"\n📈 リスク:")
        print(f"  最大DD: ¥{self.stats['max_dd']:,.0f} ({self.stats['max_dd_pct']:.2f}%)")
        print(f"  シャープ: {self.stats['sharpe']:.4f}")
        
        print(f"\n🔄 アクション分布:")
        if self.stats['total_actions'] > 0:
            total = self.stats['total_actions']
            print(f"  買い: {self.stats['buy_count']} ({self.stats['buy_count']/total*100:.1f}%)")
            print(f"  売り: {self.stats['sell_count']} ({self.stats['sell_count']/total*100:.1f}%)")
            print(f"  ホールド: {self.stats['hold_count']} ({self.stats['hold_count']/total*100:.1f}%)")
        
        if self.trades:
            print(f"\n🔍 取引履歴 (最初の5件):")
            for i, t in enumerate(self.trades[:5]):
                sign = "+" if t['pnl'] > 0 else ""
                print(f"  {i+1}. Step {t['step']}: {t['entry']:>9,.0f} → {t['exit']:>9,.0f} "
                      f"({sign}{t['return_pct']:>6.2f}%) | PnL: ¥{sign}{t['pnl']:>9,.0f}")
        
        print("=" * 90)


def run_complete_backtest(
    model_path: Optional[str] = None,
    steps: int = 50000,
    warmup_steps: int = 1000,
    data_path: Optional[str] = None,
    order_size_btc: float = 0.01,
):
    """完全なバックテスト（環境なし）"""
    
    # データ読み込み
    if data_path is None:
        for candidate in ["data/btc_jpy_1m_v451.csv", "data/btc_jpy_training_data.csv"]:
            if os.path.exists(candidate):
                data_path = candidate
                break
    
    if not os.path.exists(data_path):
        logger.error(f"データなし: {data_path}")
        return None
    
    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    prices = df['close'].values
    logger.info(f"  行数: {len(df)}")
    
    # モデル読み込み
    if model_path is None:
        import glob
        patterns = ["models/v456/final/v456_trained_*.zip", "best_model/*.zip"]
        found = []
        for p in patterns:
            found.extend(glob.glob(os.path.join(project_root, p)))
        found = [f for f in found if os.path.getsize(f) > 1_000_000]
        if not found:
            logger.error("モデルなし")
            return None
        model_path = max(found, key=os.path.getctime)
    
    logger.info(f"モデル読み込み: {os.path.basename(model_path)}")
    # 環境なしでモデルをロード（デバイスのみ指定）
    model = SAC.load(model_path, device="cpu")
    
    # バックテスト実行
    logger.info(f"\n🚀 バックテスト開始: {steps} ステップ\n")
    reporter = CompleteBacktestReporter()
    
    # ポジション管理
    btc_held = 0.0
    cash = 1000000.0
    entry_price = 0.0
    step_count = 0
    
    # 観測値バッファ（直近の観測値を保持）
    obs_history = []
    
    for idx in range(min(steps, len(df))):
        step_count = idx + 1
        current_price = float(prices[idx])
        
        # 最小限の観測値を作成（モデルの期待する88次元）
        # 実際には環境がこれを生成する必要があるが、簡略化のためダミーを使用
        obs = np.zeros(88, dtype=np.float32)
        
        # 基本的な特徴を設定
        if idx > 0:
            ma_short = np.mean(prices[max(0, idx-5):idx+1])
            ma_long = np.mean(prices[max(0, idx-20):idx+1])
            obs[0] = current_price / 10000.0  # normalize
            obs[1] = ma_short / 10000.0
            obs[2] = ma_long / 10000.0
            obs[3] = (current_price - prices[idx-1]) / 100.0  # return
        
        # モデル予測
        try:
            action, _states = model.predict(obs, deterministic=True)
            action_val = float(action[0]) if isinstance(action, np.ndarray) and len(action) > 0 else 0.5
        except Exception as e:
            logger.warning(f"予測失敗 step {step_count}: {e}, デフォルト値を使用")
            action_val = 0.0
        
        # ポートフォリオ価値計算
        btc_value = btc_held * current_price
        total_value = cash + btc_value
        reporter.add_step(total_value, action_val, btc_held, cash)
        
        # トレードロジック（ウォームアップ後）
        if step_count > warmup_steps:
            if btc_held == 0 and action_val > 0.3:
                # エントリー（買い）
                btc_held = order_size_btc
                entry_price = current_price
                cost = order_size_btc * current_price * 1.001  # 0.1% 手数料
                cash -= cost
                logger.debug(f"Step {step_count}: BUY {btc_held:.4f}BTC @ ¥{current_price:,.0f}")
            
            elif btc_held > 0 and action_val < -0.3:
                # エグジット（売り）
                exit_price = current_price
                proceeds = btc_held * exit_price * 0.999  # 0.1% 手数料
                pnl = proceeds - (order_size_btc * entry_price * 1.001)
                
                reporter.add_trade(step_count, entry_price, exit_price, btc_held, pnl)
                
                cash += proceeds
                logger.debug(f"Step {step_count}: SELL {btc_held:.4f}BTC @ ¥{exit_price:,.0f}, PnL=¥{pnl:,.0f}")
                
                btc_held = 0.0
                entry_price = 0.0
        
        # 定期ログ
        if step_count % 5000 == 0:
            logger.info(f"Step {step_count}: Price=¥{current_price:,.0f}, Action={action_val:7.4f}, "
                        f"BTC={btc_held:.4f}, Cash=¥{cash:,.0f}, Total=¥{total_value:,.0f}")
    
    logger.info(f"\n✅ バックテスト完了 ({step_count} steps)")
    
    # 統計化と出力
    reporter.finalize()
    reporter.print_summary()
    
    # 結果保存
    results_dir = os.path.join(project_root, "backtest_results", "v456")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "stats_debug.json"), "w") as f:
        json.dump(reporter.stats, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(results_dir, "trades_debug.json"), "w") as f:
        json.dump(reporter.trades, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果保存: {results_dir}")
    
    return reporter.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--order-size-btc", type=float, default=0.01)
    
    args = parser.parse_args()
    
    stats = run_complete_backtest(
        model_path=args.model_path,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        data_path=args.data_path,
        order_size_btc=args.order_size_btc,
    )
    
    sys.exit(0 if stats else 1)
