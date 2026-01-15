#!/usr/bin/env python3
"""v456 バックテスト（最終修正版）

バグ修正:
1. 現金とBTC保有量を別途追跡
2. PnLは売却時に現金で正確に計算
3. 手数料を正確に反映
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.factory_v456 import EnvironmentFactory


class FinalBacktestReporter:
    """最終版バックテストレポーター"""
    
    def __init__(self):
        self.portfolio_values = []
        self.cash_values = []
        self.btc_values = []
        self.actions = []
        self.trades = []
        self.stats = {}
    
    def add_step(self, portfolio_value, cash, btc, action):
        self.portfolio_values.append(portfolio_value)
        self.cash_values.append(cash)
        self.btc_values.append(btc)
        
        action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        self.actions.append(action_val)
    
    def add_trade(self, step, entry_price, exit_price, size, cash_change):
        """取引を記録"""
        pnl = cash_change
        self.trades.append({
            "step": step,
            "entry": entry_price,
            "exit": exit_price,
            "size": size,
            "pnl": pnl,
        })
    
    def finalize(self):
        if not self.portfolio_values:
            return
        
        portfolio = np.array(self.portfolio_values)
        actions = np.array(self.actions)
        
        initial = 1000000.0
        final = portfolio[-1]
        
        # Stats
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        total_trades = len(self.trades)
        
        # Drawdown
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
            "total_pnl": sum(t['pnl'] for t in self.trades),
            "max_dd": max_dd,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "buy_count": int(buy_count),
            "sell_count": int(sell_count),
            "hold_count": int(hold_count),
        }
    
    def print_summary(self):
        if not self.stats:
            return
        
        print("\n" + "=" * 90)
        print("🎯 BACKTEST REPORT - v456 (FINAL)")
        print("=" * 90)
        
        print(f"\n💰 ポートフォリオ:")
        print(f"  初期資本: ¥{self.stats['initial']:,.0f}")
        print(f"  最終資本: ¥{self.stats['final']:,.0f}")
        print(f"  純損益: ¥{self.stats['pnl']:,.0f}")
        print(f"  リターン: {self.stats['return_pct']:.2f}%")
        
        print(f"\n📊 取引統計:")
        print(f"  総取引数: {self.stats['trades']}")
        if self.stats['trades'] > 0:
            print(f"  勝ち: {self.stats['wins']} ({self.stats['win_rate']:.1f}%)")
            print(f"  負け: {self.stats['losses']}")
            print(f"  合計PnL: ¥{self.stats['total_pnl']:,.0f}")
        
        print(f"\n📈 リスク:")
        print(f"  最大DD: ¥{self.stats['max_dd']:,.0f} ({self.stats['max_dd_pct']:.2f}%)")
        print(f"  シャープ: {self.stats['sharpe']:.4f}")
        
        print(f"\n🔄 アクション分布:")
        total_actions = self.stats['buy_count'] + self.stats['sell_count'] + self.stats['hold_count']
        if total_actions > 0:
            print(f"  買い: {self.stats['buy_count']} ({self.stats['buy_count']/total_actions*100:.1f}%)")
            print(f"  売り: {self.stats['sell_count']} ({self.stats['sell_count']/total_actions*100:.1f}%)")
            print(f"  ホールド: {self.stats['hold_count']} ({self.stats['hold_count']/total_actions*100:.1f}%)")
        
        if self.trades:
            print(f"\n🔍 取引履歴 (最初の5件):")
            for i, t in enumerate(self.trades[:5]):
                sign = "+" if t['pnl'] > 0 else ""
                pct = (t['pnl'] / (t['entry'] * t['size']) * 100) if t['entry'] > 0 else 0
                print(f"  {i+1}. Entry={t['entry']:>9,.0f}, Exit={t['exit']:>9,.0f}, "
                      f"Size={t['size']:.4f}, PnL={sign}¥{t['pnl']:>9,.0f} ({sign}{pct:.2f}%)")
        
        print("=" * 90)


def run_final_backtest(
    model_path: Optional[str] = None,
    steps: int = 50000,
    warmup_steps: int = 1000,
    data_path: Optional[str] = None,
    order_size_btc: float = 0.01,
):
    """最終版バックテスト"""
    
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
    logger.info(f"  行数: {len(df)}")
    
    # 環境作成
    logger.info("環境初期化中...")
    try:
        factory = EnvironmentFactory(
            df=df,
            initial_balance=1_000_000.0,
            max_position=0.1,
            commission_rate=0.001,
        )
        env = factory.create_training_env()
        if env is None:
            logger.error("環境作成失敗")
            return None
        
        # ご褒美パラメータを設定（簡素化版）
        if not hasattr(env, 'reward_params'):
            env.reward_params = {}
        
        env.reward_params.update({
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0,
            'edge_penalty_rate': 0.0,
            'vol_floor_penalty': 0.0,
            'hold_ramp': 0.0,
        })
        
    except Exception as e:
        logger.error(f"環境エラー: {e}")
        return None
    
    # モデル読み込み
    if model_path is None:
        import glob
        patterns = [
            "models/v456/final/v456_simplified_*.zip",  # 最新の簡素化モデルを優先
            "models/v456/final/v456_trained_*.zip",
            "best_model/*.zip"
        ]
        found = []
        for p in patterns:
            found.extend(glob.glob(os.path.join(project_root, p)))
        found = [f for f in found if os.path.getsize(f) > 1_000_000]
        if not found:
            logger.error("モデルなし")
            return None
        model_path = max(found, key=os.path.getctime)
    
    logger.info(f"モデル読み込み: {os.path.basename(model_path)}")
    model = SAC.load(model_path, env=env)
    
    # バックテスト実行
    logger.info(f"\n🚀 バックテスト開始: {steps} ステップ\n")
    reporter = FinalBacktestReporter()
    
    obs, info = env.reset()
    done = False
    step_count = 0
    
    # ポジション管理（現金ベース）
    cash = 1000000.0
    btc_held = 0.0
    entry_price = 0.0
    
    while not done and step_count < steps:
        step_count += 1
        
        # 予測
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception as e:
            logger.warning(f"予測失敗: {e}")
            break
        
        # 環境ステップ
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            logger.debug(f"環境エラー: {e} → 終了")
            done = True
            continue
        
        done = (terminated or truncated) and step_count >= steps
        
        # 価格取得
        current_price = info.get('current_price', 0.0)
        if current_price <= 0:
            if step_count <= len(df):
                current_price = float(df['close'].iloc[step_count - 1])
            else:
                break
        
        # ポートフォリオ価値
        btc_value = btc_held * current_price
        total_value = cash + btc_value
        reporter.add_step(total_value, cash, btc_held, action)
        
        # トレードロジック
        if step_count > warmup_steps:
            action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            if btc_held == 0 and action_val > 0.3:
                # 買い
                cost = order_size_btc * current_price * 1.001
                if cash >= cost:
                    btc_held = order_size_btc
                    entry_price = current_price
                    cash -= cost
                    logger.debug(f"Step {step_count}: BUY {btc_held:.4f}BTC @ ¥{current_price:,.0f}")
            
            elif btc_held > 0 and action_val < -0.3:
                # 売り
                proceeds = btc_held * current_price * 0.999
                cash_before = cash
                cash += proceeds
                pnl = cash - (cash_before + btc_held * entry_price * 1.001)  # 手数料込み
                
                reporter.add_trade(step_count, entry_price, current_price, btc_held, proceeds - (btc_held * entry_price * 1.001))
                
                logger.debug(f"Step {step_count}: SELL {btc_held:.4f}BTC @ ¥{current_price:,.0f}, "
                           f"PnL=¥{proceeds - (btc_held * entry_price * 1.001):,.0f}")
                
                btc_held = 0.0
                entry_price = 0.0
        
        if step_count % 5000 == 0:
            logger.info(f"Step {step_count}: BTC={btc_held:.4f}, Cash=¥{cash:,.0f}, Total=¥{total_value:,.0f}")
    
    logger.info(f"\n✅ バックテスト完了 ({step_count} steps)")
    
    reporter.finalize()
    reporter.print_summary()
    
    results_dir = os.path.join(project_root, "backtest_results", "v456")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "stats.json"), "w") as f:
        json.dump(reporter.stats, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(results_dir, "trades.json"), "w") as f:
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
    
    stats = run_final_backtest(**vars(args))
    sys.exit(0 if stats else 1)
