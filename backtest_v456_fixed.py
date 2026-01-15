#!/usr/bin/env python3
"""v456 バックテスト（修正版）

修正点:
1. ポートフォリオ価値の正確な追跡（USD + BTC価値）
2. ラウンドトリップ取引の完全な記録
3. アクション分布とトレードの整合性
4. 現在保有中のポジション評価の正確性
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルートを取得
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.factory_v456 import EnvironmentFactory


class BacktestReporterFixed:
    """修正されたバックテストレポーター"""
    
    def __init__(self):
        self.portfolio_history = []  # ステップごとのポートフォリオ価値
        self.trade_history = []  # ラウンドトリップ取引
        self.action_history = []  # アクション履歴
        self.btc_history = []  # BTC保有量履歴
        self.cash_history = []  # 現金残高履歴
        
        self.stats = {
            "total_steps": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }
    
    def update_step(self, step, portfolio_value, btc_held, cash_balance, action):
        """ステップ更新"""
        self.stats["total_steps"] += 1
        self.portfolio_history.append(portfolio_value)
        self.btc_history.append(btc_held)
        self.cash_history.append(cash_balance)
        
        # アクション分布
        if isinstance(action, np.ndarray):
            action_val = float(action[0])
        else:
            action_val = float(action)
        
        if action_val > 0.3:
            self.stats["buy_count"] += 1
        elif action_val < -0.3:
            self.stats["sell_count"] += 1
        else:
            self.stats["hold_count"] += 1
        
        self.action_history.append(action_val)
    
    def record_trade(self, entry_price, exit_price, btc_size, pnl, step):
        """ラウンドトリップ取引を記録"""
        self.stats["total_trades"] += 1
        
        if pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"] += 1
        
        self.stats["net_pnl"] += pnl
        
        trade = {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "btc_size": btc_size,
            "pnl": pnl,
            "pnl_percent": (pnl / (entry_price * btc_size)) * 100 if entry_price > 0 else 0,
            "step": step,
        }
        self.trade_history.append(trade)
        
        logger.debug(f"Trade recorded at step {step}: Entry={entry_price}, Exit={exit_price}, "
                    f"Size={btc_size:.4f}BTC, PnL=¥{pnl:,.0f}")
    
    def finalize_stats(self):
        """最終統計の計算"""
        # Drawdown
        if self.portfolio_history:
            peak = max(self.portfolio_history[0], 1000000.0)
            max_dd = 0.0
            max_dd_pct = 0.0
            
            for val in self.portfolio_history:
                if val > peak:
                    peak = val
                dd = peak - val
                dd_pct = (dd / peak) * 100 if peak > 0 else 0
                
                if dd > max_dd:
                    max_dd = dd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            
            self.stats["max_drawdown"] = max_dd
            self.stats["max_drawdown_percent"] = max_dd_pct
            
            # Sharpe Ratio (1分足 = 525,600分/年)
            returns = pd.Series(self.portfolio_history).pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                annual_return = returns.mean() * 525600
                annual_vol = returns.std() * np.sqrt(525600)
                self.stats["sharpe_ratio"] = annual_return / annual_vol if annual_vol > 0 else 0
            else:
                self.stats["sharpe_ratio"] = 0.0
    
    def print_summary(self):
        """結果サマリー出力"""
        final_portfolio = self.portfolio_history[-1] if self.portfolio_history else 1000000.0
        initial_portfolio = 1000000.0
        total_return = ((final_portfolio - initial_portfolio) / initial_portfolio) * 100
        
        print("\n" + "=" * 80)
        print("🎯 BACKTEST REPORT - v456 (FIXED)")
        print("=" * 80)
        
        print(f"\n📊 取引統計:")
        print(f"  総ステップ数: {self.stats['total_steps']}")
        print(f"  総取引数: {self.stats['total_trades']}")
        print(f"    成功取引: {self.stats['winning_trades']}")
        print(f"    失敗取引: {self.stats['losing_trades']}")
        
        if self.stats['total_trades'] > 0:
            win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            print(f"  勝率: {win_rate:.1f}%")
        else:
            print(f"  勝率: N/A (取引なし)")
        
        print(f"\n💰 損益:")
        print(f"  初期資本: ¥{initial_portfolio:,.0f}")
        print(f"  最終資本: ¥{final_portfolio:,.0f}")
        print(f"  純利益: ¥{self.stats['net_pnl']:,.0f}")
        print(f"  総リターン: {total_return:.2f}%")
        print(f"  最大ドローダウン: ¥{self.stats['max_drawdown']:,.0f} ({self.stats['max_drawdown_percent']:.2f}%)")
        print(f"  シャープレシオ: {self.stats['sharpe_ratio']:.4f}")
        
        print(f"\n📈 アクション分布:")
        total_actions = self.stats['buy_count'] + self.stats['sell_count'] + self.stats['hold_count']
        if total_actions > 0:
            print(f"  買い: {self.stats['buy_count']} ({self.stats['buy_count']/total_actions*100:.1f}%)")
            print(f"  売り: {self.stats['sell_count']} ({self.stats['sell_count']/total_actions*100:.1f}%)")
            print(f"  ホールド: {self.stats['hold_count']} ({self.stats['hold_count']/total_actions*100:.1f}%)")
        
        if self.trade_history:
            print(f"\n🔍 取引詳細 (最初の5件):")
            for i, trade in enumerate(self.trade_history[:5]):
                print(f"  Trade {i+1}: Entry={trade['entry_price']:,.0f}, Exit={trade['exit_price']:,.0f}, "
                      f"Size={trade['btc_size']:.4f}BTC, PnL=¥{trade['pnl']:,.0f} ({trade['pnl_percent']:.2f}%)")
        
        print("=" * 80)


def run_backtest_fixed(
    model_path: Optional[str] = None,
    steps: int = 10000,
    warmup_steps: int = 500,
    data_path: Optional[str] = None,
    order_size_btc: float = 0.01,
):
    """v456 バックテスト実行（修正版）"""
    
    # 1. データ読み込み
    if data_path is None:
        data_candidates = [
            "data/btc_jpy_1m_v451.csv",
            "data/btc_jpy_training_data.csv",
            "data/btc_jpy_backtest_data.csv",
            "data/btc_jpy_1m_merged.csv",
        ]
        
        for candidate in data_candidates:
            candidate_path = os.path.join(project_root, candidate)
            if os.path.exists(candidate_path):
                data_path = candidate_path
                logger.info(f"✓ データファイル自動検出: {os.path.basename(candidate_path)}")
                break
        
        if data_path is None:
            logger.error("データが見つかりません")
            return None
    
    if not os.path.exists(data_path):
        logger.error(f"データが見つかりません: {data_path}")
        return None
    
    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"  行数: {len(df)}")
    
    # 2. 環境作成
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
            logger.error("環境作成に失敗しました")
            return None
    except Exception as e:
        logger.error(f"環境初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 3. モデル読み込み
    if model_path is None:
        import glob
        model_search_patterns = [
            "models/v456/final/v456_trained_*.zip",
            "models/phase3/sac_v456_phase3_*.zip",
            "best_model/*.zip",
            "checkpoints/v456/*.zip",
        ]
        
        found_models = []
        for pattern in model_search_patterns:
            full_pattern = os.path.join(project_root, pattern)
            found_models.extend(glob.glob(full_pattern))
        
        found_models = [f for f in found_models if os.path.getsize(f) > 1_000_000]
        
        if not found_models:
            logger.error("訓練済みモデルが見つかりません")
            return None
        
        model_path = max(found_models, key=os.path.getctime)
        logger.info(f"使用するモデル: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        logger.error(f"モデルが見つかりません: {model_path}")
        return None
    
    logger.info(f"モデル読み込み: {model_path}")
    model = SAC.load(model_path, env=env)
    
    # 4. バックテスト実行
    logger.info(f"\n🚀 バックテスト開始: {steps} ステップ")
    reporter = BacktestReporterFixed()
    
    obs, info = env.reset()
    done = False
    step_count = 0
    
    # ポジション追跡（正確版）
    btc_held = 0.0  # 現在のBTC保有量
    cash_balance = 1_000_000.0  # 現金残高
    position_entry_price = 0.0  # エントリー価格
    
    while not done and step_count < steps:
        step_count += 1
        
        # モデル予測
        action, _ = model.predict(obs, deterministic=True)
        
        if isinstance(action, np.ndarray):
            action_val = float(action[0])
        else:
            action_val = float(action)
        
        # 現在価格を取得（step前）
        if step_count <= len(df):
            current_price = float(df['close'].iloc[step_count - 1])
        else:
            current_price = float(df['close'].iloc[-1])
        
        # ポートフォリオ価値計算（step前）
        btc_value = btc_held * current_price
        total_value = cash_balance + btc_value
        
        # ステップ更新
        reporter.update_step(step_count, total_value, btc_held, cash_balance, action)
        
        # 環境ステップ
        obs, reward, terminated, truncated, info = env.step(action)
        done = (terminated or truncated) and step_count >= steps
        
        # アクション実行（ウォームアップ後）
        if step_count > warmup_steps:
            if btc_held == 0 and action_val > 0.3:
                # ロング エントリー
                btc_held = order_size_btc
                position_entry_price = current_price
                cash_balance -= order_size_btc * current_price * (1 + 0.001)  # 手数料を含む
                logger.debug(f"Step {step_count}: BUY {btc_held:.4f}BTC @ ¥{current_price:,.0f}")
                
            elif btc_held > 0 and action_val < -0.3:
                # ロング クローズ
                exit_price = current_price
                pnl = (exit_price - position_entry_price) * btc_held
                
                # 現金残高を更新
                cash_balance += btc_held * exit_price * (1 - 0.001)  # 手数料を含む
                
                # 取引を記録
                reporter.record_trade(position_entry_price, exit_price, btc_held, pnl, step_count)
                
                logger.debug(f"Step {step_count}: SELL {btc_held:.4f}BTC @ ¥{exit_price:,.0f}, PnL=¥{pnl:,.0f}")
                
                btc_held = 0.0
                position_entry_price = 0.0
        
        # 定期ログ
        if step_count % 5000 == 0:
            btc_value = btc_held * current_price
            total_value = cash_balance + btc_value
            logger.info(f"Step {step_count}: BTC={btc_held:.4f}, Cash=¥{cash_balance:,.0f}, Total=¥{total_value:,.0f}")
    
    logger.info(f"\n✅ バックテスト完了 (ステップ数: {step_count})")
    
    # 5. 最終統計計算と出力
    reporter.finalize_stats()
    reporter.print_summary()
    
    # 6. 結果保存
    results_dir = os.path.join(project_root, "backtest_results", "v456_fixed")
    os.makedirs(results_dir, exist_ok=True)
    
    pd.DataFrame({
        "portfolio_value": reporter.portfolio_history,
        "btc_held": reporter.btc_history,
        "cash_balance": reporter.cash_history,
    }).to_csv(os.path.join(results_dir, "portfolio_history.csv"), index=False)
    
    with open(os.path.join(results_dir, "stats.json"), "w") as f:
        json.dump(reporter.stats, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(results_dir, "trades.json"), "w") as f:
        json.dump(reporter.trade_history, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果保存: {results_dir}")
    
    return reporter.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v456 バックテスト (修正版)")
    parser.add_argument("--steps", type=int, default=10000, help="バックテストステップ数")
    parser.add_argument("--warmup-steps", type=int, default=500, help="ウォームアップステップ数")
    parser.add_argument("--model-path", type=str, default=None, help="モデルパス")
    parser.add_argument("--data-path", type=str, default=None, help="データパス")
    parser.add_argument("--order-size", type=float, default=0.01, help="取引サイズ (BTC単位)")
    
    args = parser.parse_args()
    
    stats = run_backtest_fixed(
        model_path=args.model_path,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        data_path=args.data_path,
        order_size_btc=args.order_size,
    )
    
    if stats:
        sys.exit(0)
    else:
        sys.exit(1)
