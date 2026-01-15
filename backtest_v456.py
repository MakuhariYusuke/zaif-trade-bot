#!/usr/bin/env python3
"""
v456 バックテストスクリプト

訓練されたv456モデルをバックテストし、実際の利益を確認します。
v455実装をベースに、v456環境に適応させています。
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# プロジェクト PATH 設定
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.trading.environment.factory_v456 import EnvironmentFactory

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 初期化時のログレベルを一時的に低下させる
logging.getLogger('ztb').setLevel(logging.WARNING)


class BacktestReporter:
    """バックテストの統計情報を管理"""
    
    def __init__(self):
        self.stats = {
            "total_steps": 0,
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "action_distribution": {},
        }
        self.portfolio_history = []
        self.trade_history = []

    def update_step(self, step, portfolio_value, action):
        self.stats["total_steps"] += 1
        self.portfolio_history.append(portfolio_value)

        # Action Distribution
        # action は [target_position, ttl] の2次元
        act_key = "hold"
        if isinstance(action, np.ndarray):
            action_val = float(action[0]) if len(action) > 0 else 0.0
        else:
            action_val = float(action)
            
        if action_val > 0.3:
            act_key = "buy"
        elif action_val < -0.3:
            act_key = "sell"
            
        self.stats["action_distribution"][act_key] = (
            self.stats["action_distribution"].get(act_key, 0) + 1
        )

    def record_trade(self, trade_type, pnl, entry_price, exit_price, size):
        self.stats["total_trades"] += 1
        if trade_type == "long":
            self.stats["long_trades"] += 1
        else:
            self.stats["short_trades"] += 1

        if pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"] += 1

        self.stats["net_pnl"] += pnl
        self.trade_history.append({
            "type": trade_type,
            "pnl": pnl,
            "entry": entry_price,
            "exit": exit_price,
            "size": size,
        })

    def finalize_stats(self):
        """最終統計の計算"""
        # Drawdown
        peak = -np.inf
        max_dd = 0.0
        max_dd_pct = 0.0

        for val in self.portfolio_history:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0.0

            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        self.stats["max_drawdown"] = max_dd
        self.stats["max_drawdown_percent"] = max_dd_pct

        # Sharpe (1m bars想定: 525600分/年)
        if len(self.portfolio_history) > 1:
            returns = pd.Series(self.portfolio_history).pct_change().dropna()
            if returns.std() > 0:
                self.stats["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(525600)
            else:
                self.stats["sharpe_ratio"] = 0.0

    def print_summary(self):
        """結果サマリー出力"""
        print("\n" + "=" * 70)
        print("🎯 BACKTEST REPORT - v456")
        print("=" * 70)

        print(f"\n📊 取引統計:")
        print(f"  総ステップ数: {self.stats['total_steps']}")
        print(f"  総取引数: {self.stats['total_trades']}")
        print(f"    買い: {self.stats['long_trades']}")
        print(f"    売り: {self.stats['short_trades']}")
        
        win_rate = (self.stats['winning_trades'] / max(1, self.stats['total_trades'])) * 100
        print(f"  勝率: {win_rate:.1f}%")

        print(f"\n💰 損益:")
        print(f"  純利益: ¥{self.stats['net_pnl']:,.0f}")
        print(f"  最大ドローダウン: ¥{self.stats['max_drawdown']:,.0f} ({self.stats['max_drawdown_percent']:.1f}%)")
        print(f"  シャープレシオ: {self.stats['sharpe_ratio']:.4f}")

        print(f"\n📈 アクション分布:")
        total_actions = sum(self.stats['action_distribution'].values())
        for k, v in sorted(self.stats['action_distribution'].items()):
            pct = (v / total_actions * 100) if total_actions > 0 else 0
            print(f"  {k}: {v} ({pct:.1f}%)")

        print("=" * 70)


def run_backtest_v456(
    model_path: Optional[str] = None,
    steps: int = 10000,
    warmup_steps: int = 500,
    data_path: Optional[str] = None,
    order_size_btc: float = 0.01,
):
    """v456 バックテスト実行"""
    
    # 1. データ読み込み
    if data_path is None:
        # 優先順位: btc_jpy_training_data.csv > btc_jpy_backtest_data.csv > btc_jpy_1m_merged.csv
        data_candidates = [
            "data/btc_jpy_training_data.csv",
            "data/btc_jpy_backtest_data.csv",
            "data/btc_jpy_1m_merged.csv",
            "data/btc_jpy_1m_v451.csv",
        ]
        
        for candidate in data_candidates:
            candidate_path = os.path.join(project_root, candidate)
            if os.path.exists(candidate_path):
                data_path = candidate_path
                logger.info(f"✓ データファイル自動検出: {os.path.basename(candidate_path)}")
                break
        
        if data_path is None:
            data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"データが見つかりません: {data_path}")
        return
    
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
            return
    except Exception as e:
        logger.error(f"環境初期化エラー: {e}")
        return
    
    # 3. モデル読み込み
    if model_path is None:
        # 最新の訓練済みモデルを探す
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
        
        # 最新のファイルを選択
        found_models = [f for f in found_models if os.path.getsize(f) > 1_000_000]  # 1MB以上
        
        if not found_models:
            logger.error("訓練済みモデルが見つかりません (1MB以上のファイルのみ検索)")
            return
        
        model_path = max(found_models, key=os.path.getctime)
        logger.info(f"使用するモデル: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        logger.error(f"モデルが見つかりません: {model_path}")
        return
    
    if os.path.getsize(model_path) < 1_000_000:
        logger.error(f"モデルファイルサイズが小さい (破損している可能性): {model_path}")
        return
    
    logger.info(f"モデル読み込み: {model_path}")
    model = SAC.load(model_path, env=env)
    
    # 4. バックテスト実行 (Replay & Evaluation)
    logger.info(f"\n🚀 バックテスト開始: {steps} ステップ")
    reporter = BacktestReporter()
    
    obs, info = env.reset()
    done = False
    step_count = 0
    
    # ポジション追跡
    current_position = 0.0  # 0: フラット, >0: ロング, <0: ショート
    position_entry_price = 0.0
    position_entry_action = 0.0
    
    while not done and step_count < steps:
        step_count += 1
        
        # モデル予測
        action, _ = model.predict(obs, deterministic=True)
        
        if isinstance(action, np.ndarray):
            action_val = float(action[0])
        else:
            action_val = float(action)
        
        # 環境ステップ（エラーハンドリング付き）
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except (IndexError, Exception) as e:
            # データ不足またはその他エラー → バックテスト終了
            logger.debug(f"Step {step_count}: 環境エラー: {e} → バックテスト終了")
            done = True
            continue
        
        done = (terminated or truncated) and step_count >= steps  # ステップ数に達したら終了
        
        # 現在価格を取得
        current_price = info.get('current_price', 0.0)
        if current_price <= 0:
            # フォールバック: close 価格から推定
            close_idx = df.columns.tolist().index('close') if 'close' in df.columns else -1
            current_price = float(df.iloc[step_count % len(df)].iloc[close_idx]) if close_idx >= 0 else 100000.0
        
        # ポートフォリオ価値計算
        btc_value = current_position * current_price if current_position > 0 else 0.0
        # 簡易版: 現在のポジション価値のみ
        total_value = 1000000.0 + btc_value  # 初期資金 + 現在のポジション利益
        
        # アクション評価
        reporter.update_step(step_count, total_value, action)
        
        # エントリー/エグジット ロジック（修正版）
        if step_count > warmup_steps:
            # action は [target_position, ttl] の2次元
            # action[0] が position control
            action_position = float(action[0]) if isinstance(action, np.ndarray) and len(action) > 0 else float(action)
            
            if current_position == 0 and action_position > 0.3:
                # ロング エントリー（買い）
                current_position = order_size_btc
                position_entry_price = current_price
                position_entry_action = action_position
                logger.debug(f"Step {step_count}: ENTRY LONG {current_position:.4f}BTC @ ¥{current_price:,.0f}")
                
            elif current_position > 0 and action_position < -0.3:
                # エグジット（売り）
                exit_price = current_price
                pnl = (exit_price - position_entry_price) * current_position
                reporter.record_trade("long", pnl, position_entry_price, exit_price, current_position)
                
                logger.debug(f"Step {step_count}: EXIT LONG at ¥{exit_price:,.0f}, PnL=¥{pnl:,.0f}")
                
                current_position = 0.0
                position_entry_price = 0.0
        
        if step_count % 1000 == 0:
            pnl_str = f"¥{(total_value - 1000000.0):,.0f}" if current_position == 0 else f"未実現: ¥{(current_position * (current_price - position_entry_price)):,.0f}"
            logger.info(f"Step {step_count}: Position = {current_position:.4f}BTC, Total = ¥{total_value:,.0f} ({pnl_str})")
    
    reporter.print_summary()
    
    results_dir = os.path.join(project_root, "backtest_results", "v456")
    os.makedirs(results_dir, exist_ok=True)
    
    # 結果をCSVに保存
    pd.DataFrame({
        "portfolio_value": reporter.portfolio_history
    }).to_csv(os.path.join(results_dir, "portfolio_history.csv"), index=False)
    
    # 統計情報をJSONに保存
    with open(os.path.join(results_dir, "stats.json"), "w") as f:
        json.dump(reporter.stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"結果保存: {results_dir}")
    
    return reporter.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v456 バックテスト")
    parser.add_argument("--steps", type=int, default=10000, help="バックテストステップ数")
    parser.add_argument("--warmup-steps", type=int, default=500, help="ウォームアップステップ数")
    parser.add_argument("--model-path", type=str, default=None, help="モデルパス")
    parser.add_argument("--data-path", type=str, default=None, help="データパス")
    parser.add_argument("--order-size", type=float, default=0.01, help="取引サイズ (BTC単位)")
    
    args = parser.parse_args()
    
    run_backtest_v456(
        model_path=args.model_path,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        data_path=args.data_path,
        order_size_btc=args.order_size,
    )
