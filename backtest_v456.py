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

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import (
    load_config_dict,
    extract_env_config,
)

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
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "action_distribution": {},
            "raw_action_sum": 0.0,
            "raw_action_count": 0,
            "abs_action_sum": 0.0,
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
        
        # Action Strength Stats
        self.stats["raw_action_sum"] += action_val
        self.stats["abs_action_sum"] += abs(action_val)
        self.stats["raw_action_count"] += 1

    def record_trade(self, trade_type, pnl, entry_price, exit_price, size, fee, slippage):
        self.stats["total_trades"] += 1
        if trade_type == "long":
            self.stats["long_trades"] += 1
        else:
            self.stats["short_trades"] += 1
        
        # Note: PnL here is usually Gross PnL from price diff.
        # We need to subtract costs for Net PnL.
        net_pnl = pnl - fee - slippage

        if net_pnl > 0:
            self.stats["winning_trades"] += 1
        else:
            self.stats["losing_trades"] += 1

        self.stats["gross_pnl"] += pnl
        self.stats["net_pnl"] += net_pnl
        self.stats["total_fees"] += fee
        self.stats["total_slippage"] += slippage

        self.trade_history.append({
            "type": trade_type,
            "gross_pnl": pnl,
            "net_pnl": net_pnl,
            "entry": entry_price,
            "exit": exit_price,
            "size": size,
            "fee": fee,
            "slippage": slippage
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
        print(f"  純利益 (Net PnL): ¥{self.stats['net_pnl']:,.0f}")
        print(f"  総利 (Gross PnL): ¥{self.stats['gross_pnl']:,.0f}")
        print(f"  手数料 (Fee):     -¥{self.stats['total_fees']:,.0f}")
        print(f"  スリッページ:     -¥{self.stats['total_slippage']:,.0f}")
        print(f"  最大ドローダウン: ¥{self.stats['max_drawdown']:,.0f} ({self.stats['max_drawdown_percent']:.1f}%)")
        print(f"  シャープレシオ: {self.stats['sharpe_ratio']:.4f}")

        print(f"\n📈 アクション詳細:")
        total_actions = sum(self.stats['action_distribution'].values())
        for k, v in sorted(self.stats['action_distribution'].items()):
            pct = (v / total_actions * 100) if total_actions > 0 else 0
            print(f"  {k}: {v} ({pct:.1f}%)")
        
        avg_abs_action = self.stats["abs_action_sum"] / max(1, self.stats["raw_action_count"])
        print(f"  平均アクション強度 (|Action|): {avg_abs_action:.4f}")
        
        # Profit Factor & Expectancy Calculation
        gross_profit = sum(t['gross_pnl'] for t in self.trade_history if t['gross_pnl'] > 0)
        gross_loss = sum(abs(t['gross_pnl']) for t in self.trade_history if t['gross_pnl'] < 0)
        profit_factor = gross_profit / max(1.0, gross_loss)
        
        print(f"\n📊 高度指標:")
        print(f"  Profit Factor: {profit_factor:.2f}")
        if self.stats['total_trades'] > 0:
            avg_return = self.stats['net_pnl'] / self.stats['total_trades']
            print(f"  Expectancy (Avg Net PnL/Trade): ¥{avg_return:,.0f}")
        
        print("=" * 70)

    def save_results(self, results_dir):
        """結果をファイルに保存"""
        os.makedirs(results_dir, exist_ok=True)
        
        # Portfolio History
        pd.DataFrame({
            "portfolio_value": self.portfolio_history
        }).to_csv(os.path.join(results_dir, "portfolio_history.csv"), index=False)
        
        # Stats
        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
        # Trade History (New)
        with open(os.path.join(results_dir, "trades.json"), "w") as f:
            def convert(o):
                if isinstance(o, (np.int64, np.int32)): return int(o)
                if isinstance(o, (np.float64, np.float32)): return float(o)
                return str(o)
            json.dump(self.trade_history, f, indent=2, ensure_ascii=False, default=convert)
            
        logger.info(f"結果保存: {results_dir}")


def run_backtest_v456(
    model_path: Optional[str] = None,
    steps: int = 10000,
    warmup_steps: int = 500,
    data_path: Optional[str] = None,
    order_size_btc: float = 0.01,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_fixed_ttl: bool = False,
):
    """v456 バックテスト実行"""
    
    # Config Loading
    env_config_override = {}
    if config_path and os.path.exists(config_path):
        logger.info(f"Config読み込み: {config_path}")
        full_config = load_config_dict(Path(config_path))
        env_config_override = extract_env_config(full_config)
        
        # Merge some top-level config if useful
        if "min_delta" in full_config.get("training", {}):
             env_config_override["min_delta"] = full_config["training"]["min_delta"]
        # Also check root level or specific overrides needed
        if "min_delta" in full_config:
             env_config_override["min_delta"] = full_config["min_delta"]

        logger.info(f"  Env Config Override: {list(env_config_override.keys())}")
        if "min_delta" in env_config_override:
            logger.info(f"  min_delta applied: {env_config_override['min_delta']}")

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
    
    # Calculate Features
    logger.info("特徴量計算中...")
    df = calculate_base_features(df, copy=False)
    
    try:
        env_config = {
            "initial_balance": 1_000_000.0,
            "max_position_size": 0.1,
            "transaction_cost": 0.001,
        }
        # Override with loaded config
        env_config.update(env_config_override)
        
        # Extract Wrapper Configs (not passed to env, but used in script)
        # Config can contain these keys at root or separate
        script_cooldown_steps = env_config_override.get("cooldown_steps", 0) # Fallback to env config if needed
        action_threshold = env_config_override.get("action_threshold", 0.3) # Default 0.3
        
        logger.info(f"Script Wrapper Config: Cooldown={script_cooldown_steps}, Threshold={action_threshold}")

        env = create_fast_intraday_env_v456(df=df, env_config=env_config)
        if env is None:
            logger.error("環境作成に失敗しました")
            return
        
        # Apply FixedTTLWrapper if requested
        if use_fixed_ttl:
            from ztb.trading.environment.wrappers.fixed_ttl_wrapper import FixedTTLWrapper
            logger.info("🔧 Applying FixedTTLWrapper (Backtest mode)")
            env = FixedTTLWrapper(env, fixed_ttl=1.0)
            
        del df
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
    
    # ポジション追跡 (Environment State)
    last_info_position = info.get('position', 0.0)
    current_position = last_info_position
    position_entry_price = 0.0
    
    # Cost Accumulation for current trade
    current_trade_fees = 0.0
    current_trade_slippage = 0.0
    
    # Wrapper State
    script_cooldown_counter = 0

    while not done and step_count < steps:
        step_count += 1
        
        # モデル予測
        action, _ = model.predict(obs, deterministic=True)
        
        # --- Wrapper Control Logic ---
        # NOTE: FastIntradayEnv interprets action[0] as Target Position Fraction (-1.0 to 1.0).
        # To HOLD current position, we need to set action[0] to match current position fraction.
        max_pos = env_config.get('max_position_size', 0.1)
        current_pos_fraction = current_position / max_pos if max_pos > 0 else 0.0
        
        # 1. Action Threshold (Strength Filter)
        action_val = float(action[0]) if isinstance(action, np.ndarray) and len(action) > 0 else float(action)
        
        # If we are flatt, require strong signal to enter
        if abs(current_position) < 1e-6:
            if abs(action_val) < action_threshold:
                # Force Flat (Hold 0)
                if isinstance(action, np.ndarray):
                    action[0] = 0.0
                else:
                    action = np.array([0.0, 0.0]) # Assuming 2D action
        
        # 2. Cooldown (Frequency Control)
        if script_cooldown_counter > 0:
            # Force Hold (Keep current position)
            # This prevents entering new trades OR exiting current ones if cooldown applies
            # But usually cooldown allows EXIT, just prevents ENTRY.
            # Let's Implement: Prevent ENTRY only.
            
            if abs(current_position) < 1e-6:
                # If flat, stay flat
                if isinstance(action, np.ndarray):
                    action[0] = 0.0
                else:
                    action = np.array([0.0, 0.0])
                script_cooldown_counter -= 1
            else:
                # If we have position, we allow EXIT signals, but check logic?
                # Usually Cooldown is post-exit. So we are likely flat if counter > 0.
                # Just in case we are not flat (e.g. cooldown logic changed), 
                # we decrement counter but let model decide exit.
                script_cooldown_counter -= 1

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
            
        new_position = info.get('position', 0.0)
        
        # Cost Accumulation
        step_fee = info.get('fee_paid', 0.0)
        step_slippage = info.get('slippage_paid', 0.0)
        
        # Accumulate to active trade if we have position or just traded
        if abs(new_position) > 1e-6 or abs(current_position) > 1e-6:
             current_trade_fees += step_fee
             current_trade_slippage += step_slippage
        
        # ポートフォリオ価値計算 (Balanceはenv内で管理)
        total_value = info.get('balance', 1000000.0)
        
        # アクション評価
        reporter.update_step(step_count, total_value, action)
        
        # --- Trade Detection Logic (State Transition) ---
        if step_count > warmup_steps:
            
            # Change detection
            if abs(new_position - current_position) > 1e-6:
                
                # 1. Exit Detection (Position reduced or flipped)
                if abs(new_position) < abs(current_position) - 1e-6 or (new_position * current_position < -1e-6):
                    # Exited (partially or fully)
                    
                    exit_size = abs(current_position)
                    
                    pnl = (current_price - position_entry_price) * current_position
                    
                    direction = "long" if current_position > 0 else "short"
                    
                    # Record Trade with Accumulated Fees
                    # Note: If we flip (Long -> Short), the fee for the Short Entry part is already in current_trade_fees
                    # This logic splits the fee arbitrarily or dumps it all on the closing trade?
                    # Ideally: The fee for "closing" the long is part of the Long Trade.
                    # The fee for "opening" the short is part of the Short Trade.
                    # info['fee_paid'] includes BOTH if it was a flip in one step.
                    # Simplification: Assign ALL accumulated fees to the trade being closed.
                    # This might over-penalize the closed trade and under-penalize the new one immediately.
                    # But since we usually hold for a while, the new trade will accumulate its own exit fees later.
                    # The only error is the "Entry Fee" of the new trade being assigned to the "Exit Fee" of the old one.
                    # Given they are contiguous, the Net PnL over time is correct.
                    # For metrics precision, it's slightly off on Flips, but OK for now.
                    
                    reporter.record_trade(
                        direction, pnl, position_entry_price, current_price, exit_size,
                        current_trade_fees, current_trade_slippage
                    )
                    
                    logger.debug(f"Step {step_count}: EXIT {direction.upper()} at ¥{current_price:,.0f}, GrossPnL=¥{pnl:,.0f}, Net=¥{pnl-current_trade_fees-current_trade_slippage:,.0f}")
                    
                    # Reset accumulators
                    current_trade_fees = 0.0
                    current_trade_slippage = 0.0
                    
                    # Activate Cooldown
                    script_cooldown_counter = script_cooldown_steps

                # 2. Entry Detection (Position increased or flipped)
                if abs(new_position) > 1e-6:
                    # New position established
                    # If it's a flip, we treat it as new entry at current price
                    
                    if abs(current_position) < 1e-6 or (new_position * current_position < -1e-6):
                        # Pure Entry or Flip Entry
                        position_entry_price = current_price
                        
                        direction = "LONG" if new_position > 0 else "SHORT"
                        logger.debug(f"Step {step_count}: ENTRY {direction} {new_position:.4f}BTC @ ¥{current_price:,.0f}")
                        
                        # If Flip, we just reset fees above, so current_trade_fees is 0.
                        # BUT, we might have had a fee in this very step which belonged partially to entry.
                        # Since we dumped it all on Exit, the new trade starts with 0 cost.
                        # This is acceptable for simple Net PnL tracking.
        
        # Update State
        current_position = new_position
        
        if step_count % 1000 == 0:
            pnl_str = f"¥{(total_value - 1000000.0):,.0f}" 
            logger.info(f"Step {step_count}: Position = {current_position:.4f}BTC, Total = ¥{total_value:,.0f}")
    
    # 終了時にポジション強制決済
    if abs(current_position) > 1e-6:
        exit_price = current_price
        pnl = (exit_price - position_entry_price) * current_position
        direction = "long" if current_position > 0 else "short"
        reporter.record_trade(
            direction, pnl, position_entry_price, exit_price, abs(current_position),
            current_trade_fees, current_trade_slippage
        )
        logger.info(f"強制決済: PnL={pnl:,.0f}")

    reporter.print_summary()
    
    if output_dir is None:
        output_dir = os.path.join(project_root, "backtest_results", "v456")
    
    reporter.save_results(output_dir)
    
    return reporter.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v456 バックテスト")
    parser.add_argument("--steps", type=int, default=10000, help="バックテストステップ数")
    parser.add_argument("--warmup-steps", type=int, default=500, help="ウォームアップステップ数")
    parser.add_argument("--model-path", type=str, default=None, help="モデルパス")
    parser.add_argument("--data-path", type=str, default=None, help="データパス")
    parser.add_argument("--order-size", type=float, default=0.01, help="取引サイズ (BTC単位)")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--output", type=str, default=None, help="結果保存先ディレクトリ")
    parser.add_argument("--fixed-ttl", action="store_true", help="FixedTTLWrapperを使用する")
    
    args = parser.parse_args()
    
    run_backtest_v456(
        model_path=args.model_path,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        data_path=args.data_path,
        order_size_btc=args.order_size,
        config_path=args.config,
        output_dir=args.output,
        use_fixed_ttl=args.fixed_ttl,
    )
