#!/usr/bin/env python3
"""
v456 バックテストスクリプト (Modified for v457)
Moved to scripts/v457/backtest_v457.py
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
from gymnasium import spaces

# プロジェクト PATH 設定
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import (
    load_config_dict,
    extract_env_config,
    extract_seed,
)
from ztb.utils.seed_manager import set_global_seed
from utils.results_utils import save_backtest_results

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
            "ttl_action_distribution": {},
            "raw_action_sum": 0.0,
            "raw_action_count": 0,
            "abs_action_sum": 0.0,
            "ttl_action_sum": 0.0,
            "ttl_action_count": 0,
            "avg_abs_action": 0.0,
            "avg_ttl_action": 0.0,
            "profit_factor": 0.0,
            "ttl_forced_exits": 0,
            "cooldown_triggers": 0,
            "ttl_enabled": None,
            "start_index": None,
            "seed": None,
            "baseline_mode": None,
            "reward_scale": None,
            "reward_clip": None,
        }
        self.portfolio_history = []
        self.trade_history = []

    @staticmethod
    def _bucket_ttl_action(ttl_value: float) -> str:
        ttl_value = max(0.0, min(ttl_value, 1.0))
        bucket_index = min(int(ttl_value * 5), 4)
        low = bucket_index * 0.2
        high = low + 0.2
        return f"{low:.1f}-{high:.1f}"

    def update_step(self, step, portfolio_value, action, env_info=None):
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

        if isinstance(action, np.ndarray) and action.size > 1:
            ttl_val = float(action[1])
            ttl_key = self._bucket_ttl_action(ttl_val)
            self.stats["ttl_action_distribution"][ttl_key] = (
                self.stats["ttl_action_distribution"].get(ttl_key, 0) + 1
            )
            self.stats["ttl_action_sum"] += ttl_val
            self.stats["ttl_action_count"] += 1
        
        # Action Strength Stats
        self.stats["raw_action_sum"] += action_val
        self.stats["abs_action_sum"] += abs(action_val)
        self.stats["raw_action_count"] += 1

        if isinstance(env_info, dict):
            ttl_forced = env_info.get("ttl_forced_exits")
            cooldown_triggers = env_info.get("cooldown_triggers")
            if ttl_forced is not None:
                self.stats["ttl_forced_exits"] = int(ttl_forced)
            if cooldown_triggers is not None:
                self.stats["cooldown_triggers"] = int(cooldown_triggers)
            if "ttl_enabled" in env_info:
                self.stats["ttl_enabled"] = bool(env_info.get("ttl_enabled"))

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

        # Action Strength
        if self.stats["raw_action_count"] > 0:
            self.stats["avg_abs_action"] = self.stats["abs_action_sum"] / self.stats["raw_action_count"]
        else:
            self.stats["avg_abs_action"] = 0.0

        if self.stats["ttl_action_count"] > 0:
            self.stats["avg_ttl_action"] = self.stats["ttl_action_sum"] / self.stats["ttl_action_count"]
        else:
            self.stats["avg_ttl_action"] = 0.0

        # Profit Factor (Net PnL)
        gross_profit = sum(t['net_pnl'] for t in self.trade_history if t['net_pnl'] > 0)
        gross_loss = sum(abs(t['net_pnl']) for t in self.trade_history if t['net_pnl'] < 0)
        self.stats["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Additional metrics
        winning_trades = [t for t in self.trade_history if t['net_pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['net_pnl'] < 0]
        self.stats["avg_win"] = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0.0
        self.stats["avg_loss"] = np.mean([abs(t['net_pnl']) for t in losing_trades]) if losing_trades else 0.0
        self.stats["expectancy"] = self.stats["avg_win"] * (len(winning_trades) / len(self.trade_history)) - self.stats["avg_loss"] * (len(losing_trades) / len(self.trade_history)) if self.trade_history else 0.0
        # Trades per day (assuming 1m data, 1440 min/day)
        total_days = len(self.portfolio_history) / 1440
        self.stats["trades_per_day"] = self.stats["total_trades"] / total_days if total_days > 0 else 0.0

    def print_summary(self):
        """結果サマリー出力"""
        print("\n" + "=" * 70)
        print("🎯 BACKTEST REPORT - v457")
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
        
        avg_abs_action = self.stats.get("avg_abs_action", 0.0)
        print(f"  平均アクション強度 (|Action|): {avg_abs_action:.4f}")

        ttl_distribution = self.stats.get("ttl_action_distribution", {})
        if ttl_distribution:
            print(f"\n⌛ TTL アクション分布:")
            total_ttl_actions = sum(ttl_distribution.values())
            for k, v in sorted(ttl_distribution.items()):
                pct = (v / total_ttl_actions * 100) if total_ttl_actions > 0 else 0
                print(f"  {k}: {v} ({pct:.1f}%)")
            avg_ttl_action = self.stats.get("avg_ttl_action", 0.0)
            print(f"  平均 TTL アクション: {avg_ttl_action:.4f}")

        print(f"  TTL Forced Exits: {self.stats.get('ttl_forced_exits', 0)}")
        print(f"  Cooldown Triggers: {self.stats.get('cooldown_triggers', 0)}")
        
        print(f"\n📊 高度指標:")
        print(f"  Profit Factor: {self.stats.get('profit_factor', 0.0):.2f}")
        print(f"  Expectancy: ¥{self.stats.get('expectancy', 0.0):,.0f}")
        print(f"  Avg Win: ¥{self.stats.get('avg_win', 0.0):,.0f}")
        print(f"  Avg Loss: ¥{self.stats.get('avg_loss', 0.0):,.0f}")
        print(f"  Trades/Day: {self.stats.get('trades_per_day', 0.0):.2f}")
        if self.stats['total_trades'] > 0:
            avg_return = self.stats['net_pnl'] / self.stats['total_trades']
            print(f"  Avg Net PnL/Trade: ¥{avg_return:,.0f}")
        
        print("=" * 70)

    def save_results(self, results_dir):
        """結果をファイルに保存"""
        os.makedirs(results_dir, exist_ok=True)

        # Standardized outputs (results_utils)
        metadata = {
            "seed": self.stats.get("seed"),
            "start_index": self.stats.get("start_index"),
            "ttl_enabled": self.stats.get("ttl_enabled"),
            "model_path": self.stats.get("model_path"),
            "data_path": self.stats.get("data_path"),
            "action_space_type": self.stats.get("action_space_type"),
            "config_path": self.stats.get("config_path"),
        }
        save_backtest_results(
            portfolio_values=self.portfolio_history,
            trade_history=self.trade_history,
            metrics=self.stats,
            output_dir=results_dir,
            filename_prefix="backtest",
            metadata=metadata,
        )
        
        # Legacy outputs for compatibility
        pd.DataFrame({
            "portfolio_value": self.portfolio_history
        }).to_csv(os.path.join(results_dir, "portfolio_history.csv"), index=False)
        
        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
        with open(os.path.join(results_dir, "trades.json"), "w") as f:
            def convert(o):
                if isinstance(o, (np.int64, np.int32)):
                    return int(o)
                if isinstance(o, (np.float64, np.float32)):
                    return float(o)
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
    seed: Optional[int] = None,
    baseline_mode: Optional[str] = None,
):
    """v456 バックテスト実行"""
    
    # Config Loading
    env_config_override = {}
    seed_from_config = None
    if config_path and os.path.exists(config_path):
        logger.info(f"Config読み込み: {config_path}")
        full_config = load_config_dict(Path(config_path))
        env_config_override = extract_env_config(full_config)
        seed_from_config = extract_seed(full_config)
        
        # Merge some top-level config if useful
        if "min_delta" in full_config.get("training", {}):
             env_config_override["min_delta"] = full_config["training"]["min_delta"]
        # Also check root level or specific overrides needed
        if "min_delta" in full_config:
             env_config_override["min_delta"] = full_config["min_delta"]

        logger.info(f"  Env Config Override: {list(env_config_override.keys())}")
        if "min_delta" in env_config_override:
            logger.info(f"  min_delta applied: {env_config_override['min_delta']}")

    if seed is None:
        seed = seed_from_config
    if seed is not None:
        set_global_seed(seed)
        logger.info(f"Seed fixed: {seed}")

    # 1. データ読み込み
    if data_path is None:
        # Default data
        data_path = project_root / "data" / "btc_jpy_1m_v451.csv"
    
    data_path = Path(data_path)
    if not data_path.exists():
        # Try finding relative to project root if absolute path check fails
        if not data_path.is_absolute():
            data_path = project_root / data_path
            
    if not data_path.exists():
        logger.error(f"データが見つかりません: {data_path}")
        return
    
    logger.info(f"データ読み込み: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"  行数: {len(df)}")
    
    # Apply validation split from config
    if config_path and os.path.exists(config_path):
        full_config = load_config_dict(Path(config_path))
        data_config = full_config.get("data", {})
        validation_start = data_config.get("validation_start", 0)
        validation_end = data_config.get("validation_end", len(df))
        df = df.iloc[validation_start:validation_end].reset_index(drop=True)
        logger.info(f"Applied validation split: rows {validation_start} to {validation_end} ({len(df)} rows)")
    
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
        
        # Check if 1d_position is specified in config override
        if env_config_override.get("environment", {}).get("action_space_type") == "1d_position":
             env_config["action_space_type"] = "1d_position"
        elif env_config_override.get("action_space_type") == "1d_position":
             env_config["action_space_type"] = "1d_position"

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
    
    # 3. モデル読み込み (baseline_mode 時はスキップ)
    model = None
    if baseline_mode is None:
        if model_path is None:
            # 最新の訓練済みモデルを探す
            import glob
            model_search_patterns = [
                str(project_root / "models" / "v457_3" / "final" / "*.zip"),
                str(project_root / "models" / "v457_2" / "final" / "*.zip"),
            ]
            
            found_models = []
            for pat in model_search_patterns:
                found_models.extend(glob.glob(pat))
                
            if found_models:
                # Sort by creation time
                model_path = max(found_models, key=os.path.getctime)
                logger.info(f"モデル自動検出: {model_path}")
            else:
                logger.error("モデルが見つかりません。--model-path を指定してください。")
                return

        model_path = Path(model_path)
        if not model_path.exists():
             if not model_path.is_absolute():
                 model_path = project_root / model_path
                 
        if not model_path.exists():
            logger.error(f"モデルファイルが見つかりません: {model_path}")
            return

        logger.info(f"モデル読み込み: {model_path}")
        try:
            model = SAC.load(model_path, env=env)
        except Exception as e:
            logger.error(f"モデルロードエラー ({e})")
            # Retry with custom_objects if needed or just fail
            return
    else:
        logger.info(f"Baseline mode enabled: {baseline_mode}")

    # 4. バックテストループ
    logger.info(f"\n🚀 バックテスト開始: {steps} ステップ")
    
    if seed is not None:
        obs, info = env.reset(seed=seed)
        logger.info(f"Env reset: start_index={info.get('start_index')}")
    else:
        obs, info = env.reset()
    done = False
    
    reporter = BacktestReporter()
    reporter.stats["seed"] = seed
    reporter.stats["start_index"] = info.get("start_index") if isinstance(info, dict) else None
    reporter.stats["model_path"] = str(model_path) if model_path else None
    reporter.stats["data_path"] = str(data_path)
    reporter.stats["config_path"] = str(config_path) if config_path else None
    reporter.stats["action_space_type"] = getattr(
        env.unwrapped if hasattr(env, "unwrapped") else env,
        "action_space_type",
        env_config.get("action_space_type") if isinstance(env_config, dict) else None,
    )
    reporter.stats["baseline_mode"] = baseline_mode
    target_env = env.unwrapped if hasattr(env, "unwrapped") else env
    reporter.stats["reward_scale"] = getattr(target_env, "reward_scale", None)
    reporter.stats["reward_clip"] = getattr(target_env, "reward_clip", None)
    
    current_position = 0.0
    position_entry_price = 0.0
    
    current_trade_fees = 0.0
    current_trade_slippage = 0.0
    
    script_cooldown_counter = 0

    for step_count in range(1, steps + 1):
        if baseline_mode is None:
            action, _states = model.predict(obs, deterministic=True)
        else:
            baseline_target = 0.0
            if baseline_mode == "long":
                baseline_target = 1.0
            elif baseline_mode == "short":
                baseline_target = -1.0
            elif baseline_mode == "flat":
                baseline_target = 0.0
            if isinstance(env.action_space, spaces.Box) and env.action_space.shape == (2,):
                action = np.array([baseline_target, 1.0], dtype=np.float32)
            else:
                action = np.array([baseline_target], dtype=np.float32)
        
        # Script-side Wrapper Logic (similar to Training Env wrappers)
        # Note: If FixedTTLWrapper is applied to env, 'action' here is 1D (target_pos).
        # We need to handle 1D or 2D depending on wrapper.
        
        # To HOLD current position, we need to set action[0] to match current position fraction.
        max_pos = env_config.get('max_position_size', 0.1)
        current_pos_fraction = current_position / max_pos if max_pos > 0 else 0.0
        
        # Action Value Extraction
        if isinstance(action, np.ndarray):
            pos_action_val = float(action[0]) if action.size > 0 else 0.0
        else:
            pos_action_val = float(action)
        
        # 1. Action Threshold (Strength Filter)
        # If we are flatt, require strong signal to enter
        if abs(current_position) < 1e-6:
            if abs(pos_action_val) < action_threshold:
                # Force Flat (Hold 0)
                if isinstance(action, np.ndarray):
                    action[0] = 0.0
                else:
                    # If 1D scalar/array
                    if isinstance(action, np.ndarray):
                        action = np.array([0.0])
                    else:
                        action = 0.0

        # 2. Cooldown (Frequency Control)
        if script_cooldown_counter > 0:
            if abs(current_position) < 1e-6:
                # If flat, stay flat
                if isinstance(action, np.ndarray):
                    action[0] = 0.0
                else:
                    if isinstance(action, np.ndarray):
                        action = np.array([0.0])
                    else:
                        action = 0.0
                script_cooldown_counter -= 1
            else:
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
        
        new_position = info.get('position', 0.0)
        
        # Cost Accumulation
        step_fee = info.get('fee_paid', 0.0)
        step_slippage = info.get('slippage_paid', 0.0)
        
        if abs(new_position) > 1e-6 or abs(current_position) > 1e-6:
             current_trade_fees += step_fee
             current_trade_slippage += step_slippage
        
        total_value = info.get('portfolio_value', info.get('balance', 1000000.0))
        
        reporter.update_step(step_count, total_value, action, info)
        
        # --- Trade Detection Logic ---
        if step_count > warmup_steps:
             if abs(new_position - current_position) > 1e-6:
                # 1. Exit Detection
                if abs(new_position) < abs(current_position) - 1e-6 or (new_position * current_position < -1e-6):
                    exit_size = abs(current_position)
                    pnl = (current_price - position_entry_price) * current_position
                    direction = "long" if current_position > 0 else "short"
                    
                    reporter.record_trade(
                        direction, pnl, position_entry_price, current_price, exit_size,
                        current_trade_fees, current_trade_slippage
                    )
                    
                    current_trade_fees = 0.0
                    current_trade_slippage = 0.0
                    script_cooldown_counter = script_cooldown_steps

                # 2. Entry Detection
                if abs(new_position) > 1e-6:
                    if abs(current_position) < 1e-6 or (new_position * current_position < -1e-6):
                        position_entry_price = current_price

        # Update State
        current_position = new_position
        
        if step_count % 1000 == 0:
            logger.info(f"Step {step_count}: Position = {current_position:.4f}BTC, Total = ¥{total_value:,.0f}")
    
    # 終了時にポジション強制決済
    if abs(current_position) > 1e-6:
        exit_price = current_price if 'current_price' in locals() else df.iloc[-1]['close']
        pnl = (exit_price - position_entry_price) * current_position
        direction = "long" if current_position > 0 else "short"
        reporter.record_trade(
            direction, pnl, position_entry_price, exit_price, abs(current_position),
            current_trade_fees, current_trade_slippage
        )
        logger.info(f"強制決済: PnL={pnl:,.0f}")

    reporter.finalize_stats()
    reporter.print_summary()
    
    if output_dir is None:
        output_dir = project_root / "backtest_results" / "v457"
    
    reporter.save_results(output_dir)
    return reporter.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v457 Backtest")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--order-size", type=float, default=0.01)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fixed-ttl", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["long", "short", "flat"],
        default=None,
        help="Run a baseline policy (override model actions).",
    )
    
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
        seed=args.seed,
        baseline_mode=args.baseline,
    )
