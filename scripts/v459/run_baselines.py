#!/usr/bin/env python3
"""
Phase 4.5 Gate C3: ベースライン比較実験

SAC実験（P1-1/P1-3）と同一環境・同一条件で以下のベースラインを実行:
1. Random: 均等確率でBUY/SELL/HOLD
2. Buy & Hold: 全期間ロングポジション保持
3. RSI/MACD Momentum: RSI 30/70 + MACD Crossoverベース

既存実装を最大限活用:
- HeavyTradingEnv: SAC実験と同一環境
- RuleBasedBaseline (v456): RSI/MACDシグナル生成
- BaselineStrategy (ztb/analysis): フレームワーク基盤

66# (0番§5.2) 要求メトリクス:
- Net ROI, Gross PnL, Total Fees, Profit Factor, Max Drawdown
- Win Rate, Total Trades, Avg PnL/Trade, buy_count/sell_count
"""

import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# === 定数（SAC実験 run_phase45_p1.py と統一） ===
DATA_PATH = str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet")
OUTPUT_DIR = project_root / "results" / "phase45_baselines"
SEEDS = [42, 123, 456, 789]
TOTAL_STEPS = 50000  # SAC実験と同一ステップ数
INITIAL_BALANCE = 100000.0

# === 環境設定（SAC実験 create_experiment_config と同一条件） ===
# NOTE: ベースラインでは報酬は意思決定に使わないが、
#       SAC実験と環境動作を完全一致させるため同一reward_settingsを使用
REWARD_SETTINGS = {
    "use_simple_reward": True,
    "balance_penalty": 0.0,
    "position_penalty_scale": 0.0,
    "inventory_penalty_scale": 0.0,
    "trade_frequency_penalty": 0.0,
    "trade_cooldown_penalty": 0.0,
    "consecutive_trade_penalty": 0.0,
    "hold_penalty_multiplier": 1.0,
    "volatility_penalty_scale": 0.0,
    "consistency_penalty": 0.0,
    "redundant_trade_penalty": 0.0,
    "confidence_penalty_factor": 0.0,
    "balance_shaping_enabled": False,
    "action_entropy_shaping_enabled": False,
    "reward_scale": 100.0,
}


def create_env(df: pd.DataFrame, seed: int) -> HeavyTradingEnv:
    """SAC実験と同一条件でHeavyTradingEnvを生成"""
    config = EnvironmentConfig(
        transaction_cost=0.001,
        max_steps=TOTAL_STEPS,
        feature_names=list(df.columns),
        initial_portfolio_value=INITIAL_BALANCE,
        use_continuous_actions=True,
        reward_settings=RewardSettings.from_dict(REWARD_SETTINGS),
    )
    env = HeavyTradingEnv(df, config)
    return env


def extract_env_metrics(env: HeavyTradingEnv) -> Dict[str, Any]:
    """環境から全メトリクスを抽出（Gate C0: 実測値使用）"""
    pm = env.position_manager
    initial = INITIAL_BALANCE
    final = env.portfolio_value if hasattr(env, 'portfolio_value') else pm.realized_pnl + initial

    metrics = {
        "initial_balance": initial,
        "final_balance": final,
        "total_trades": pm.trades_count,
        "buy_count": pm.buy_count,
        "sell_count": pm.sell_count,
        "gross_pnl": pm.realized_pnl + pm.total_fees + pm.total_slippage,
        "total_fees": pm.total_fees,
        "total_slippage": pm.total_slippage,
        "net_pnl": final - initial,
        "realized_pnl": pm.realized_pnl,
    }

    # ROI
    metrics["net_roi"] = (final - initial) / initial * 100
    metrics["gross_roi"] = metrics["gross_pnl"] / initial * 100 if initial > 0 else 0.0

    # Avg PnL / trade
    if metrics["total_trades"] > 0:
        metrics["avg_gross_pnl_per_trade"] = metrics["gross_pnl"] / metrics["total_trades"]
        metrics["avg_fee_per_trade"] = metrics["total_fees"] / metrics["total_trades"]
    else:
        metrics["avg_gross_pnl_per_trade"] = 0.0
        metrics["avg_fee_per_trade"] = 0.0

    return metrics


# ============================================================================
# ベースライン戦略
# ============================================================================

def run_random_baseline(df: pd.DataFrame, seed: int) -> Dict[str, Any]:
    """Random Baseline: 均等確率でBUY(1)/SELL(2)/HOLD(0)"""
    env = create_env(df, seed)
    rng = np.random.RandomState(seed)

    obs, info = env.reset()
    total_reward = 0.0
    rewards = []

    for step in range(TOTAL_STEPS):
        # 連続行動空間: [-1, 1] の一様乱数
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        if terminated or truncated:
            obs, info = env.reset()

    metrics = extract_env_metrics(env)
    metrics["total_reward"] = total_reward
    metrics["strategy"] = "Random"
    metrics["seed"] = seed
    del env
    gc.collect()
    return metrics


def run_buy_and_hold_baseline(df: pd.DataFrame, seed: int) -> Dict[str, Any]:
    """Buy & Hold: 最初にBUYし、以後全てHOLD"""
    env = create_env(df, seed)
    obs, info = env.reset()
    total_reward = 0.0

    for step in range(TOTAL_STEPS):
        if step == 0:
            # 強いBUYアクション（連続値 +1.0）
            action = np.array([1.0])
        else:
            # HOLD（連続値 0.0 → 閾値以下でHOLD扱い）
            action = np.array([0.0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
            # リセット後も即座にBUY
            if step < TOTAL_STEPS - 1:
                action = np.array([1.0])
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

    metrics = extract_env_metrics(env)
    metrics["total_reward"] = total_reward
    metrics["strategy"] = "BuyAndHold"
    metrics["seed"] = seed
    del env
    gc.collect()
    return metrics


def run_momentum_baseline(df: pd.DataFrame, seed: int) -> Dict[str, Any]:
    """Momentum: RSI(14) < 30 → BUY, RSI(14) > 70 → SELL, else HOLD
    
    RSI列は特徴量parquetに含まれている（rsi_14等）。
    存在しない場合はclose列から計算。
    """
    env = create_env(df, seed)
    obs, info = env.reset()
    total_reward = 0.0

    # RSI列を特定（特徴量parquetに含まれている）
    rsi_col = None
    for col in df.columns:
        if col == 'RSI' or col == 'RSI_M1':  # 14期間RSI
            rsi_col = col
            break
    if rsi_col is None:
        for col in df.columns:
            if 'rsi' in col.lower() and '14' in col:
                rsi_col = col
                break
    
    if rsi_col is None and 'close' in df.columns:
        # RSIを計算
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_values = (100 - (100 / (1 + rs))).fillna(50).values
    elif rsi_col is not None:
        rsi_values = df[rsi_col].fillna(50).values
    else:
        # RSI計算不可 → Randomフォールバック
        logger.warning("RSI列もclose列もない → Randomにフォールバック")
        return run_random_baseline(df, seed)

    for step in range(TOTAL_STEPS):
        # 環境の現在ステップに対応するRSI値
        env_step = getattr(env, 'current_step', step)
        idx = min(env_step, len(rsi_values) - 1)
        rsi = rsi_values[idx]

        if rsi < 30:
            action = np.array([0.8])   # BUY
        elif rsi > 70:
            action = np.array([-0.8])  # SELL
        else:
            action = np.array([0.0])   # HOLD

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    metrics = extract_env_metrics(env)
    metrics["total_reward"] = total_reward
    metrics["strategy"] = "Momentum_RSI"
    metrics["seed"] = seed
    del env
    gc.collect()
    return metrics


# ============================================================================
# メイン
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Phase 4.5 Gate C3: ベースライン比較実験")
    print(f"ステップ数: {TOTAL_STEPS}, シード: {SEEDS}")
    print("戦略: Random / BuyAndHold / Momentum(RSI)")
    print("=" * 70)

    # データ読込
    logger.info(f"データ読込: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"データサイズ: {df.shape}")

    strategies = [
        ("Random", run_random_baseline),
        ("BuyAndHold", run_buy_and_hold_baseline),
        ("Momentum_RSI", run_momentum_baseline),
    ]

    all_results = []
    total_start = time.time()

    for strategy_name, run_fn in strategies:
        print(f"\n{'='*60}")
        print(f"戦略: {strategy_name}")
        print(f"{'='*60}")

        for seed in SEEDS:
            start = time.time()
            print(f"  seed={seed} 実行中...", end="", flush=True)

            try:
                result = run_fn(df, seed)
                elapsed = time.time() - start
                result["total_time_seconds"] = elapsed
                result["success"] = True
                print(f" ✅ ({elapsed:.0f}秒) ROI: {result['net_roi']:.2f}%, "
                      f"Trades: {result['total_trades']}, "
                      f"Gross: {result['gross_pnl']:+,.0f}")
            except Exception as e:
                elapsed = time.time() - start
                result = {
                    "strategy": strategy_name,
                    "seed": seed,
                    "success": False,
                    "error": str(e),
                    "total_time_seconds": elapsed,
                }
                print(f" ❌ ({elapsed:.0f}秒) Error: {e}")

            all_results.append(result)

    total_elapsed = time.time() - total_start

    # === サマリー出力 ===
    print("\n" + "=" * 70)
    print("📊 BASELINE RESULTS SUMMARY")
    print("=" * 70)

    for strategy_name, _ in strategies:
        cat_results = [r for r in all_results
                       if r.get("strategy") == strategy_name and r.get("success")]
        if not cat_results:
            print(f"\n{strategy_name}: 全失敗")
            continue

        rois = [r["net_roi"] for r in cat_results]
        gross_pnls = [r["gross_pnl"] for r in cat_results]
        fees = [r["total_fees"] for r in cat_results]
        trades = [r["total_trades"] for r in cat_results]
        buys = [r["buy_count"] for r in cat_results]
        sells = [r["sell_count"] for r in cat_results]

        print(f"\n{strategy_name} (n={len(cat_results)}):")
        print(f"  Net ROI:   {np.mean(rois):+.2f}% ± {np.std(rois):.2f}%")
        print(f"  Gross PnL: {np.mean(gross_pnls):+,.0f} ± {np.std(gross_pnls):,.0f}")
        print(f"  Fees:      {np.mean(fees):,.0f}")
        print(f"  Trades:    {np.mean(trades):.0f} (BUY: {np.mean(buys):.0f}, SELL: {np.mean(sells):.0f})")
        if np.mean(trades) > 0:
            print(f"  粗利/取引: {np.mean(gross_pnls)/np.mean(trades):+.2f} JPY")
            print(f"  手数料/取引: {np.mean(fees)/np.mean(trades):.2f} JPY")

    # === 結果保存 ===
    results_file = OUTPUT_DIR / f"baseline_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_elapsed_seconds": total_elapsed,
            "total_steps": TOTAL_STEPS,
            "seeds": SEEDS,
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Results saved: {results_file}")
    print(f"Total time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
