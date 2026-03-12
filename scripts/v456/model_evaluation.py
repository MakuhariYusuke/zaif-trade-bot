#!/usr/bin/env python3
"""
Week 4: Comprehensive Metrics Collection
訓練済みモデルの詳細なメトリクス評価

Phase 1.4: 設定統一を反映
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.config.environment_config import get_evaluation_config
from stable_baselines3 import SAC

CONFIG = get_evaluation_config()


def evaluate_trained_model(model_path: str, n_episodes: int = 50) -> dict:
    """訓練済みモデルの包括的評価"""
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'btc_jpy_1m_v454.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Prepare features
    df_copy = df.copy()
    base_cols = [f'base_{i}' for i in range(30)]
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    # 特徴量検証: 欠損列はエラーで検出
    missing_base = [col for col in base_cols if col not in df_copy.columns]
    missing_mtf = [col for col in mtf_cols if col not in df_copy.columns]
    missing_regime = [col for col in regime_cols if col not in df_copy.columns]
    
    if missing_base or missing_mtf or missing_regime:
        error_msg = "❌ Missing feature columns for evaluation:\n"
        if missing_base:
            error_msg += f"  Base: {missing_base}\n"
        if missing_mtf:
            error_msg += f"  MTF: {missing_mtf}\n"
        if missing_regime:
            error_msg += f"  Regime: {missing_regime}\n"
        error_msg += f"\nAvailable columns ({len(df_copy.columns)}): {df_copy.columns.tolist()}\n"
        raise ValueError(error_msg)
    
    print("✓ All required feature columns present")
    
    for col in ['atr', 'impact_proxy']:
        if col not in df_copy.columns:
            print(f"⚠️  Optional column '{col}' not found. Creating placeholder.")
            df_copy[col] = np.ones(len(df_copy))
    
    env = FastIntradayEnvV456(
        df=df_copy,
        base_feature_columns=base_cols[:30],
        mtf_feature_columns=mtf_cols[:27],
        regime_feature_columns=regime_cols[:13],
        initial_balance=CONFIG.INITIAL_BALANCE,
        max_position=CONFIG.MAX_POSITION,
        max_steps=CONFIG.MAX_STEPS,
        drawdown_limit=0.3,
    )
    
    # モデル読み込み
    try:
        model = SAC.load(model_path, env=env)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return {}
    
    results = {
        "episode_metrics": [],
        "reward_distribution": defaultdict(list),
        "action_distribution": {"SELL": 0, "HOLD": 0, "BUY": 0},
        "episode_lengths": [],
        "final_balances": [],
        "pnls": [],
        "max_drawdowns": [],
        "sharpe_ratios": [],
    }
    
    print(f"\nEvaluating {n_episodes} episodes...")
    print("-" * 80)
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        ep_length = 0
        ep_reward = 0.0
        initial_balance = env.balance
        max_balance = initial_balance
        min_balance = initial_balance
        rewards = []
        actions = []
        balances = [initial_balance]
        
        while not (done or truncated):
            # Model prediction
            action, _states = model.predict(obs, deterministic=True)
            # Ensure action is in correct format
            if isinstance(action, (int, np.integer)):
                action = float(action)
            
            obs, reward, done, truncated, info = env.step(action)
            
            ep_length += 1
            ep_reward += reward
            rewards.append(reward)
            actions.append(action)
            
            balance = info.get('balance', env.balance)
            balances.append(balance)
            max_balance = max(max_balance, balance)
            min_balance = min(min_balance, balance)
            
            results["reward_distribution"][f"step_{min(ep_length, 10)}"].append(reward)
            
            # Action tracking - normalize action to scalar
            action_scalar = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            if action_scalar < -0.33:
                results["action_distribution"]["SELL"] += 1
            elif action_scalar < 0.33:
                results["action_distribution"]["HOLD"] += 1
            else:
                results["action_distribution"]["BUY"] += 1
        
        # Episode metrics
        final_balance = env.balance
        pnl = final_balance - initial_balance
        max_dd = (min_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
        
        # Sharpe ratio calculation
        if len(rewards) > 1:
            ret = np.array(rewards)
            sharpe = (np.mean(ret) / (np.std(ret) + 1e-8)) * np.sqrt(252 * 10)  # Annualized
        else:
            sharpe = 0
        
        results["episode_metrics"].append({
            "episode": ep,
            "length": ep_length,
            "reward": ep_reward,
            "final_balance": final_balance,
            "pnl": pnl,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        })
        
        results["episode_lengths"].append(ep_length)
        results["final_balances"].append(final_balance)
        results["pnls"].append(pnl)
        results["max_drawdowns"].append(max_dd)
        results["sharpe_ratios"].append(sharpe)
        
        # Progress report
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: Len={ep_length}, "
                  f"PnL={pnl:+.0f}, MDD={max_dd:.2%}, Sharpe={sharpe:.2f}")
    
    return results


def print_metrics_report(results: dict):
    """詳細なメトリクスレポート"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS REPORT")
    print("=" * 80)
    
    # Episode Length Analysis
    print("\n📊 Episode Length Distribution")
    print("-" * 80)
    ep_lens = np.array(results["episode_lengths"])
    print(f"  Mean:   {ep_lens.mean():.1f} steps")
    print(f"  Std:    {ep_lens.std():.1f}")
    print(f"  Min:    {ep_lens.min()}")
    print(f"  Max:    {ep_lens.max()}")
    print(f"  Median: {np.median(ep_lens):.1f}")
    print(f"  P25:    {np.percentile(ep_lens, 25):.1f}")
    print(f"  P75:    {np.percentile(ep_lens, 75):.1f}")
    
    # Financial Metrics
    print("\n💰 Financial Performance")
    print("-" * 80)
    pnls = np.array(results["pnls"])
    balances = np.array(results["final_balances"])
    print(f"  PnL Mean:      {pnls.mean():+.0f} JPY")
    print(f"  PnL Std:       {pnls.std():.0f} JPY")
    print(f"  PnL Min:       {pnls.min():+.0f} JPY")
    print(f"  PnL Max:       {pnls.max():+.0f} JPY")
    print(f"  Win Rate:      {(pnls > 0).sum() / len(pnls) * 100:.1f}%")
    print(f"  Avg Final Balance: {balances.mean():,.0f} JPY")
    
    # Risk Metrics
    print("\n⚠️  Risk Metrics")
    print("-" * 80)
    mdds = np.array(results["max_drawdowns"])
    sharpes = np.array(results["sharpe_ratios"])
    print(f"  Max Drawdown Mean:  {mdds.mean():.2%}")
    print(f"  Max Drawdown Worst: {mdds.min():.2%}")
    print(f"  Sharpe Ratio Mean:  {sharpes.mean():.2f}")
    print(f"  Sharpe Ratio Std:   {sharpes.std():.2f}")
    
    # Action Distribution
    print("\n🎯 Action Distribution")
    print("-" * 80)
    total_actions = sum(results["action_distribution"].values())
    for action, count in results["action_distribution"].items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action:<8}: {count:>6} ({pct:>6.1f}%)")
    
    # Reward Distribution by Step
    print("\n📈 Reward by Step Position")
    print("-" * 80)
    for step_key in sorted(results["reward_distribution"].keys())[:5]:
        rewards = np.array(results["reward_distribution"][step_key])
        if len(rewards) > 0:
            print(f"  {step_key}: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
                  f"min={rewards.min():.4f}, max={rewards.max():.4f}")
    
    print("\n" + "=" * 80)


def save_results(results: dict, output_file: str = "metrics_report.json"):
    """結果をJSONで保存"""
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    import sys
    import glob
    
    # Find latest model automatically
    if len(sys.argv) < 2:
        model_files = glob.glob("models/**/sac_*v456_*.zip", recursive=True)
        model_files.extend(glob.glob("models/**/sac_*fixed*.zip", recursive=True))
        if model_files:
            model_path = sorted(model_files, key=lambda p: Path(p).stat().st_mtime)[-1]
            print(f"📦 Found latest model: {model_path}")
        else:
            print("❌ No models found!")
            sys.exit(1)
    else:
        model_path = sys.argv[1]
    
    print("=" * 80)
    print("Model Evaluation Suite")
    print("=" * 80)
    
    results = evaluate_trained_model(model_path, n_episodes=50)
    
    if results:
        print_metrics_report(results)
        save_results(results, "results/week4_evaluation_metrics.json")
