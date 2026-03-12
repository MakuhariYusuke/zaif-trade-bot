#!/usr/bin/env python3
"""
Week 4: Risk-Return Profile Analysis
リスク・リターンプロファイルの詳細分析
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from stable_baselines3 import SAC


def calculate_risk_return_metrics(pnls: np.ndarray, rewards: np.ndarray, 
                                   episode_lengths: np.ndarray) -> dict:
    """リスク・リターンメトリクス計算"""
    
    # Return metrics
    total_return = pnls.sum()
    avg_return_per_episode = pnls.mean()
    return_std = pnls.std()
    return_skewness = stats.skew(pnls)
    return_kurtosis = stats.kurtosis(pnls)
    
    # Sortino ratio (downside risk only)
    downside_returns = np.minimum(pnls, 0)
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    sortino = avg_return_per_episode / (downside_std + 1e-8)
    
    # Maximum drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = total_return / (abs(max_dd) * 100000 + 1e-8) if max_dd < 0 else 0
    
    # Win rate and profit factor
    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls) if len(pnls) > 0 else 0
    
    total_wins = pnls[pnls > 0].sum()
    total_losses = abs(pnls[pnls < 0].sum())
    profit_factor = total_wins / (total_losses + 1e-8) if total_losses > 0 else np.inf
    
    # Reward metrics
    reward_mean = rewards.mean()
    reward_std = rewards.std()
    reward_sharpe = reward_mean / (reward_std + 1e-8)
    
    # Efficiency (return per step)
    steps_per_episode = episode_lengths.mean()
    efficiency = avg_return_per_episode / (steps_per_episode + 1e-8)
    
    return {
        "return_metrics": {
            "total_return": float(total_return),
            "avg_per_episode": float(avg_return_per_episode),
            "std": float(return_std),
            "skewness": float(return_skewness),
            "kurtosis": float(return_kurtosis),
        },
        "risk_metrics": {
            "max_drawdown": float(max_dd),
            "downside_std": float(downside_std),
            "var_95": float(np.percentile(pnls, 5)),  # 95% VaR
            "cvar_95": float(pnls[pnls <= np.percentile(pnls, 5)].mean()),  # CVaR
        },
        "risk_adjusted_returns": {
            "sortino": float(sortino),
            "calmar": float(calmar),
            "reward_sharpe": float(reward_sharpe),
        },
        "profitability": {
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor) if profit_factor != np.inf else "unlimited",
            "avg_win": float(pnls[pnls > 0].mean()) if len(pnls[pnls > 0]) > 0 else 0,
            "avg_loss": float(pnls[pnls < 0].mean()) if len(pnls[pnls < 0]) > 0 else 0,
        },
        "efficiency": {
            "return_per_step": float(efficiency),
            "steps_per_episode": float(steps_per_episode),
            "trades_per_episode": float(100000 * efficiency / (steps_per_episode + 1e-8)),
        }
    }


def analyze_risk_return_profile(model_path: str, n_episodes: int = 100) -> dict:
    """Risk-Return分析"""
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'btc_jpy_1m_v454.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    env = FastIntradayEnvV456(
        data=df,
        initial_balance=100000,
        max_position=0.01,
        fee_rate=0.001,
        slippage_rate=0.0005,
        drawdown_limit=0.3,
    )
    
    try:
        model = SAC.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {}
    
    pnls = []
    rewards = []
    episode_lengths = []
    equity_curves = []
    
    print(f"Analyzing risk-return profile over {n_episodes} episodes...")
    print("-" * 80)
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        initial_balance = env.balance
        ep_reward = 0.0
        ep_length = 0
        equity_curve = [initial_balance]
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            equity_curve.append(info.get('balance', env.balance))
        
        final_balance = env.balance
        pnl = final_balance - initial_balance
        
        pnls.append(pnl)
        rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        equity_curves.append(equity_curve)
        
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1:3d}: PnL={pnl:+.0f}, Length={ep_length}, Reward={ep_reward:.4f}")
    
    pnls_arr = np.array(pnls)
    rewards_arr = np.array(rewards)
    lengths_arr = np.array(episode_lengths)
    
    metrics = calculate_risk_return_metrics(pnls_arr, rewards_arr, lengths_arr)
    
    # Percentile analysis
    percentiles = [10, 25, 50, 75, 90]
    metrics["pnl_percentiles"] = {
        f"p{p}": float(np.percentile(pnls_arr, p)) for p in percentiles
    }
    
    return metrics


def print_risk_return_report(metrics: dict):
    """Risk-Returnレポート"""
    
    print("\n" + "=" * 80)
    print("RISK-RETURN PROFILE ANALYSIS")
    print("=" * 80)
    
    ret = metrics.get("return_metrics", {})
    print("\n💰 Return Metrics")
    print("-" * 80)
    print(f"  Total Return:        {ret.get('total_return', 0):+,.0f} JPY")
    print(f"  Mean per Episode:    {ret.get('avg_per_episode', 0):+,.0f} JPY")
    print(f"  Std Dev:             {ret.get('std', 0):,.0f} JPY")
    print(f"  Skewness:            {ret.get('skewness', 0):>6.2f}")
    print(f"  Kurtosis:            {ret.get('kurtosis', 0):>6.2f}")
    
    risk = metrics.get("risk_metrics", {})
    print("\n⚠️  Risk Metrics")
    print("-" * 80)
    print(f"  Max Drawdown:        {risk.get('max_drawdown', 0):>6.2%}")
    print(f"  Downside Std:        {risk.get('downside_std', 0):>10,.0f} JPY")
    print(f"  VaR (95%):          {risk.get('var_95', 0):>10,.0f} JPY")
    print(f"  CVaR (95%):         {risk.get('cvar_95', 0):>10,.0f} JPY")
    
    rar = metrics.get("risk_adjusted_returns", {})
    print("\n📊 Risk-Adjusted Returns")
    print("-" * 80)
    print(f"  Sortino Ratio:       {rar.get('sortino', 0):>10.2f}")
    print(f"  Calmar Ratio:        {rar.get('calmar', 0):>10.2f}")
    print(f"  Reward Sharpe:       {rar.get('reward_sharpe', 0):>10.2f}")
    
    prof = metrics.get("profitability", {})
    print("\n🎯 Profitability")
    print("-" * 80)
    print(f"  Win Rate:            {prof.get('win_rate', 0):>6.1%}")
    profit_factor = prof.get('profit_factor', 0)
    if isinstance(profit_factor, str):
        print(f"  Profit Factor:       {profit_factor}")
    else:
        print(f"  Profit Factor:       {profit_factor:>6.2f}")
    print(f"  Avg Win:             {prof.get('avg_win', 0):>10,.0f} JPY")
    print(f"  Avg Loss:            {prof.get('avg_loss', 0):>10,.0f} JPY")
    
    eff = metrics.get("efficiency", {})
    print("\n⚡ Efficiency")
    print("-" * 80)
    print(f"  Return per Step:     {eff.get('return_per_step', 0):>10,.1f} JPY/step")
    print(f"  Avg Steps/Episode:   {eff.get('steps_per_episode', 0):>10.1f}")
    
    perc = metrics.get("pnl_percentiles", {})
    print("\n📈 PnL Percentiles")
    print("-" * 80)
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {perc.get(f'p{p}', 0):>10,.0f} JPY")
    
    print("\n" + "=" * 80)


def save_results(metrics: dict, output_file: str = "risk_return_metrics.json"):
    """Save to JSON"""
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    model_path = "models/week4_fixed/sac_model" if len(sys.argv) < 2 else sys.argv[1]
    
    print("=" * 80)
    print("Risk-Return Profile Analysis")
    print("=" * 80)
    
    metrics = analyze_risk_return_profile(model_path, n_episodes=100)
    
    if metrics:
        print_risk_return_report(metrics)
        save_results(metrics, "results/week4_risk_return_metrics.json")
