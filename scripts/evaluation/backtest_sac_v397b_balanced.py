"""
SAC v397b_balanced Backtest Script
改善した報酬設定の効果を検証

検証ポイント:
1. HOLD比率: 92.1% → 60%以下に改善
2. 取引回数: 0回 → 50回以上に改善
3. 収益率: 0% → プラス収益を達成
4. 報酬分布: 100%ネガティブ → ポジティブ報酬の出現
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.environment.environment import HeavyTradingEnv


def run_backtest(
    model_path: str,
    data_path: str,
    max_steps: int = 5000
) -> Dict[str, Any]:
    """バックテスト実行"""
    
    print("=" * 80)
    print("SAC v397b_balanced Backtest")
    print("=" * 80)
    
    # データロード
    df = pd.read_csv(data_path)
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)
    
    # 環境作成（v397b設定）
    config = {
        "initial_balance": 100000,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.15,
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 200.0,
            "reward_clip_min": -2.0,
            "reward_clip_max": 2.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.001,
            "inactivity_penalty_window": 3,
            "inactivity_hold_threshold": 0.05,
            "enable_opportunity_cost": False,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.05,
            "trade_execution_position_threshold": 0.01,
            "trade_execution_action_multiplier": 1.5
        }
    }
    
    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    
    # モデルロード
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path)
    
    # バックテスト実行
    print(f"\nRunning backtest ({len(df)} steps)...")
    obs, _ = env.reset()
    
    action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    rewards = []
    positions = []
    position_changes = []
    
    step = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 統計収集
        rewards.append(reward)
        
        # ポジション追跡
        if hasattr(env, 'position'):
            positions.append(env.position)
            if len(positions) > 1:
                position_changes.append(abs(positions[-1] - positions[-2]))
        
        # 連続アクションから離散アクションへ変換
        from ztb.trading.environment.constants import continuous_to_discrete_action
        if isinstance(action, np.ndarray):
            continuous_value = action.item()
        else:
            continuous_value = action
        
        discrete_action = continuous_to_discrete_action(continuous_value, threshold=0.15)
        
        if discrete_action == 0:
            action_counts["BUY"] += 1
        elif discrete_action == 1:
            action_counts["HOLD"] += 1
        elif discrete_action == 2:
            action_counts["SELL"] += 1
        
        step += 1
        if step % 1000 == 0:
            print(f"Step {step}/{len(df)}")
    
    # 最終統計
    total_actions = sum(action_counts.values())
    action_distribution = {
        k: (v / total_actions * 100) if total_actions > 0 else 0
        for k, v in action_counts.items()
    }
    
    # 最終残高の取得（報酬の累積から推定）
    initial_balance = 100000
    total_pnl = sum(rewards) * 100.0  # reward_scaleで調整
    final_balance = initial_balance + total_pnl
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    trade_count = action_counts["BUY"] + action_counts["SELL"]
    
    rewards_array = np.array(rewards)
    positions_array = np.array(positions) if positions else np.array([])
    position_changes_array = np.array(position_changes) if position_changes else np.array([])
    
    # 報酬分析
    negative_rewards = rewards_array[rewards_array < 0]
    positive_rewards = rewards_array[rewards_array > 0]
    zero_rewards = rewards_array[rewards_array == 0]
    
    results = {
        "backtest_info": {
            "model": "sac_v397b_balanced",
            "steps": step,
            "data_file": data_path
        },
        "action_distribution": action_distribution,
        "action_counts": action_counts,
        "trade_count": trade_count,
        "performance": {
            "final_balance": float(final_balance),
            "initial_balance": float(initial_balance),
            "total_return_pct": float(total_return)
        },
        "reward_stats": {
            "mean": float(rewards_array.mean()),
            "std": float(rewards_array.std()),
            "min": float(rewards_array.min()),
            "max": float(rewards_array.max()),
            "negative_count": int(len(negative_rewards)),
            "negative_pct": float(len(negative_rewards) / len(rewards_array) * 100),
            "positive_count": int(len(positive_rewards)),
            "positive_pct": float(len(positive_rewards) / len(rewards_array) * 100),
            "zero_count": int(len(zero_rewards)),
            "zero_pct": float(len(zero_rewards) / len(rewards_array) * 100)
        },
        "position_stats": {
            "mean": float(positions_array.mean()) if len(positions_array) > 0 else 0.0,
            "std": float(positions_array.std()) if len(positions_array) > 0 else 0.0,
            "min": float(positions_array.min()) if len(positions_array) > 0 else 0.0,
            "max": float(positions_array.max()) if len(positions_array) > 0 else 0.0,
            "zero_position_pct": float(np.sum(positions_array == 0) / len(positions_array) * 100) if len(positions_array) > 0 else 0.0
        },
        "position_change_stats": {
            "mean": float(position_changes_array.mean()) if len(position_changes_array) > 0 else 0.0,
            "std": float(position_changes_array.std()) if len(position_changes_array) > 0 else 0.0,
            "max": float(position_changes_array.max()) if len(position_changes_array) > 0 else 0.0,
            "significant_changes": int(np.sum(position_changes_array > 0.01)) if len(position_changes_array) > 0 else 0
        }
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """結果を見やすく表示"""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    print("\n📊 Action Distribution:")
    for action, pct in results["action_distribution"].items():
        count = results["action_counts"][action]
        print(f"  {action:6s}: {pct:6.2f}% ({count:,} times)")
    
    print(f"\n💰 Performance:")
    perf = results["performance"]
    print(f"  Initial Balance: ¥{perf['initial_balance']:,.0f}")
    print(f"  Final Balance:   ¥{perf['final_balance']:,.0f}")
    print(f"  Total Return:    {perf['total_return_pct']:+.2f}%")
    
    print(f"\n📈 Trading:")
    print(f"  Total Trades: {results['trade_count']}")
    
    print(f"\n🎁 Reward Statistics:")
    rs = results["reward_stats"]
    print(f"  Mean:     {rs['mean']:+.6f}")
    print(f"  Std:      {rs['std']:.6f}")
    print(f"  Range:    [{rs['min']:+.6f}, {rs['max']:+.6f}]")
    print(f"  Negative: {rs['negative_pct']:.1f}% ({rs['negative_count']:,})")
    print(f"  Positive: {rs['positive_pct']:.1f}% ({rs['positive_count']:,})")
    print(f"  Zero:     {rs['zero_pct']:.1f}% ({rs['zero_count']:,})")
    
    print(f"\n📍 Position Statistics:")
    ps = results["position_stats"]
    print(f"  Mean:     {ps['mean']:.6f}")
    print(f"  Range:    [{ps['min']:.6f}, {ps['max']:.6f}]")
    print(f"  Zero Pos: {ps['zero_position_pct']:.1f}%")
    
    pcs = results["position_change_stats"]
    print(f"  Significant Changes (>1%): {pcs['significant_changes']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    model_path = project_root / "checkpoints" / "sac_session" / "sac_v397b_balanced_final.zip"
    data_path = project_root / "btc_jpy_real_dataset.csv"
    
    results = run_backtest(str(model_path), str(data_path), max_steps=5000)
    
    # 結果表示
    print_results(results)
    
    # JSON保存
    output_path = project_root / "docs" / "evaluation" / "backtest_sac_v397b_balanced_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # v397aとの比較
    print("\n" + "=" * 80)
    print("COMPARISON: v397a vs v397b")
    print("=" * 80)
    
    v397a_results = {
        "HOLD": 92.1,
        "trades": 0,
        "return": 0.0,
        "positive_reward_pct": 0.0
    }
    
    v397b_results = {
        "HOLD": results["action_distribution"]["HOLD"],
        "trades": results["trade_count"],
        "return": results["performance"]["total_return_pct"],
        "positive_reward_pct": results["reward_stats"]["positive_pct"]
    }
    
    print(f"\n{'Metric':<20} {'v397a':>15} {'v397b':>15} {'Change':>15}")
    print("-" * 70)
    print(f"{'HOLD %':<20} {v397a_results['HOLD']:>14.1f}% {v397b_results['HOLD']:>14.1f}% {v397b_results['HOLD']-v397a_results['HOLD']:>+14.1f}%")
    print(f"{'Trade Count':<20} {v397a_results['trades']:>15} {v397b_results['trades']:>15} {v397b_results['trades']-v397a_results['trades']:>+15}")
    print(f"{'Return %':<20} {v397a_results['return']:>+14.2f}% {v397b_results['return']:>+14.2f}% {v397b_results['return']-v397a_results['return']:>+14.2f}%")
    print(f"{'Positive Reward %':<20} {v397a_results['positive_reward_pct']:>14.1f}% {v397b_results['positive_reward_pct']:>14.1f}% {v397b_results['positive_reward_pct']-v397a_results['positive_reward_pct']:>+14.1f}%")
    
    print("\n" + "=" * 80)
