"""
SAC v397c_fixed_scale Backtest Script
max_position_size=1.0維持、RewardCalculator正規化による改善効果を検証
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
from ztb.trading.environment.constants import continuous_to_discrete_action


def run_backtest(
    model_path: str,
    data_path: str,
    max_steps: int = 5000
) -> Dict[str, Any]:
    """バックテスト実行"""
    
    print("=" * 80)
    print("SAC v397c_fixed_scale Backtest")
    print("=" * 80)
    
    # データロード
    df = pd.read_csv(data_path)
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)
    
    # 環境作成（v397c設定）
    config = {
        "initial_balance": 100000,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.25,
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 1000.0,
            "reward_clip_min": -10.0,
            "reward_clip_max": 10.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.01,
            "inactivity_penalty_window": 3,
            "inactivity_hold_threshold": 0.05,
            "enable_opportunity_cost": True,
            "opportunity_cost_rate": 0.005,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.05,
            "trade_execution_position_threshold": 0.01,
            "trade_execution_action_multiplier": 1.5
        }
    }
    
    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    model = SAC.load(model_path)
    
    # バックテスト実行
    print(f"\nRunning backtest ({len(df)} steps)...")
    obs, _ = env.reset()
    
    action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    rewards = []
    pnls = []
    portfolio_values = []
    positions = []
    position_changes = []
    
    step = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 統計収集
        rewards.append(reward)
        if info:
            pnls.append(info.get("pnl", 0.0))
            portfolio_values.append(info.get("portfolio_value", env.portfolio_value))
        
        # ポジション追跡
        if hasattr(env, 'position'):
            positions.append(env.position)
            if len(positions) > 1:
                position_changes.append(abs(positions[-1] - positions[-2]))
        
        # 連続アクションから離散アクションへ変換
        if isinstance(action, np.ndarray):
            continuous_value = action.item()
        else:
            continuous_value = action
        
        discrete_action = continuous_to_discrete_action(
            continuous_value, threshold=getattr(env, "action_threshold", 0.33)
        )

        if discrete_action == 0:
            action_counts["HOLD"] += 1
        elif discrete_action == 1:
            action_counts["BUY"] += 1
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
    
    # 最終残高の取得（環境から取得）
    initial_balance = getattr(env, "initial_portfolio_value", 100000.0)
    final_balance = portfolio_values[-1] if portfolio_values else env.portfolio_value
    realized_pnl = env.realized_pnl
    unrealized_pnl = env.total_pnl - env.realized_pnl
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    trade_count = action_counts["BUY"] + action_counts["SELL"]
    executed_trades = getattr(env, "trades_count", trade_count)
    
    rewards_array = np.array(rewards)
    pnls_array = np.array(pnls) if pnls else np.array([])
    positions_array = np.array(positions) if positions else np.array([])
    position_changes_array = np.array(position_changes) if position_changes else np.array([])
    
    # 報酬分析
    negative_rewards = rewards_array[rewards_array < 0]
    positive_rewards = rewards_array[rewards_array > 0]
    zero_rewards = rewards_array[rewards_array == 0]
    
    results = {
        "backtest_info": {
            "model": "sac_v397c_fixed_scale",
            "steps": step,
            "data_file": data_path
        },
        "action_distribution": action_distribution,
        "action_counts": action_counts,
        "trade_count": trade_count,
    "executed_trades": int(executed_trades),
        "performance": {
            "final_balance": float(final_balance),
            "initial_balance": float(initial_balance),
            "total_return_pct": float(total_return)
        },
        "pnl_breakdown": {
            "realized_pnl": float(realized_pnl),
            "unrealized_pnl": float(unrealized_pnl),
            "step_pnl_mean": float(pnls_array.mean()) if len(pnls_array) > 0 else 0.0,
            "step_pnl_std": float(pnls_array.std()) if len(pnls_array) > 0 else 0.0,
            "step_pnl_min": float(pnls_array.min()) if len(pnls_array) > 0 else 0.0,
            "step_pnl_max": float(pnls_array.max()) if len(pnls_array) > 0 else 0.0,
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
    pnl_break = results["pnl_breakdown"]
    print(f"  Realized PnL:    ¥{pnl_break['realized_pnl']:,.0f}")
    print(f"  Unrealized PnL:  ¥{pnl_break['unrealized_pnl']:,.0f}")
    print(f"  Step PnL μ/σ:   ¥{pnl_break['step_pnl_mean']:,.2f} / ¥{pnl_break['step_pnl_std']:,.2f}")
    print(f"  Step PnL range: [{pnl_break['step_pnl_min']:,.2f}, {pnl_break['step_pnl_max']:,.2f}]")
    
    print(f"\n📈 Trading:")
    print(f"  Total Trades: {results['trade_count']}")
    print(f"  Executed Trades: {results['executed_trades']}")
    
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
    model_path = project_root / "checkpoints" / "sac_session" / "sac_v397c_fixed_scale_final.zip"
    data_path = project_root / "btc_jpy_real_dataset.csv"
    
    results = run_backtest(str(model_path), str(data_path), max_steps=5000)
    
    # 結果表示
    print_results(results)
    
    # JSON保存
    output_path = project_root / "docs" / "evaluation" / "backtest_sac_v397c_fixed_scale_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # v396, v397a, v397bとの比較
    print("\n" + "=" * 80)
    print("COMPARISON: v396 vs v397a vs v397b vs v397c")
    print("=" * 80)
    
    comparison_data = {
        "v396": {"HOLD": 72.1, "trades": 36, "return": 20.08, "pos_reward": "N/A"},
        "v397a": {"HOLD": 92.1, "trades": 0, "return": 0.0, "pos_reward": 0.0},
        "v397b": {"HOLD": 5.9, "trades": 4702, "return": -2.04, "pos_reward": 0.0},
        "v397c": {
            "HOLD": results["action_distribution"].get("HOLD", 0.0),
            "trades": results["trade_count"],
            "return": results["performance"]["total_return_pct"],
            "pos_reward": results["reward_stats"]["positive_pct"]
        }
    }

    print(f"\n{'Model':<10} {'HOLD %':>10} {'Trades':>10} {'Return %':>12} {'Pos Reward %':>15}")
    print("-" * 70)
    for model, data in comparison_data.items():
        print(f"{model:<10} {data['HOLD']:>9.1f}% {data['trades']:>10} {data['return']:>+11.2f}% {str(data['pos_reward']):>14}%")

    print("\n" + "=" * 80)
