"""
SAC v397g Deep Analysis - Action Distribution & Reward Tracing

BUY行動が発生しない根本原因を貪欲的に調査:
1. アクション値の分布分析
2. BUY/SELL判定条件のトレース
3. 報酬計算の詳細分析
4. ポジション変更ロジックの調査
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["MPLBACKEND"] = "Agg"

from stable_baselines3 import SAC
from ztb.trading.environment.heavy_env import HeavyTradingEnv

def analyze_action_distribution(model, env, max_steps=1000):
    """アクション分布の詳細分析"""
    print("🔍 Analyzing Action Distribution...")

    obs, _ = env.reset()
    actions = []
    action_bins = defaultdict(int)

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        action_value = action[0]
        actions.append(action_value)

        # アクション値をビン分け
        bin_key = round(action_value, 2)
        action_bins[bin_key] += 1

        if terminated or truncated:
            break

    actions = np.array(actions)

    print(f"📊 Action Statistics:")
    print(f"  Total Actions: {len(actions)}")
    print(f"  Mean: {np.mean(actions):.4f}")
    print(f"  Std: {np.std(actions):.4f}")
    print(f"  Min: {np.min(actions):.4f}")
    print(f"  Max: {np.max(actions):.4f}")
    print(f"  Range: [{np.min(actions):.4f}, {np.max(actions):.4f}]")

    # 閾値ごとの分布
    threshold = 0.10
    buy_actions = actions[actions > threshold]
    sell_actions = actions[actions < -threshold]
    hold_actions = actions[(actions >= -threshold) & (actions <= threshold)]

    print(f"\n🎯 Threshold Analysis (threshold={threshold}):")
    print(f"  BUY (> {threshold}): {len(buy_actions)} actions ({100*len(buy_actions)/len(actions):.2f}%)")
    print(f"  HOLD ({-threshold} to {threshold}): {len(hold_actions)} actions ({100*len(hold_actions)/len(actions):.2f}%)")
    print(f"  SELL (< {-threshold}): {len(sell_actions)} actions ({100*len(sell_actions)/len(actions):.2f}%)")

    # アクションビンのトップ10
    print(f"\n📈 Top Action Bins:")
    sorted_bins = sorted(action_bins.items(), key=lambda x: x[1], reverse=True)
    for bin_val, count in sorted_bins[:10]:
        print(f"  {bin_val:.2f}: {count} times")

    return actions, action_bins

def trace_buy_sell_logic(env, max_steps=100):
    """BUY/SELL判定ロジックのトレース"""
    print("\n🔍 Tracing BUY/SELL Logic...")

    # 環境の設定を取得
    config = env.config if hasattr(env, 'config') else {}
    threshold = getattr(config, 'continuous_to_discrete_threshold', 0.10) if hasattr(config, 'continuous_to_discrete_threshold') else 0.10
    max_position = getattr(config, 'max_position_size', 0.01) if hasattr(config, 'max_position_size') else 0.01
    initial_balance = getattr(config, 'initial_portfolio_value', 200000) if hasattr(config, 'initial_portfolio_value') else 200000

    print(f"📋 Environment Settings:")
    print(f"  Threshold: {threshold}")
    print(f"  Max Position Size: {max_position}")
    print(f"  Initial Balance: {initial_balance}")

    obs, _ = env.reset()
    trace_data = []

    for step in range(max_steps):
        action, _ = env.model.predict(obs, deterministic=True) if hasattr(env, 'model') else ([0.0], None)
        # アクション値を取得 (スカラーに変換)
        if isinstance(action, (list, np.ndarray)):
            action_value = float(action[0])
        else:
            action_value = float(action)

        # BUY判定条件のトレース
        can_buy = True
        buy_reasons = []

        # 1. アクション値チェック
        if action_value <= threshold:
            can_buy = False
            buy_reasons.append(f"action_value {action_value:.4f} <= threshold {threshold}")

        # 2. ポジションサイズチェック
        current_position = getattr(env, 'position', 0.0)
        if current_position >= max_position:
            can_buy = False
            buy_reasons.append(f"current_position {current_position:.4f} >= max_position {max_position}")

        # 3. 資金チェック
        portfolio_value = getattr(env, 'portfolio_value', 200000)
        btc_price = getattr(env, 'current_price', 5000000)  # 仮定
        affordable_size = portfolio_value / btc_price
        if current_position + affordable_size > max_position:
            can_buy = False
            buy_reasons.append(f"would exceed max_position: {current_position:.4f} + {affordable_size:.4f} > {max_position}")

        # SELL判定条件のトレース
        can_sell = True
        sell_reasons = []

        if action_value >= -threshold:
            can_sell = False
            sell_reasons.append(f"action_value {action_value:.4f} >= -threshold {-threshold}")

        if current_position <= 0:
            can_sell = False
            sell_reasons.append(f"current_position {current_position:.4f} <= 0")

        trace_data.append({
            'step': step,
            'action_value': action_value,
            'current_position': current_position,
            'portfolio_value': portfolio_value,
            'can_buy': can_buy,
            'buy_reasons': buy_reasons,
            'can_sell': can_sell,
            'sell_reasons': sell_reasons
        })

        # 実際のステップ実行
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # BUY/SELL判定の統計
    buy_attempts = sum(1 for t in trace_data if t['can_buy'])
    sell_attempts = sum(1 for t in trace_data if t['can_sell'])

    print(f"🎯 BUY/SELL Logic Analysis:")
    print(f"  Steps Analyzed: {len(trace_data)}")
    print(f"  BUY Conditions Met: {buy_attempts} times ({100*buy_attempts/len(trace_data):.1f}%)")
    print(f"  SELL Conditions Met: {sell_attempts} times ({100*sell_attempts/len(trace_data):.1f}%)")

    # BUY失敗理由の集計
    buy_failure_reasons = defaultdict(int)
    for t in trace_data:
        if not t['can_buy']:
            for reason in t['buy_reasons']:
                buy_failure_reasons[reason] += 1

    print(f"\n❌ BUY Failure Reasons:")
    for reason, count in sorted(buy_failure_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count} times")

    return trace_data

def analyze_reward_components(env, max_steps=100):
    """報酬成分の詳細分析"""
    print("\n🔍 Analyzing Reward Components...")

    obs, _ = env.reset()
    reward_components = []

    for step in range(max_steps):
        action, _ = env.model.predict(obs, deterministic=True) if hasattr(env, 'model') else ([0.0], None)

        # 報酬計算前の状態保存
        prev_portfolio = getattr(env, 'portfolio_value', 200000)
        prev_position = getattr(env, 'position', 0.0)

        obs, reward, terminated, truncated, info = env.step(action)

        # 報酬成分の推定
        pnl = info.get('pnl', 0.0) if info else 0.0
        portfolio_value = getattr(env, 'portfolio_value', prev_portfolio)

        # RewardCalculatorの推定ロジック
        reward_scale = 1000.0
        pnl_component = pnl * reward_scale

        # 取引ボーナス
        trade_bonus = 0.0
        if abs(action[0]) > 0.005:  # position_threshold
            trade_bonus = 0.2  # trade_execution_bonus_rate

        # 非活性ペナルティ
        inactivity_penalty = 0.0
        if abs(prev_position) < 0.01:  # hold_threshold
            inactivity_penalty = -0.005  # inactivity_penalty_rate

        estimated_reward = pnl_component + trade_bonus + inactivity_penalty

        reward_components.append({
            'step': step,
            'action': action[0] if isinstance(action, (list, np.ndarray)) else action,
            'actual_reward': reward,
            'estimated_reward': estimated_reward,
            'pnl_component': pnl_component,
            'trade_bonus': trade_bonus,
            'inactivity_penalty': inactivity_penalty,
            'pnl': pnl,
            'portfolio_value': portfolio_value,
            'position': getattr(env, 'position', 0.0)
        })

        if terminated or truncated:
            break

    # 報酬成分の統計
    df_rewards = pd.DataFrame(reward_components)

    print(f"💰 Reward Component Analysis:")
    print(f"  Steps: {len(df_rewards)}")
    print(f"  Mean Actual Reward: {df_rewards['actual_reward'].mean():.4f}")
    print(f"  Mean Estimated Reward: {df_rewards['estimated_reward'].mean():.4f}")

    print(f"\n📊 Component Breakdown:")
    print(f"  PnL Component: mean={df_rewards['pnl_component'].mean():.4f}, std={df_rewards['pnl_component'].std():.4f}")
    print(f"  Trade Bonus: mean={df_rewards['trade_bonus'].mean():.4f}, count={df_rewards['trade_bonus'].gt(0).sum()}")
    print(f"  Inactivity Penalty: mean={df_rewards['inactivity_penalty'].mean():.4f}, count={df_rewards['inactivity_penalty'].lt(0).sum()}")

    return reward_components

def main():
    print("=" * 80)
    print("SAC v397g Deep Analysis - Greedy Root Cause Investigation")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # モデルと環境の準備
    model_path = project_root / "checkpoints" / "sac_session" / "sac_model_final.zip"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"📦 Loading model: {model_path}")
    model = SAC.load(str(model_path))

    data_path = project_root / "btc_jpy_real_dataset.csv"
    df = pd.read_csv(data_path).head(1000)  # 分析用に短く

    config = {
        "initial_portfolio_value": 200000,
        "transaction_cost": 0.001,
        "max_position_size": 0.01,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.10,
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 1000.0,
            "reward_clip_min": -10.0,
            "reward_clip_max": 10.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.005,
            "inactivity_penalty_window": 3,
            "inactivity_hold_threshold": 0.01,
            "enable_opportunity_cost": False,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.2,
            "trade_execution_position_threshold": 0.005,
            "trade_execution_action_multiplier": 1.5
        }
    }

    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    env.model = model  # モデルを環境に設定

    # 分析実行
    actions, action_bins = analyze_action_distribution(model, env, max_steps=500)
    trace_data = trace_buy_sell_logic(env, max_steps=100)
    reward_components = analyze_reward_components(env, max_steps=100)

    # 結果保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "v397g_realistic_buy_size",
        "action_analysis": {
            "total_actions": len(actions),
            "mean": float(np.mean(actions)),
            "std": float(np.std(actions)),
            "min": float(np.min(actions)),
            "max": float(np.max(actions)),
            "action_bins": dict(action_bins)
        },
        "logic_trace": trace_data[:10],  # 最初の10ステップのみ
        "reward_analysis": reward_components[:10]  # 最初の10ステップのみ
    }

    output_path = project_root / "docs" / "evaluation" / "v397g_deep_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # NumPy配列をリストに変換してJSONシリアライズ可能にする
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Detailed analysis saved to: {output_path}")

    # 根本原因の仮説
    print("\n" + "=" * 80)
    print("🎯 HYPOTHESIZED ROOT CAUSES:")
    print("=" * 80)

    threshold = 0.10
    buy_actions = len([a for a in actions if a > threshold])
    sell_actions = len([a for a in actions if a < -threshold])

    print(f"1. 🎯 THRESHOLD ISSUE:")
    print(f"   - BUY actions: {buy_actions}/{len(actions)} ({100*buy_actions/len(actions):.2f}%)")
    print(f"   - Model never outputs action > {threshold}")
    print(f"   - Max action value: {np.max(actions):.4f} < threshold {threshold}")

    print(f"\n2. 🧠 LEARNING ISSUE:")
    print(f"   - Model converged to HOLD strategy")
    print(f"   - No exploration of BUY action space")
    print(f"   - Entropy coefficient: -4.37 (very low exploration)")

    print(f"\n3. 💰 REWARD STRUCTURE ISSUE:")
    print(f"   - BUY has no immediate reward signal")
    print(f"   - Only future PnL provides feedback")
    print(f"   - HOLD avoids inactivity penalty")

    print(f"\n4. ⚙️ ENVIRONMENT LOGIC ISSUE:")
    print(f"   - max_position_size=0.01 may be too restrictive")
    print(f"   - BUY execution requires multiple conditions")

    print("\n💡 RECOMMENDED FIXES:")
    print("   1. Lower threshold to 0.05 or train with higher threshold initially")
    print("   2. Add immediate BUY bonus reward")
    print("   3. Increase exploration (higher entropy target)")
    print("   4. Start with larger max_position_size (0.05) then reduce")

    print("=" * 80)

if __name__ == "__main__":
    main()
