"""
SAC v397b_balanced Deep Analysis Script
v397bの詳細分析：報酬成分、時系列パターン、学習動態

分析項目:
1. 報酬成分の内訳（PnL、ボーナス、ペナルティの分離）
2. 時系列パターン（報酬、アクション、ポジションの推移）
3. 取引パターン（保有期間、取引サイズ、タイミング）
4. 学習動態（訓練メトリクスの推移）
5. v396との比較
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.trading.environment.environment import HeavyTradingEnv


def analyze_reward_components(
    model_path: str, data_path: str, max_steps: int = 5000
) -> Dict[str, Any]:
    """報酬成分を詳細分析"""

    print("=" * 80)
    print("Detailed Reward Component Analysis")
    print("=" * 80)

    # データロード
    df = pd.read_csv(data_path)
    if max_steps and len(df) > max_steps:
        df = df.head(max_steps)

    # 環境作成
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
            "trade_execution_action_multiplier": 1.5,
        },
    }

    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    model = SAC.load(model_path)

    # シミュレーション実行
    obs, _ = env.reset()

    # データ収集
    total_rewards = []
    pnl_components = []
    trade_bonuses = []
    inactivity_penalties = []
    positions = []
    position_changes = []
    continuous_actions = []
    discrete_actions = []
    prices = []

    step = 0
    done = False
    last_position = 0

    print(f"\nRunning detailed simulation ({len(df)} steps)...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # 現在の状態を記録
        current_price = df.iloc[min(env.current_step, len(df) - 1)]["close"]
        prices.append(current_price)

        # アクション記録
        from ztb.trading.environment.constants import continuous_to_discrete_action

        if isinstance(action, np.ndarray):
            continuous_value = action.item()
        else:
            continuous_value = action
        continuous_actions.append(continuous_value)

        discrete_action = continuous_to_discrete_action(
            continuous_value, threshold=0.15
        )
        discrete_actions.append(discrete_action)

        # ステップ実行
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 報酬成分を推定（環境内部の計算を再現）
        current_position = env.position
        positions.append(current_position)

        position_change = abs(current_position - last_position)
        position_changes.append(position_change)

        # PnL成分（簡易推定）
        if step > 0:
            price_change = (prices[-1] - prices[-2]) / prices[-2]
            pnl_estimate = price_change * last_position * config["initial_balance"]
            pnl_scaled = pnl_estimate * config["reward_settings"]["reward_scale"]
        else:
            pnl_scaled = 0

        pnl_components.append(pnl_scaled)

        # 取引ボーナス
        if (
            position_change
            > config["reward_settings"]["trade_execution_position_threshold"]
        ):
            trade_bonus = config["reward_settings"]["trade_execution_bonus_rate"]
        else:
            trade_bonus = 0
        trade_bonuses.append(trade_bonus)

        # 不活性ペナルティ（簡易推定）
        if (
            abs(position_change)
            < config["reward_settings"]["inactivity_hold_threshold"]
        ):
            inactivity_penalty = -config["reward_settings"]["inactivity_penalty_rate"]
        else:
            inactivity_penalty = 0
        inactivity_penalties.append(inactivity_penalty)

        total_rewards.append(reward)
        last_position = current_position

        step += 1
        if step % 1000 == 0:
            print(f"Step {step}/{len(df)}")

    # 統計計算
    total_rewards_arr = np.array(total_rewards)
    pnl_components_arr = np.array(pnl_components)
    trade_bonuses_arr = np.array(trade_bonuses)
    inactivity_penalties_arr = np.array(inactivity_penalties)
    positions_arr = np.array(positions)
    position_changes_arr = np.array(position_changes)
    continuous_actions_arr = np.array(continuous_actions)
    discrete_actions_arr = np.array(discrete_actions)
    prices_arr = np.array(prices)

    # 報酬成分分析
    results = {
        "reward_components": {
            "total_reward": {
                "mean": float(total_rewards_arr.mean()),
                "std": float(total_rewards_arr.std()),
                "min": float(total_rewards_arr.min()),
                "max": float(total_rewards_arr.max()),
                "sum": float(total_rewards_arr.sum()),
            },
            "pnl_component_estimate": {
                "mean": float(pnl_components_arr.mean()),
                "std": float(pnl_components_arr.std()),
                "min": float(pnl_components_arr.min()),
                "max": float(pnl_components_arr.max()),
                "sum": float(pnl_components_arr.sum()),
                "contribution_pct": float(
                    pnl_components_arr.sum() / total_rewards_arr.sum() * 100
                )
                if total_rewards_arr.sum() != 0
                else 0,
            },
            "trade_bonus": {
                "mean": float(trade_bonuses_arr.mean()),
                "fired_count": int(np.sum(trade_bonuses_arr > 0)),
                "fired_pct": float(
                    np.sum(trade_bonuses_arr > 0) / len(trade_bonuses_arr) * 100
                ),
                "sum": float(trade_bonuses_arr.sum()),
                "contribution_pct": float(
                    trade_bonuses_arr.sum() / abs(total_rewards_arr.sum()) * 100
                )
                if total_rewards_arr.sum() != 0
                else 0,
            },
            "inactivity_penalty": {
                "mean": float(inactivity_penalties_arr.mean()),
                "fired_count": int(np.sum(inactivity_penalties_arr < 0)),
                "fired_pct": float(
                    np.sum(inactivity_penalties_arr < 0)
                    / len(inactivity_penalties_arr)
                    * 100
                ),
                "sum": float(inactivity_penalties_arr.sum()),
                "contribution_pct": float(
                    abs(inactivity_penalties_arr.sum())
                    / abs(total_rewards_arr.sum())
                    * 100
                )
                if total_rewards_arr.sum() != 0
                else 0,
            },
        },
        "time_series_patterns": {
            "reward_autocorrelation": float(
                np.corrcoef(total_rewards_arr[:-1], total_rewards_arr[1:])[0, 1]
            )
            if len(total_rewards_arr) > 1
            else 0,
            "position_autocorrelation": float(
                np.corrcoef(positions_arr[:-1], positions_arr[1:])[0, 1]
            )
            if len(positions_arr) > 1
            else 0,
            "action_position_correlation": float(
                np.corrcoef(continuous_actions_arr, positions_arr)[0, 1]
            ),
            "reward_position_change_correlation": float(
                np.corrcoef(total_rewards_arr, position_changes_arr)[0, 1]
            ),
        },
        "trading_patterns": {
            "position_hold_periods": analyze_hold_periods(positions_arr),
            "position_size_distribution": {
                "mean": float(positions_arr.mean()),
                "std": float(positions_arr.std()),
                "median": float(np.median(positions_arr)),
                "quartiles": [
                    float(q) for q in np.percentile(positions_arr, [25, 50, 75])
                ],
            },
            "position_change_distribution": {
                "mean": float(position_changes_arr.mean()),
                "std": float(position_changes_arr.std()),
                "median": float(np.median(position_changes_arr)),
                "large_changes_count": int(np.sum(position_changes_arr > 0.1)),
                "small_changes_count": int(
                    np.sum((position_changes_arr > 0) & (position_changes_arr <= 0.01))
                ),
            },
        },
        "market_interaction": {
            "price_trend": {
                "mean_return": float(
                    np.diff(prices_arr).mean() / prices_arr[:-1].mean()
                ),
                "volatility": float(np.std(np.diff(prices_arr) / prices_arr[:-1])),
                "total_price_change_pct": float(
                    (prices_arr[-1] - prices_arr[0]) / prices_arr[0] * 100
                ),
            },
            "position_vs_trend": analyze_position_vs_trend(positions_arr, prices_arr),
        },
        "action_distribution": {
            "BUY_pct": float(
                np.sum(discrete_actions_arr == 0) / len(discrete_actions_arr) * 100
            ),
            "HOLD_pct": float(
                np.sum(discrete_actions_arr == 1) / len(discrete_actions_arr) * 100
            ),
            "SELL_pct": float(
                np.sum(discrete_actions_arr == 2) / len(discrete_actions_arr) * 100
            ),
            "continuous_action_mean": float(continuous_actions_arr.mean()),
            "continuous_action_std": float(continuous_actions_arr.std()),
        },
        "time_series_data": {
            "steps": list(range(len(total_rewards))),
            "total_rewards": total_rewards[:100],  # 最初の100ステップのサンプル
            "pnl_components": pnl_components[:100],
            "trade_bonuses": trade_bonuses[:100],
            "positions": positions[:100],
            "discrete_actions": discrete_actions[:100],
            "prices": prices[:100],
        },
    }

    return results


def analyze_hold_periods(positions: np.ndarray) -> Dict[str, Any]:
    """ポジション保有期間を分析"""
    # ポジションが変化した時点を検出
    position_changes_indices = np.where(np.abs(np.diff(positions)) > 0.001)[0]

    if len(position_changes_indices) == 0:
        return {
            "mean_hold_period": len(positions),
            "median_hold_period": len(positions),
            "max_hold_period": len(positions),
            "min_hold_period": len(positions),
            "total_periods": 1,
        }

    # 保有期間を計算
    hold_periods = np.diff(
        np.concatenate([[0], position_changes_indices, [len(positions)]])
    )

    return {
        "mean_hold_period": float(hold_periods.mean()),
        "median_hold_period": float(np.median(hold_periods)),
        "max_hold_period": int(hold_periods.max()),
        "min_hold_period": int(hold_periods.min()),
        "total_periods": int(len(hold_periods)),
    }


def analyze_position_vs_trend(
    positions: np.ndarray, prices: np.ndarray
) -> Dict[str, Any]:
    """ポジションと価格トレンドの関係を分析"""
    # 価格変化率
    price_returns = np.diff(prices) / prices[:-1]

    # ポジションを1ステップ遅らせて対応させる
    positions_aligned = positions[:-1]

    # ロング/ショート/ゼロに分類
    long_positions = positions_aligned > 0.01
    short_positions = positions_aligned < -0.01

    # トレンドとの一致率
    long_in_uptrend = np.sum((long_positions) & (price_returns > 0))
    long_in_downtrend = np.sum((long_positions) & (price_returns < 0))
    short_in_uptrend = np.sum((short_positions) & (price_returns > 0))
    short_in_downtrend = np.sum((short_positions) & (price_returns < 0))

    total_long = np.sum(long_positions)
    total_short = np.sum(short_positions)

    return {
        "long_positions_count": int(total_long),
        "short_positions_count": int(total_short),
        "long_in_uptrend_pct": float(long_in_uptrend / total_long * 100)
        if total_long > 0
        else 0,
        "long_in_downtrend_pct": float(long_in_downtrend / total_long * 100)
        if total_long > 0
        else 0,
        "short_in_uptrend_pct": float(short_in_uptrend / total_short * 100)
        if total_short > 0
        else 0,
        "short_in_downtrend_pct": float(short_in_downtrend / total_short * 100)
        if total_short > 0
        else 0,
        "correct_direction_pct": float(
            (long_in_uptrend + short_in_downtrend) / len(price_returns) * 100
        ),
        "wrong_direction_pct": float(
            (long_in_downtrend + short_in_uptrend) / len(price_returns) * 100
        ),
    }


def print_detailed_analysis(results: Dict[str, Any]):
    """詳細分析結果を表示"""
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS RESULTS")
    print("=" * 80)

    # 報酬成分
    print("\n📊 Reward Components Breakdown:")
    rc = results["reward_components"]
    print("\n  Total Reward:")
    print(f"    Mean:  {rc['total_reward']['mean']:+.6f}")
    print(f"    Sum:   {rc['total_reward']['sum']:+.2f}")

    print("\n  PnL Component (estimated):")
    print(f"    Mean:         {rc['pnl_component_estimate']['mean']:+.6f}")
    print(f"    Sum:          {rc['pnl_component_estimate']['sum']:+.2f}")
    print(f"    Contribution: {rc['pnl_component_estimate']['contribution_pct']:+.1f}%")

    print("\n  Trade Execution Bonus:")
    print(
        f"    Fired:        {rc['trade_bonus']['fired_count']:,} times ({rc['trade_bonus']['fired_pct']:.1f}%)"
    )
    print(f"    Sum:          {rc['trade_bonus']['sum']:+.2f}")
    print(f"    Contribution: {rc['trade_bonus']['contribution_pct']:.1f}%")

    print("\n  Inactivity Penalty:")
    print(
        f"    Fired:        {rc['inactivity_penalty']['fired_count']:,} times ({rc['inactivity_penalty']['fired_pct']:.1f}%)"
    )
    print(f"    Sum:          {rc['inactivity_penalty']['sum']:+.2f}")
    print(f"    Contribution: {rc['inactivity_penalty']['contribution_pct']:.1f}%")

    # 時系列パターン
    print("\n📈 Time Series Patterns:")
    ts = results["time_series_patterns"]
    print(f"  Reward Autocorrelation:              {ts['reward_autocorrelation']:+.3f}")
    print(
        f"  Position Autocorrelation:            {ts['position_autocorrelation']:+.3f}"
    )
    print(
        f"  Action-Position Correlation:         {ts['action_position_correlation']:+.3f}"
    )
    print(
        f"  Reward-PositionChange Correlation:   {ts['reward_position_change_correlation']:+.3f}"
    )

    # 取引パターン
    print("\n💼 Trading Patterns:")
    tp = results["trading_patterns"]
    print("  Position Hold Periods:")
    print(f"    Mean:   {tp['position_hold_periods']['mean_hold_period']:.1f} steps")
    print(f"    Median: {tp['position_hold_periods']['median_hold_period']:.1f} steps")
    print(
        f"    Range:  [{tp['position_hold_periods']['min_hold_period']}, {tp['position_hold_periods']['max_hold_period']}]"
    )
    print(f"    Total Periods: {tp['position_hold_periods']['total_periods']}")

    print("\n  Position Size Distribution:")
    print(f"    Mean:     {tp['position_size_distribution']['mean']:+.6f}")
    print(f"    Std:      {tp['position_size_distribution']['std']:.6f}")
    print(f"    Quartiles: {tp['position_size_distribution']['quartiles']}")

    print("\n  Position Changes:")
    print(
        f"    Large (>10%):  {tp['position_change_distribution']['large_changes_count']:,}"
    )
    print(
        f"    Small (≤1%):   {tp['position_change_distribution']['small_changes_count']:,}"
    )

    # 市場との相互作用
    print("\n🌍 Market Interaction:")
    mi = results["market_interaction"]
    print("  Price Trend:")
    print(f"    Total Change:  {mi['price_trend']['total_price_change_pct']:+.2f}%")
    print(f"    Volatility:    {mi['price_trend']['volatility']:.4f}")

    pvt = mi["position_vs_trend"]
    print("\n  Position vs Trend:")
    print(f"    Long Positions:  {pvt['long_positions_count']:,}")
    print(f"      In Uptrend:    {pvt['long_in_uptrend_pct']:.1f}%")
    print(f"      In Downtrend:  {pvt['long_in_downtrend_pct']:.1f}%")
    print(f"    Short Positions: {pvt['short_positions_count']:,}")
    print(f"      In Uptrend:    {pvt['short_in_uptrend_pct']:.1f}%")
    print(f"      In Downtrend:  {pvt['short_in_downtrend_pct']:.1f}%")
    print(f"    Correct Direction: {pvt['correct_direction_pct']:.1f}%")
    print(f"    Wrong Direction:   {pvt['wrong_direction_pct']:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    model_path = (
        project_root / "checkpoints" / "sac_session" / "sac_v397b_balanced_final.zip"
    )
    data_path = project_root / "btc_jpy_real_dataset.csv"

    results = analyze_reward_components(str(model_path), str(data_path), max_steps=5000)

    # 結果表示
    print_detailed_analysis(results)

    # JSON保存
    output_path = project_root / "docs" / "evaluation" / "v397b_deep_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed analysis saved to: {output_path}")
