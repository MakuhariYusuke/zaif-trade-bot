"""
SAC環境診断スクリプト（シンプル版）

UnifiedTrainerで訓練している環境を直接使用して、
報酬分布、行動分布、観測値スケールを診断する。
"""

import json
import numpy as np
from pathlib import Path

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.training.core.config_builder import ConfigBuilder
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def diagnose_environment(config_path: str, num_episodes: int = 3, num_steps: int = 100):
    """
    環境診断実行
    
    Args:
        config_path: 設定ファイルパス
        num_episodes: 診断エピソード数
        num_steps: エピソードあたりのステップ数
    """
    print(f"\n{'='*80}")
    print(f"SAC環境診断")
    print(f"設定: {config_path}")
    print(f"エピソード数: {num_episodes}, ステップ/エピソード: {num_steps}")
    print(f"{'='*80}\n")
    
    # 設定読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # データ読み込み
    dataset_path = config.get('data_path', 'btc_jpy_real_dataset.csv')
    df = load_csv_data_optimized(dataset_path)
    print(f"データ読み込み: {dataset_path} ({len(df)} rows)")
    
    # 統一設定構築
    builder = ConfigBuilder(config)
    unified_config = builder.build_unified_config()
    
    # 環境設定取得
    env_config_dict = unified_config.get("environment", {})
    
    # SAC用に連続行動空間を有効化
    env_config_dict["use_continuous_actions"] = True
    env_config_dict["enable_action_masking"] = False
    
    # EnvironmentConfigオブジェクトに変換
    config_obj = EnvironmentConfig.from_dict(env_config_dict)
    
    # 環境作成
    env = HeavyTradingEnv(df=df, config=config_obj)
    if env is None:
        print("❌ 環境が作成されていません")
        return
    
    print(f"✓ 環境取得成功")
    print(f"  - Action Space: {env.action_space}")
    print(f"  - Observation Space: {env.observation_space}")
    
    # 診断データ収集
    all_rewards = []
    all_observations = []
    all_actions_continuous = []
    all_actions_discrete = []
    
    for ep in range(1, num_episodes + 1):
        print(f"\n--- Episode {ep} ---")
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result  # Gymnasium形式
        else:
            obs = reset_result  # 古い形式
        
        episode_rewards = []
        episode_obs = []
        episode_act_cont = []
        episode_act_disc = []
        
        for step in range(num_steps):
            # ランダム連続行動
            action_cont = env.action_space.sample()
            
            # 連続→離散変換
            from ztb.trading.environment.constants import continuous_to_discrete_action
            if isinstance(action_cont, np.ndarray):
                continuous_value = float(action_cont[0])
            else:
                continuous_value = float(action_cont)
            discrete_action = continuous_to_discrete_action(continuous_value)
            
            # Step実行（Gymnasium形式: 5つの戻り値）
            step_result = env.step(action_cont)
            if len(step_result) == 5:
                next_obs, reward, done, truncated, info = step_result
                done = done or truncated  # Gymnasium形式
            else:
                next_obs, reward, done, info = step_result
            
            # データ記録
            episode_rewards.append(reward)
            if isinstance(obs, np.ndarray):
                episode_obs.append(obs.copy())
            else:
                episode_obs.append(np.array(obs))
            episode_act_cont.append(continuous_value)
            episode_act_disc.append(discrete_action)
            
            # 進捗表示
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{num_steps} | Reward: {reward:+.4f}")
            
            obs = next_obs
            
            if done:
                print(f"  Episode終了 at step {step+1}")
                break
        
        # エピソードデータを全体に追加
        all_rewards.extend(episode_rewards)
        all_observations.extend(episode_obs)
        all_actions_continuous.extend(episode_act_cont)
        all_actions_discrete.extend(episode_act_disc)
        
        # エピソード統計
        ep_rewards = np.array(episode_rewards)
        print(f"  報酬統計:")
        print(f"    平均: {ep_rewards.mean():+.6f}")
        print(f"    範囲: [{ep_rewards.min():+.6f}, {ep_rewards.max():+.6f}]")
        print(f"    標準偏差: {ep_rewards.std():.6f}")
    
    # 全体分析
    print(f"\n{'='*80}")
    print(f"全体統計 ({len(all_rewards)} steps)")
    print(f"{'='*80}")
    
    rewards = np.array(all_rewards)
    observations = np.array(all_observations)
    actions_cont = np.array(all_actions_continuous)
    actions_disc = np.array(all_actions_discrete)
    
    # 報酬分析
    print(f"\n【報酬分析】")
    print(f"  平均: {rewards.mean():+.6f}")
    print(f"  中央値: {np.median(rewards):+.6f}")
    print(f"  標準偏差: {rewards.std():.6f}")
    print(f"  範囲: [{rewards.min():+.6f}, {rewards.max():+.6f}]")
    print(f"  5%tile: {np.percentile(rewards, 5):+.6f}")
    print(f"  95%tile: {np.percentile(rewards, 95):+.6f}")
    print(f"  正の報酬: {(rewards > 0).sum()} ({(rewards > 0).sum()/len(rewards)*100:.1f}%)")
    print(f"  負の報酬: {(rewards < 0).sum()} ({(rewards < 0).sum()/len(rewards)*100:.1f}%)")
    print(f"  ゼロ: {(rewards == 0).sum()} ({(rewards == 0).sum()/len(rewards)*100:.1f}%)")
    
    # 観測値分析
    print(f"\n【観測値分析】")
    print(f"  Shape: {observations.shape}")
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0)
    obs_min = observations.min(axis=0)
    obs_max = observations.max(axis=0)
    print(f"  特徴量ごとの平均範囲: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
    print(f"  特徴量ごとの標準偏差範囲: [{obs_std.min():.4f}, {obs_std.max():.4f}]")
    print(f"  特徴量ごとの最小値範囲: [{obs_min.min():.4f}, {obs_min.max():.4f}]")
    print(f"  特徴量ごとの最大値範囲: [{obs_max.min():.4f}, {obs_max.max():.4f}]")
    
    # 異常な特徴量を検出
    large_std_indices = np.where(obs_std > 10.0)[0]
    if len(large_std_indices) > 0:
        print(f"  ⚠️ 標準偏差が大きい特徴量 (> 10.0): {len(large_std_indices)}個")
        for idx in large_std_indices[:5]:  # 最初の5個だけ表示
            print(f"      特徴量 {idx}: std={obs_std[idx]:.2f}, range=[{obs_min[idx]:.2f}, {obs_max[idx]:.2f}]")
    
    # 行動分析
    print(f"\n【行動分析】")
    print(f"  連続行動範囲: [{actions_cont.min():.4f}, {actions_cont.max():.4f}]")
    print(f"  連続行動平均: {actions_cont.mean():.4f}")
    print(f"  連続行動標準偏差: {actions_cont.std():.4f}")
    
    unique_disc, counts_disc = np.unique(actions_disc, return_counts=True)
    print(f"  離散行動分布:")
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    for action, count in zip(unique_disc, counts_disc):
        print(f"    {action_names.get(action, f'Action {action}')}: {count} ({count/len(actions_disc)*100:.1f}%)")
    
    # 問題検出
    print(f"\n{'='*80}")
    print(f"問題検出")
    print(f"{'='*80}")
    
    issues = []
    
    # 報酬分散チェック
    if rewards.std() > 1.0:
        issues.append(f"⚠️ 報酬の標準偏差が大きい: {rewards.std():.4f} (> 1.0)")
    
    # 報酬範囲チェック
    if rewards.max() > 10.0 or rewards.min() < -10.0:
        issues.append(f"⚠️ 報酬範囲が広すぎる: [{rewards.min():.4f}, {rewards.max():.4f}]")
    
    # ゼロ報酬チェック
    zero_pct = (rewards == 0).sum() / len(rewards)
    if zero_pct > 0.5:
        issues.append(f"⚠️ ゼロ報酬が多すぎる: {zero_pct*100:.1f}%")
    
    # 観測値スケールチェック
    if obs_std.max() > 100.0:
        issues.append(f"⚠️ 観測値のスケールが大きすぎる: max_std={obs_std.max():.2f}")
    
    # 行動の偏りチェック
    max_action_pct = counts_disc.max() / len(actions_disc)
    if max_action_pct > 0.8:
        max_action = unique_disc[counts_disc.argmax()]
        issues.append(f"⚠️ 行動が偏りすぎている: {action_names.get(max_action, f'Action {max_action}')} が {max_action_pct*100:.1f}%")
    
    if issues:
        print(f"\n検出された問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✓ 重大な問題は検出されませんでした")
    
    # 結果保存
    output = {
        "config": config_path,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "total_steps": len(all_rewards),
        "rewards": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "median": float(np.median(rewards)),
        },
        "observations": {
            "shape": observations.shape,
            "mean_range": [float(obs_mean.min()), float(obs_mean.max())],
            "std_range": [float(obs_std.min()), float(obs_std.max())],
            "large_std_count": int(len(large_std_indices)),
        },
        "actions": {
            "continuous_range": [float(actions_cont.min()), float(actions_cont.max())],
            "discrete_distribution": {int(k): int(v) for k, v in zip(unique_disc, counts_disc)},
        },
        "issues": issues,
    }
    
    output_path = "sac_environment_diagnostics_simple.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n診断結果を保存: {output_path}")
    print(f"\n{'='*80}")
    print(f"診断完了")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    diagnose_environment(
        config_path="configs/sac_v395h_normalized.json",
        num_episodes=3,
        num_steps=100
    )
