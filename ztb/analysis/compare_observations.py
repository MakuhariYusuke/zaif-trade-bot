"""
観測値の実データ比較
v395g（正規化なし）とv395i（正規化あり）で、実際の観測値がどう違うかを確認
"""
import json
import numpy as np
from pathlib import Path

from ztb.utils.data_utils import load_csv_data_optimized
from ztb.training.core.config_builder import ConfigBuilder
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def analyze_observations(config_path, config_name, num_samples=10):
    """観測値を解析"""
    print(f"\n{'='*80}")
    print(f"観測値分析: {config_name}")
    print(f"設定: {config_path}")
    print(f"{'='*80}\n")
    
    # 設定読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # データ読み込み
    dataset_path = config.get('data_path', 'btc_jpy_real_dataset.csv')
    # パスがルート相対の場合、data/datasetsを追加
    if not Path(dataset_path).exists() and not dataset_path.startswith('data/'):
        dataset_path = f"data/datasets/{dataset_path}"
    df = load_csv_data_optimized(dataset_path)
    
    # 統一設定構築
    builder = ConfigBuilder(config)
    unified_config = builder.build_unified_config()
    
    # 環境設定取得
    env_config_dict = unified_config.get("environment", {})
    env_config_dict["use_continuous_actions"] = True
    env_config_dict["enable_action_masking"] = False
    
    # EnvironmentConfigオブジェクトに変換
    config_obj = EnvironmentConfig.from_dict(env_config_dict)
    
    # 環境作成
    env = HeavyTradingEnv(df=df, config=config_obj)
    
    print(f"環境作成成功")
    print(f"  - use_standardized_observations: {getattr(config_obj, 'use_standardized_observations', False)}")
    print(f"  - Observation Space: {env.observation_space}")
    
    # スケーラー情報を確認
    if hasattr(env.observation_builder, 'scaler_mean'):
        if env.observation_builder.scaler_mean is not None:
            print(f"\n✅ スケーラー設定あり")
            print(f"  - scaler_mean: min={env.observation_builder.scaler_mean.min():.6f}, max={env.observation_builder.scaler_mean.max():.6f}")
            print(f"  - scaler_std: min={env.observation_builder.scaler_std.min():.6f}, max={env.observation_builder.scaler_std.max():.6f}")
        else:
            print(f"\n❌ スケーラー設定なし（正規化されません）")
    else:
        print(f"\n❌ スケーラー属性なし")
    
    # 観測値サンプリング
    observations = []
    env.reset()
    
    for i in range(num_samples):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        
        if done or truncated:
            env.reset()
    
    observations = np.array(observations)
    
    # 統計情報
    print(f"\n【観測値統計】（{num_samples}サンプル）")
    print(f"  Shape: {observations.shape}")
    print(f"  全体範囲: [{observations.min():.6f}, {observations.max():.6f}]")
    print(f"  全体平均: {observations.mean():.6f}")
    print(f"  全体標準偏差: {observations.std():.6f}")
    
    # 特徴量ごとの統計
    feature_stats = []
    for i in range(observations.shape[1]):
        feature_vals = observations[:, i]
        stats = {
            'index': i,
            'mean': float(feature_vals.mean()),
            'std': float(feature_vals.std()),
            'min': float(feature_vals.min()),
            'max': float(feature_vals.max())
        }
        feature_stats.append(stats)
    
    # 極端な値を持つ特徴量を表示
    print(f"\n【極端な値を持つ特徴量（上位5）】")
    sorted_by_max = sorted(feature_stats, key=lambda x: abs(x['max']), reverse=True)
    for i, stat in enumerate(sorted_by_max[:5], 1):
        print(f"  {i}. Feature {stat['index']}: [{stat['min']:.6f}, {stat['max']:.6f}], mean={stat['mean']:.6f}, std={stat['std']:.6f}")
    
    # 標準偏差が大きい特徴量
    print(f"\n【標準偏差が大きい特徴量（上位5）】")
    sorted_by_std = sorted(feature_stats, key=lambda x: x['std'], reverse=True)
    for i, stat in enumerate(sorted_by_std[:5], 1):
        print(f"  {i}. Feature {stat['index']}: std={stat['std']:.6f}, mean={stat['mean']:.6f}, range=[{stat['min']:.6f}, {stat['max']:.6f}]")
    
    return {
        'config_name': config_name,
        'use_standardized_observations': getattr(config_obj, 'use_standardized_observations', False),
        'has_scaler': env.observation_builder.scaler_mean is not None if hasattr(env.observation_builder, 'scaler_mean') else False,
        'observations_shape': observations.shape,
        'overall_stats': {
            'min': float(observations.min()),
            'max': float(observations.max()),
            'mean': float(observations.mean()),
            'std': float(observations.std())
        },
        'feature_stats': feature_stats
    }


def main():
    print("\n" + "="*80)
    print("観測値実データ比較分析")
    print("="*80)
    
    # v395g（正規化なし）
    result_v395g = analyze_observations(
        "configs/sac_v395g_micro_reward.json",
        "v395g (正規化なし)"
    )
    
    # v395i（正規化あり）
    result_v395i = analyze_observations(
        "configs/sac_v395i_complete_fix.json",
        "v395i (正規化あり)"
    )
    
    # 比較サマリー
    print(f"\n{'='*80}")
    print("比較サマリー")
    print(f"{'='*80}\n")
    
    print("【v395g（正規化なし）】")
    print(f"  use_standardized_observations: {result_v395g['use_standardized_observations']}")
    print(f"  スケーラー: {result_v395g['has_scaler']}")
    print(f"  観測値範囲: [{result_v395g['overall_stats']['min']:.6f}, {result_v395g['overall_stats']['max']:.6f}]")
    print(f"  観測値平均: {result_v395g['overall_stats']['mean']:.6f}")
    print(f"  観測値標準偏差: {result_v395g['overall_stats']['std']:.6f}")
    
    print("\n【v395i（正規化あり）】")
    print(f"  use_standardized_observations: {result_v395i['use_standardized_observations']}")
    print(f"  スケーラー: {result_v395i['has_scaler']}")
    print(f"  観測値範囲: [{result_v395i['overall_stats']['min']:.6f}, {result_v395i['overall_stats']['max']:.6f}]")
    print(f"  観測値平均: {result_v395i['overall_stats']['mean']:.6f}")
    print(f"  観測値標準偏差: {result_v395i['overall_stats']['std']:.6f}")
    
    print("\n【改善効果】")
    max_reduction = (result_v395g['overall_stats']['max'] - result_v395i['overall_stats']['max']) / result_v395g['overall_stats']['max'] * 100
    std_reduction = (result_v395g['overall_stats']['std'] - result_v395i['overall_stats']['std']) / result_v395g['overall_stats']['std'] * 100
    
    print(f"  最大値削減: {max_reduction:.2f}%")
    print(f"  標準偏差削減: {std_reduction:.2f}%")
    
    # 結果をJSONに保存
    with open("observation_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            'v395g': result_v395g,
            'v395i': result_v395i
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("結果を observation_comparison.json に保存しました")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
