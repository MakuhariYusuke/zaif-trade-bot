#!/usr/bin/env python3
"""
Week 4 改善版検証スクリプト

改善版モデルの評価:
- エピソード長の改善 (1.2 → 50+ ステップ)
- アクション分布の改善 (100% HOLD → 分散)
- 報酬改善 (-0.078 → より高い値)
- 初期残高 50,000 円での動作確認

実行: python analysis/validate_week4_improved.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_environment(market_data: pd.DataFrame, initial_balance: float = 50000.0):
    """テスト用環境作成"""
    
    base_cols = [f'base_{i}' for i in range(30)]
    mtf_cols = [f'mtf_{i}' for i in range(27)]
    regime_cols = [f'regime_{i}' for i in range(13)]
    
    df = market_data.copy()
    
    for col_list in [base_cols, mtf_cols, regime_cols]:
        for col in col_list:
            if col not in df.columns:
                df[col] = np.random.randn(len(df))
    
    for col in ['atr', 'impact_proxy']:
        if col not in df.columns:
            df[col] = np.random.rand(len(df)) + 1.0
    
    return FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols[:30],
        mtf_feature_columns=mtf_cols[:27],
        regime_feature_columns=regime_cols[:13],
        initial_balance=initial_balance,
        max_position=initial_balance / 100,
        max_steps=500,
        drawdown_limit=0.3,
        prewarm_steps=100
    )


def evaluate_model(model_path: Path, market_data: pd.DataFrame, episodes: int = 30):
    """モデル評価"""
    
    logger.info("=" * 70)
    logger.info(f"モデル評価: {model_path.name}")
    logger.info("=" * 70)
    
    try:
        model = SAC.load(str(model_path))
        env = create_test_environment(market_data, initial_balance=50000.0)
        
        episode_rewards = []
        episode_lengths = []
        action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for ep in range(episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            ep_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated
                
                # アクション分類
                target_pos = action[0]
                if target_pos < -0.3:
                    action_counts['sell'] += 1
                elif target_pos > 0.3:
                    action_counts['buy'] += 1
                else:
                    action_counts['hold'] += 1
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            
            if (ep + 1) % 10 == 0:
                logger.info(f"  エピソード {ep + 1}/{episodes} - "
                          f"報酬: {ep_reward:7.4f}, 長さ: {ep_length:3d} ステップ")
        
        env.close()
        
        # 統計計算
        rewards_array = np.array(episode_rewards)
        lengths_array = np.array(episode_lengths)
        
        total_actions = sum(action_counts.values())
        
        results = {
            'episodes': episodes,
            'reward_mean': float(np.mean(rewards_array)),
            'reward_std': float(np.std(rewards_array)),
            'reward_min': float(np.min(rewards_array)),
            'reward_max': float(np.max(rewards_array)),
            'length_mean': float(np.mean(lengths_array)),
            'length_std': float(np.std(lengths_array)),
            'length_min': float(np.min(lengths_array)),
            'length_max': float(np.max(lengths_array)),
            'action_distribution': action_counts,
            'action_pct': {
                'buy': (action_counts['buy'] / total_actions * 100) if total_actions > 0 else 0,
                'sell': (action_counts['sell'] / total_actions * 100) if total_actions > 0 else 0,
                'hold': (action_counts['hold'] / total_actions * 100) if total_actions > 0 else 0
            }
        }
        
        logger.info("")
        logger.info("評価結果:")
        logger.info(f"  報酬 (平均±std): {results['reward_mean']:.6f} ± {results['reward_std']:.6f}")
        logger.info(f"  報酬 (範囲): [{results['reward_min']:.6f}, {results['reward_max']:.6f}]")
        logger.info(f"  エピソード長 (平均±std): {results['length_mean']:.1f} ± {results['length_std']:.1f} ステップ")
        logger.info(f"  エピソード長 (範囲): [{results['length_min']:.0f}, {results['length_max']:.0f}] ステップ")
        logger.info(f"  アクション分布:")
        logger.info(f"    買: {results['action_pct']['buy']:6.1f}%")
        logger.info(f"    売: {results['action_pct']['sell']:6.1f}%")
        logger.info(f"    保有: {results['action_pct']['hold']:6.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"評価エラー: {e}")
        return None


def compare_with_baseline(improved_results, baseline_values):
    """改善版とベースライン（元の5000ステップ）の比較"""
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("改善版 vs ベースライン (5000ステップ版) 比較")
    logger.info("=" * 70)
    
    print()
    print(f"{'メトリクス':<25} {'ベースライン':>15} {'改善版':>15} {'改善度':>15}")
    print("-" * 70)
    
    # エピソード長
    baseline_length = 1.2
    improved_length = improved_results['length_mean']
    improvement = improved_length / baseline_length
    print(f"{'エピソード長 (ステップ)':<25} {baseline_length:>15.1f} {improved_length:>15.1f} {improvement:>15.1f}x")
    
    # HOLD比率
    baseline_hold = 100.0
    improved_hold = improved_results['action_pct']['hold']
    print(f"{'HOLD比率 (%)':<25} {baseline_hold:>15.1f}% {improved_hold:>15.1f}% {baseline_hold - improved_hold:>14.1f}pp")
    
    # BUY+SELL比率
    baseline_trade = 0.0
    improved_trade = improved_results['action_pct']['buy'] + improved_results['action_pct']['sell']
    print(f"{'BUY+SELL比率 (%)':<25} {baseline_trade:>15.1f}% {improved_trade:>15.1f}% {improved_trade - baseline_trade:>14.1f}pp")
    
    # 報酬
    baseline_reward = -0.0783
    improved_reward = improved_results['reward_mean']
    improvement_ratio = improved_reward / baseline_reward if baseline_reward != 0 else 0
    print(f"{'報酬 (平均)':<25} {baseline_reward:>15.6f} {improved_reward:>15.6f} {improvement_ratio:>15.1f}x")
    
    print()
    
    # 判定
    logger.info("=" * 70)
    logger.info("改善判定")
    logger.info("=" * 70)
    
    issues = []
    
    if improved_length < 10:
        issues.append("❌ エピソード長が依然として短い (< 10ステップ)")
    else:
        logger.info("✓ Issue 1 解決: エピソード長改善")
    
    if improved_hold > 80:
        issues.append(f"❌ HOLD比率が依然として高い (>{improved_hold:.0f}%)")
    else:
        logger.info("✓ Issue 2 部分解決: アクション多様化開始")
    
    if improved_results['reward_mean'] < -0.1:
        issues.append(f"⚠️ 報酬がネガティブ ({improved_results['reward_mean']:.4f})")
    else:
        logger.info("✓ 報酬改善: 高い値を達成")
    
    if issues:
        print()
        for issue in issues:
            logger.warning(issue)
    else:
        print()
        logger.info("🎉 すべての改善項目で成功!")
    
    return issues


def main():
    """メイン実行"""
    
    print()
    print("=" * 70)
    print("Week 4 改善版検証")
    print("=" * 70)
    print()
    
    # 最新モデルを探す
    model_dir = PROJECT_ROOT / 'models' / 'week4_improved'
    if not model_dir.exists():
        logger.error(f"モデルディレクトリが見つかりません: {model_dir}")
        return
    
    model_files = list(model_dir.glob('sac_improved_v456_*.zip'))
    if not model_files:
        logger.error("モデルが見つかりません")
        return
    
    model_path = sorted(model_files)[-1]
    logger.info(f"最新モデル: {model_path.name}")
    
    # 市場データ読み込み
    data_path = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
    market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 評価実行
    results = evaluate_model(model_path, market_data, episodes=30)
    
    if results:
        # ベースラインとの比較
        baseline_values = {
            'length': 1.2,
            'hold_ratio': 100.0,
            'reward': -0.0783
        }
        
        issues = compare_with_baseline(results, baseline_values)
        
        # 次のステップ判定
        print()
        logger.info("=" * 70)
        logger.info("次のステップ")
        logger.info("=" * 70)
        
        if not issues or len(issues) <= 1:
            logger.info("✓ 段階1 (パラメータ調整) が成功")
            logger.info("→ 段階2 (報酬関数チューニング) へ進みます")
        else:
            logger.warning("⚠️ さらなる改善が必要")
            logger.info("→ 設定を見直してから再度実行してください")


if __name__ == '__main__':
    main()
