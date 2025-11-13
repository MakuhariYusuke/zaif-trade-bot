#!/usr/bin/env python3
"""
ベースラインバックテスト（SIGNAL_GUIDANCEなし）
SIGNAL_GUIDANCEの効果を比較するための基準となるバックテスト
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, List, Any

# パス設定
sys.path.append('.')

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from backtest.data_generator import generate_synthetic_data

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineBacktestEnv:
    """SIGNAL_GUIDANCEなしのベースラインバックテスト環境"""

    def __init__(self, df, config, max_steps: int = 5000):
        self.env = HeavyTradingEnv(df, config)
        self.max_steps = max_steps
        self.portfolio_values = []
        self.actions_taken = []
        self.rewards = []

    def run_backtest(self, num_episodes: int = 3) -> Dict[str, Any]:
        """バックテスト実行"""
        all_results = []

        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes} 開始")

            obs, info = self.env.reset()
            self.portfolio_values = [self.env.portfolio_value]
            self.actions_taken = []
            self.rewards = []

            for step in range(self.max_steps):
                # ランダムアクション（ベースライン）
                action = self.env.action_space.sample()

                obs, reward, terminated, truncated, info = self.env.step(action)

                self.portfolio_values.append(self.env.portfolio_value)
                self.actions_taken.append(action)
                self.rewards.append(reward)

                if terminated or truncated:
                    break

            # エピソード結果計算
            initial_balance = self.portfolio_values[0]
            final_balance = self.portfolio_values[-1]
            total_return_pct = (final_balance - initial_balance) / initial_balance * 100
            total_reward = sum(self.rewards)

            episode_result = {
                'episode': episode + 1,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return_pct': total_return_pct,
                'total_reward': total_reward,
                'steps': len(self.portfolio_values) - 1,
                'portfolio_values': self.portfolio_values.copy()
            }

            all_results.append(episode_result)
            logger.info(f"Episode {episode + 1} 完了: {total_return_pct:.2f}% リターン")

        # 全体結果集計
        avg_return = np.mean([r['total_return_pct'] for r in all_results])
        std_return = np.std([r['total_return_pct'] for r in all_results])
        avg_reward = np.mean([r['total_reward'] for r in all_results])

        summary = {
            'num_episodes': num_episodes,
            'avg_return_pct': avg_return,
            'std_return_pct': std_return,
            'avg_reward': avg_reward,
            'episodes': all_results
        }

        return summary

def main():
    """メイン実行関数"""
    logger.info("ベースラインバックテスト開始")

    # データ生成
    logger.info("📊 Generating synthetic market data...")
    data_df = generate_synthetic_data(
        n_periods=5000,  # 5000期間分のデータ
        start_price=50000.0,
        volatility=500
    )

    logger.info(f"✅ Generated {len(data_df)} data points")

    # V4FeatureExtractorで特徴量を拡張
    logger.info("🔧 Applying V4FeatureExtractor...")
    feature_extractor = V4FeatureExtractor(config={})

    # 特徴量抽出
    enhanced_df = feature_extractor.generate_features(data_df, feature_set="full", model_type="sac")

    logger.info(f"✅ Enhanced features: {len(enhanced_df.columns)} columns")

    # 環境設定
    env_config = EnvironmentConfig(
        transaction_cost=0.001,    # 0.1% 手数料
        max_position_size=0.1,    # 最大ポジションサイズ 10%
        feature_names=list(enhanced_df.columns),  # データフレームの実際の特徴量を使用
        reward_scaling=1.0,
        max_steps=len(enhanced_df),
    )

    backtest_env = BaselineBacktestEnv(enhanced_df, env_config, max_steps=5000)
    results = backtest_env.run_backtest(num_episodes=3)

    logger.info("ベースラインバックテスト完了")
    logger.info(f"平均リターン: {results['avg_return_pct']:.2f}% ± {results['std_return_pct']:.2f}%")
    logger.info(f"平均報酬: {results['avg_reward']:.2f}")

    # 結果保存
    import json
    with open('baseline_backtest_results.json', 'w') as f:
        # numpy値をJSONシリアライズ可能に変換
        json_results = results.copy()
        for episode in json_results['episodes']:
            episode['portfolio_values'] = [float(v) for v in episode['portfolio_values']]
        json.dump(json_results, f, indent=2)

    logger.info("結果を baseline_backtest_results.json に保存しました")

if __name__ == "__main__":
    main()