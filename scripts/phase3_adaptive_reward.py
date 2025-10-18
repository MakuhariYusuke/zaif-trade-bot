#!/usr/bin/env python3
"""
Phase 3: Adaptive Reward System - SAC v426 Improvement Plan

このスクリプトは、SAC v424の適応性不足問題を解決するために、
相関認識特徴量に基づく適応型報酬システムを実装します。

目標:
- 相関スコアに基づく動的報酬調整
- レジーム特化型報酬関数
- 市場適応性の高い学習システム

これにより、SAC v424の適応性 (0.262) を大幅に向上させます。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveRewardSystem:
    """
    適応型報酬システムクラス

    SAC v424の適応性不足を解決するための動的報酬調整システム。
    """

    def __init__(self, data_path: str = "data/btc_jpy_correlation_aware_v426_dataset.csv"):
        self.data_path = Path(data_path)
        self.output_path = self.data_path.parent / "adaptive_reward_system_v426.json"
        self.report_path = self.data_path.parent / "phase3_adaptive_reward_report.md"

        # 報酬パラメータ
        self.reward_configs = {
            'cost_aware': {
                'base_penalty': -0.001,
                'correlation_bonus': 0.01,
                'regime_multiplier': 1.0,
                'volatility_penalty': -0.005
            },
            'strong_penalty': {
                'base_penalty': -0.01,
                'correlation_bonus': 0.05,
                'regime_multiplier': 2.0,
                'volatility_penalty': -0.02
            },
            'correlation_focused': {
                'base_penalty': -0.005,
                'correlation_bonus': 0.1,
                'regime_multiplier': 1.5,
                'volatility_penalty': -0.01
            }
        }

    def load_data(self) -> pd.DataFrame:
        """Phase 2で作成した相関認識データセットを読み込み"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {self.data_path}")

        logger.info(f"データを読み込み中: {self.data_path}")
        df = pd.read_csv(self.data_path)

        # timestampをdatetimeに変換
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"データ読み込み完了: {len(df)} 行")
        return df

    def calculate_adaptive_reward(self, row: pd.Series, config_name: str = 'correlation_focused') -> float:
        """
        適応型報酬を計算

        相関認識特徴量に基づいて動的に報酬を調整します。
        """
        config = self.reward_configs[config_name]

        # 基本報酬（取引コストを考慮）
        base_reward = config['base_penalty']

        # 相関ボーナス
        correlation_score = row.get('market_correlation_score', 0)
        correlation_bonus = correlation_score * config['correlation_bonus']
        base_reward += correlation_bonus

        # レジーム特化調整
        regime = row.get('market_regime', 'unknown')
        regime_multiplier = self.get_regime_multiplier(regime, config)
        base_reward *= regime_multiplier

        # ボラティリティペナルティ
        volatility = row.get('volatility', 0.01)
        if volatility > 0.05:  # 高ボラティリティ時は慎重に
            volatility_penalty = config['volatility_penalty'] * (volatility / 0.05)
            base_reward += volatility_penalty

        # 価格位置相関ボーナス
        price_position_corr = row.get('price_position_corr', 0)
        if abs(price_position_corr) > 0.5:  # 強いトレンド相関
            trend_bonus = price_position_corr * 0.02
            base_reward += trend_bonus

        # アクション価格相関ボーナス
        action_price_corr = row.get('action_price_corr', 0)
        if abs(action_price_corr) > 0.7:  # 高い予測精度
            prediction_bonus = action_price_corr * 0.03
            base_reward += prediction_bonus

        return base_reward

    def get_regime_multiplier(self, regime: str, config: Dict) -> float:
        """市場レジームに応じた報酬倍率を取得"""
        regime_multipliers = {
            'strong_bull': 1.5,    # 強気市場では積極的に報酬
            'moderate_bull': 1.2,  # 中程度の強気
            'sideways': 0.8,       # 横ばいでは慎重に
            'moderate_bear': 1.2,  # 中程度の弱気
            'strong_bear': 1.5,    # 強気市場では積極的に報酬
            'high_volatility': 0.5, # 高ボラティリティでは大幅減
            'low_volatility': 1.1   # 低ボラティリティでは軽く増
        }

        base_multiplier = regime_multipliers.get(regime, 1.0)
        return base_multiplier * config['regime_multiplier']

    def create_reward_curriculum(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        報酬カリキュラムを作成

        学習段階に応じて異なる報酬設定を提供します。
        """
        logger.info("報酬カリキュラムを作成中...")

        curriculum = {}

        for config_name, config in self.reward_configs.items():
            logger.info(f"カリキュラムステージ '{config_name}' を処理中...")

            # サンプルデータで報酬を計算
            sample_rewards = []
            regime_rewards = {}

            for idx, row in df.iterrows():
                reward = self.calculate_adaptive_reward(row, config_name)
                sample_rewards.append(reward)

                regime = row.get('market_regime', 'unknown')
                if regime not in regime_rewards:
                    regime_rewards[regime] = []
                regime_rewards[regime].append(reward)

                # パフォーマンスのために最初の1000サンプルのみ使用
                if idx >= 1000:
                    break

            # 統計を計算
            curriculum[config_name] = {
                'config': config,
                'reward_stats': {
                    'mean': np.mean(sample_rewards),
                    'std': np.std(sample_rewards),
                    'min': np.min(sample_rewards),
                    'max': np.max(sample_rewards),
                    'positive_ratio': np.mean([r > 0 for r in sample_rewards])
                },
                'regime_stats': {
                    regime: {
                        'mean': np.mean(rewards),
                        'count': len(rewards)
                    }
                    for regime, rewards in regime_rewards.items()
                }
            }

        logger.info("報酬カリキュラム作成完了")
        return curriculum

    def optimize_reward_parameters(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        報酬パラメータを最適化

        相関スコアを最大化するパラメータを探索します。
        """
        logger.info("報酬パラメータ最適化を開始...")

        # 最適化対象パラメータ
        param_grid = {
            'correlation_bonus': [0.05, 0.1, 0.15, 0.2],
            'regime_multiplier': [1.0, 1.5, 2.0, 2.5],
            'volatility_penalty': [-0.005, -0.01, -0.02, -0.05]
        }

        best_params = {}
        best_score = -np.inf

        # グリッドサーチ（簡易版）
        for corr_bonus in param_grid['correlation_bonus']:
            for regime_mult in param_grid['regime_multiplier']:
                for vol_penalty in param_grid['volatility_penalty']:

                    # パラメータ設定
                    test_config = {
                        'base_penalty': -0.005,
                        'correlation_bonus': corr_bonus,
                        'regime_multiplier': regime_mult,
                        'volatility_penalty': vol_penalty
                    }

                    # 報酬計算と評価
                    total_correlation = 0
                    count = 0

                    for idx, row in df.iterrows():
                        reward = self.calculate_adaptive_reward_with_config(row, test_config)
                        correlation = row.get('market_correlation_score', 0)
                        total_correlation += abs(correlation) * reward
                        count += 1

                        if count >= 1000:  # サンプル制限
                            break

                    avg_correlation = total_correlation / count if count > 0 else 0

                    if avg_correlation > best_score:
                        best_score = avg_correlation
                        best_params = test_config.copy()

        optimized_config = {
            'optimized_parameters': best_params,
            'best_correlation_score': best_score,
            'optimization_method': 'grid_search',
            'sample_size': 1000
        }

        logger.info(f"パラメータ最適化完了: 最高相関スコア = {best_score:.4f}")
        return optimized_config

    def calculate_adaptive_reward_with_config(self, row: pd.Series, config: Dict) -> float:
        """指定された設定で適応型報酬を計算"""
        # 基本報酬
        base_reward = config['base_penalty']

        # 相関ボーナス
        correlation_score = row.get('market_correlation_score', 0)
        correlation_bonus = correlation_score * config['correlation_bonus']
        base_reward += correlation_bonus

        # レジーム特化調整
        regime = row.get('market_regime', 'unknown')
        regime_multiplier = self.get_regime_multiplier(regime, config)
        base_reward *= regime_multiplier

        # ボラティリティペナルティ
        volatility = row.get('volatility', 0.01)
        if volatility > 0.05:
            volatility_penalty = config['volatility_penalty'] * (volatility / 0.05)
            base_reward += volatility_penalty

        return base_reward

    def save_reward_system(self, curriculum: Dict, optimization: Dict) -> None:
        """適応型報酬システムを保存"""
        logger.info(f"適応型報酬システムを保存中: {self.output_path}")

        reward_system = {
            'version': 'v426',
            'phase': 3,
            'description': 'Adaptive Reward System for SAC v426',
            'curriculum_stages': curriculum,
            'optimized_config': optimization,
            'created_at': pd.Timestamp.now().isoformat(),
            'target_improvement': {
                'from_adaptability': 0.262,  # SAC v424
                'target_adaptability': 0.8,   # v426目標
                'correlation_target': 0.1
            }
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(reward_system, f, indent=2, ensure_ascii=False)

        logger.info("適応型報酬システム保存完了")

    def generate_reward_report(self, curriculum: Dict, optimization: Dict) -> None:
        """報酬システムレポートを生成"""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write("# Phase 3: Adaptive Reward System Report\n\n")
            f.write("## SAC v426 Adaptive Reward Implementation\n\n")

            f.write("### 目標\n")
            f.write("- SAC v424の適応性不足解決（適応性 0.262 → 0.8+）\n")
            f.write("- 相関認識特徴量に基づく動的報酬調整\n")
            f.write("- レジーム特化型学習システム\n\n")

            f.write("### カリキュラムステージ\n\n")

            for stage_name, stage_data in curriculum.items():
                f.write(f"#### {stage_name.replace('_', ' ').title()}\n")
                stats = stage_data['reward_stats']
                f.write(f"- 平均報酬: {stats['mean']:.6f}\n")
                f.write(f"- 報酬範囲: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"- 正報酬比率: {stats['positive_ratio']:.2%}\n")
                f.write(f"- 相関ボーナス: {stage_data['config']['correlation_bonus']}\n")
                f.write(f"- レジーム倍率: {stage_data['config']['regime_multiplier']}\n\n")

                f.write("レジーム別平均報酬:\n")
                for regime, r_stats in stage_data['regime_stats'].items():
                    f.write(f"- {regime}: {r_stats['mean']:.6f} ({r_stats['count']} サンプル)\n")
                f.write("\n")

            f.write("### パラメータ最適化結果\n\n")
            opt = optimization
            f.write(f"- 最適相関スコア: {opt['best_correlation_score']:.4f}\n")
            f.write(f"- 最適化手法: {opt['optimization_method']}\n")
            f.write(f"- サンプルサイズ: {opt['sample_size']}\n\n")

            f.write("最適パラメータ:\n")
            for param, value in opt['optimized_parameters'].items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")

            f.write("### 次のステップ\n")
            f.write("- Phase 4: SAC v426学習実装\n")
            f.write("- Phase 5: 包括的評価と検証\n")
            f.write("- 適応性目標: 0.8以上\n\n")

        logger.info(f"報酬レポート生成完了: {self.report_path}")

    def run_phase3(self) -> None:
        """Phase 3の完全な実行"""
        logger.info("=== Phase 3: Adaptive Reward System開始 ===")

        try:
            # 1. データ読み込み
            df = self.load_data()

            # 2. 報酬カリキュラム作成
            curriculum = self.create_reward_curriculum(df)

            # 3. パラメータ最適化
            optimization = self.optimize_reward_parameters(df)

            # 4. 報酬システム保存
            self.save_reward_system(curriculum, optimization)

            # 5. レポート生成
            self.generate_reward_report(curriculum, optimization)

            logger.info("=== Phase 3: Adaptive Reward System完了 ===")
            logger.info(f"出力ファイル: {self.output_path}")
            logger.info(f"最適相関スコア: {optimization['best_correlation_score']:.4f}")

        except Exception as e:
            logger.error(f"Phase 3実行中にエラー発生: {e}")
            raise

def main():
    """メイン実行関数"""
    reward_system = AdaptiveRewardSystem()
    reward_system.run_phase3()

if __name__ == "__main__":
    main()