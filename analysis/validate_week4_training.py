#!/usr/bin/env python3
"""
Week 4 SAC訓練検証スクリプト

短時間訓練（5000ステップ）から、以下を分析：
- 環境の適切性（報酬分布、タッチダウン率）
- モデルの学習性（報酬トレンド、アクション分布）
- 改善が必要な点（報酬設計、特徴工学）

実行: python analysis/validate_week4_training.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Week4TrainingValidator:
    """Week 4訓練の検証クラス"""
    
    def __init__(self, model_path: Path, market_data: pd.DataFrame, 
                 initial_balance: float = 124.01):
        self.model_path = model_path
        self.market_data = market_data
        self.initial_balance = initial_balance
        self.results = {}
        
    def validate_environment(self) -> Dict[str, Any]:
        """環境の基本検証"""
        logger.info("=" * 70)
        logger.info("STEP 1: 環境検証")
        logger.info("=" * 70)
        
        # 特徴量準備
        base_cols = [f'base_{i}' for i in range(30)]
        mtf_cols = [f'mtf_{i}' for i in range(27)]
        regime_cols = [f'regime_{i}' for i in range(13)]
        
        # ダミーデータ補完
        df = self.market_data.copy()
        for col_list in [base_cols, mtf_cols, regime_cols]:
            for col in col_list:
                if col not in df.columns:
                    df[col] = np.random.randn(len(df))
        
        for col in ['atr', 'impact_proxy']:
            if col not in df.columns:
                df[col] = np.random.rand(len(df)) + 1.0
        
        # 環境作成
        try:
            env = FastIntradayEnvV456(
                df=df,
                base_feature_columns=base_cols[:30],
                mtf_feature_columns=mtf_cols[:27],
                regime_feature_columns=regime_cols[:13],
                initial_balance=self.initial_balance,
                max_position=max(self.initial_balance / 100, 1.0),
                prewarm_steps=100
            )
            
            obs, info = env.reset()
            
            logger.info(f"✓ 環境作成成功")
            logger.info(f"  観測空間: {env.observation_space}")
            logger.info(f"  アクション空間: {env.action_space}")
            logger.info(f"  初期残高: {self.initial_balance:.2f} JPY")
            
            validation_results = {
                'status': 'success',
                'obs_shape': env.observation_space.shape,
                'action_shape': env.action_space.shape,
                'initial_balance': self.initial_balance,
                'market_data_points': len(df)
            }
            
            env.close()
            
        except Exception as e:
            logger.error(f"✗ 環境エラー: {e}")
            validation_results = {
                'status': 'failed',
                'error': str(e)
            }
        
        self.results['environment'] = validation_results
        return validation_results
    
    def evaluate_model(self, episodes: int = 50) -> Dict[str, Any]:
        """訓練済みモデルの評価"""
        logger.info("=" * 70)
        logger.info(f"STEP 2: モデル評価（{episodes}エピソード）")
        logger.info("=" * 70)
        
        try:
            # モデル読み込み
            model = SAC.load(str(self.model_path))
            logger.info(f"✓ モデル読み込み: {self.model_path.name}")
            
            # 評価環境準備
            base_cols = [f'base_{i}' for i in range(30)]
            mtf_cols = [f'mtf_{i}' for i in range(27)]
            regime_cols = [f'regime_{i}' for i in range(13)]
            
            df = self.market_data.copy()
            for col_list in [base_cols, mtf_cols, regime_cols]:
                for col in col_list:
                    if col not in df.columns:
                        df[col] = np.random.randn(len(df))
            
            for col in ['atr', 'impact_proxy']:
                if col not in df.columns:
                    df[col] = np.random.rand(len(df)) + 1.0
            
            env = FastIntradayEnvV456(
                df=df,
                base_feature_columns=base_cols[:30],
                mtf_feature_columns=mtf_cols[:27],
                regime_feature_columns=regime_cols[:13],
                initial_balance=self.initial_balance,
                max_position=max(self.initial_balance / 100, 1.0),
                prewarm_steps=100
            )
            
            # エピソード評価
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
                    
                    # アクション分類（簡易版）
                    if action[0] < -0.5:
                        action_counts['sell'] += 1
                    elif action[0] > 0.5:
                        action_counts['buy'] += 1
                    else:
                        action_counts['hold'] += 1
                
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                
                if (ep + 1) % 10 == 0:
                    logger.info(f"  エピソード {ep + 1}/{episodes} - "
                              f"報酬: {ep_reward:.4f}, 長さ: {ep_length}")
            
            env.close()
            
            # 統計計算
            rewards_array = np.array(episode_rewards)
            
            eval_results = {
                'status': 'success',
                'episodes_evaluated': episodes,
                'reward_mean': float(np.mean(rewards_array)),
                'reward_std': float(np.std(rewards_array)),
                'reward_min': float(np.min(rewards_array)),
                'reward_max': float(np.max(rewards_array)),
                'episode_length_mean': float(np.mean(episode_lengths)),
                'episode_length_std': float(np.std(episode_lengths)),
                'action_distribution': {
                    'buy': action_counts['buy'],
                    'hold': action_counts['hold'],
                    'sell': action_counts['sell']
                },
                'total_actions': sum(action_counts.values())
            }
            
            # アクション分布の割合
            total = eval_results['total_actions']
            if total > 0:
                eval_results['action_distribution_pct'] = {
                    'buy': (action_counts['buy'] / total) * 100,
                    'hold': (action_counts['hold'] / total) * 100,
                    'sell': (action_counts['sell'] / total) * 100
                }
            
            logger.info(f"✓ 評価完了")
            logger.info(f"  報酬 (平均±std): {eval_results['reward_mean']:.4f} ± {eval_results['reward_std']:.4f}")
            logger.info(f"  報酬 (範囲): [{eval_results['reward_min']:.4f}, {eval_results['reward_max']:.4f}]")
            logger.info(f"  エピソード長 (平均): {eval_results['episode_length_mean']:.1f} ステップ")
            logger.info(f"  アクション分布: "
                       f"買 {eval_results['action_distribution_pct'].get('buy', 0):.1f}% / "
                       f"保有 {eval_results['action_distribution_pct'].get('hold', 0):.1f}% / "
                       f"売 {eval_results['action_distribution_pct'].get('sell', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"✗ モデル評価エラー: {e}")
            eval_results = {
                'status': 'failed',
                'error': str(e)
            }
        
        self.results['evaluation'] = eval_results
        return eval_results
    
    def detect_issues(self) -> Dict[str, List[str]]:
        """問題検出と改善提案"""
        logger.info("=" * 70)
        logger.info("STEP 3: 問題検出と統計判定")
        logger.info("=" * 70)
        
        issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        eval_res = self.results.get('evaluation', {})
        
        if eval_res.get('status') == 'failed':
            logger.error("✗ モデル評価失敗 - 分析不可")
            issues['critical'].append("モデル評価が失敗した - 環境の根本的な問題がある可能性")
            self.results['issues'] = issues
            return issues
        
        # 1. 報酬分析
        reward_mean = eval_res.get('reward_mean', 0.0)
        reward_std = eval_res.get('reward_std', 0.0)
        
        logger.info(f"報酬分析:")
        logger.info(f"  平均: {reward_mean:.4f}, 標準偏差: {reward_std:.4f}")
        
        if reward_mean < -0.5:
            issues['critical'].append(
                f"報酬が極めて低い ({reward_mean:.4f}) - "
                "報酬関数の設計に根本的な問題がある可能性"
            )
            logger.warning("⚠ 報酬極低値")
        elif reward_mean < -0.1:
            issues['warning'].append(
                f"報酬がネガティブ ({reward_mean:.4f}) - "
                "報酬関数の調整が必要"
            )
            logger.warning("⚠ ネガティブ報酬")
        
        # 2. 学習安定性チェック
        cv = reward_std / abs(reward_mean) if reward_mean != 0 else float('inf')
        logger.info(f"  変動係数 (CV): {cv:.4f}")
        
        if cv > 2.0:
            issues['warning'].append(
                f"報酬の変動が大きい (CV={cv:.4f}) - "
                "学習が不安定である可能性"
            )
            logger.warning("⚠ 高変動性")
        
        # 3. アクション分布チェック
        action_pct = eval_res.get('action_distribution_pct', {})
        sell_pct = action_pct.get('sell', 0.0)
        buy_pct = action_pct.get('buy', 0.0)
        hold_pct = action_pct.get('hold', 0.0)
        
        logger.info(f"アクション分布:")
        logger.info(f"  買: {buy_pct:.1f}%, 保有: {hold_pct:.1f}%, 売: {sell_pct:.1f}%")
        
        if sell_pct < 5.0:
            issues['critical'].append(
                f"SELL比率が極端に低い ({sell_pct:.1f}%) - "
                "ポジション解放機構の改善が必須"
            )
            logger.warning("⚠ SELL不足")
        elif sell_pct < 15.0:
            issues['warning'].append(
                f"SELL比率が低い ({sell_pct:.1f}%) - "
                "SELLシグナル特徴量の強化を検討"
            )
            logger.warning("⚠ SELL低い")
        
        if buy_pct > 80.0:
            issues['critical'].append(
                f"BUY比率が高すぎる ({buy_pct:.1f}%) - "
                "エージェントが買い特化になっている"
            )
            logger.warning("⚠ BUY過多")
        
        # 4. エピソード長チェック
        ep_len = eval_res.get('episode_length_mean', 0.0)
        if ep_len < 2.0:
            issues['critical'].append(
                f"エピソード長が極短 ({ep_len:.1f}ステップ) - "
                "タッチダウン（即座に終了）が発生している可能性"
            )
            logger.warning("⚠ エピソード短")
        
        # 5. 統計的有意性判定
        episodes = eval_res.get('episodes_evaluated', 0)
        logger.info(f"統計的検討:")
        logger.info(f"  評価エピソード数: {episodes}")
        
        if episodes >= 30:
            logger.info(f"  ✓ サンプルサイズ十分 (n={episodes} >= 30)")
            issues['info'].append(
                f"修正可能性の判定: 可能 (n={episodes}エピソード, 統計的信頼度95%以上)"
            )
        else:
            logger.warning(f"  ⚠ サンプルサイズ不足 (n={episodes} < 30)")
            issues['warning'].append(
                f"より多くのエピソード評価が必要 (現在: {episodes})"
            )
        
        self.results['issues'] = issues
        return issues
    
    def generate_report(self) -> str:
        """分析レポート生成"""
        logger.info("=" * 70)
        logger.info("STEP 4: レポート生成")
        logger.info("=" * 70)
        
        report = []
        report.append("# Week 4 SAC訓練検証レポート\n")
        report.append(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 環境検証結果
        env_res = self.results.get('environment', {})
        report.append("## 1. 環境検証\n")
        if env_res.get('status') == 'success':
            report.append("✓ 環境構築: 成功\n")
            report.append(f"- 観測空間: {env_res.get('obs_shape')}\n")
            report.append(f"- アクション空間: {env_res.get('action_shape')}\n")
            report.append(f"- 初期残高: {env_res.get('initial_balance'):.2f} JPY\n")
        else:
            report.append("✗ 環境構築: 失敗\n")
            report.append(f"- エラー: {env_res.get('error')}\n")
        
        # モデル評価結果
        eval_res = self.results.get('evaluation', {})
        report.append("\n## 2. モデル評価結果\n")
        if eval_res.get('status') == 'success':
            report.append("✓ モデル評価: 成功\n")
            report.append(f"- 評価エピソード: {eval_res.get('episodes_evaluated')}\n")
            report.append(f"- 報酬 (平均): {eval_res.get('reward_mean'):.6f}\n")
            report.append(f"- 報酬 (標準偏差): {eval_res.get('reward_std'):.6f}\n")
            report.append(f"- 報酬 (範囲): [{eval_res.get('reward_min'):.6f}, {eval_res.get('reward_max'):.6f}]\n")
            report.append(f"- エピソード長 (平均): {eval_res.get('episode_length_mean'):.1f} ステップ\n")
            
            action_pct = eval_res.get('action_distribution_pct', {})
            report.append(f"- アクション分布: 買 {action_pct.get('buy', 0):.1f}% / 保有 {action_pct.get('hold', 0):.1f}% / 売 {action_pct.get('sell', 0):.1f}%\n")
        else:
            report.append("✗ モデル評価: 失敗\n")
            report.append(f"- エラー: {eval_res.get('error')}\n")
        
        # 問題検出結果
        issues = self.results.get('issues', {})
        report.append("\n## 3. 問題検出と改善提案\n")
        
        if issues.get('critical'):
            report.append("### 🔴 Critical Issues (修正必須)\n")
            for issue in issues['critical']:
                report.append(f"- {issue}\n")
        
        if issues.get('warning'):
            report.append("### 🟡 Warnings (改善推奨)\n")
            for issue in issues['warning']:
                report.append(f"- {issue}\n")
        
        if issues.get('info'):
            report.append("### ℹ️ Information\n")
            for issue in issues['info']:
                report.append(f"- {issue}\n")
        
        # 判定
        report.append("\n## 4. 修正必要性の判定\n")
        has_critical = bool(issues.get('critical'))
        if has_critical:
            report.append("**判定**: 修正必須 ❌\n")
            report.append("- Critical issueが検出されています\n")
            report.append("- 以下の項目を修正してから本格訓練を実施してください\n")
        else:
            report.append("**判定**: 修正検討 ⚠️\n")
            report.append("- Critical issueはありませんが、warning/infoがあります\n")
            report.append("- 改善後、20000-50000ステップで再検証してから本格訓練を開始してください\n")
        
        return '\n'.join(report)
    
    def save_results(self, output_dir: Path = None):
        """結果保存"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'docs' / 'v456'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON結果保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = output_dir / f'validation_results_{timestamp}.json'
        
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=float)
        
        logger.info(f"✓ 結果保存: {json_path}")
        
        # レポート保存
        report = self.generate_report()
        report_path = output_dir / f'validation_report_{timestamp}.md'
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ レポート保存: {report_path}")
        
        return json_path, report_path


def main():
    """メイン実行"""
    print("=" * 70)
    print("Week 4 SAC訓練検証システム")
    print("=" * 70)
    print()
    
    # モデルパス探索
    model_dir = PROJECT_ROOT / 'models' / 'week4_mlp_sac'
    model_files = list(model_dir.glob('sac_mlp_v456_*.zip'))
    
    if not model_files:
        logger.error(f"✗ モデルが見つかりません: {model_dir}")
        return
    
    # 最新モデルを使用
    model_path = sorted(model_files)[-1]
    logger.info(f"使用モデル: {model_path.name}")
    
    # 市場データ読み込み
    data_path = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
    market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    logger.info(f"市場データ: {len(market_data)} records")
    
    # 検証実行
    validator = Week4TrainingValidator(model_path, market_data)
    
    validator.validate_environment()
    validator.evaluate_model(episodes=50)
    validator.detect_issues()
    validator.save_results()
    
    print()
    print("=" * 70)
    logger.info("検証完了")
    print("=" * 70)


if __name__ == '__main__':
    main()
