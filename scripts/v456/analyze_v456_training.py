#!/usr/bin/env python3
"""
v456 訓練結果解析スクリプト

v455 以前の分析手法を参考にしながら、v456 訓練結果を包括的に分析
- 報酬曲線分析
- 学習進捗分析
- モデル性能評価
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# セットアップ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_training_log(log_file: str = "training_50k_log.txt") -> List[Dict]:
    """訓練ログを解析してマイルストーン情報を抽出"""
    logger.info(f"Loading training log from {log_file}")
    
    milestones = []
    
    try:
        with open(log_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                # Milestone パターンの抽出
                # Example: "⏱️  Milestone 5,000 steps | Avg Reward: -6.3611 | Episodes: 5"
                if "Milestone" in line and "Avg Reward" in line:
                    try:
                        # ログの形式に対応: "... Milestone X,XXX steps | Avg Reward: Y | Episodes: Z"
                        if '|' in line:
                            parts = line.split('|')
                            # parts[0]: "... Milestone X,XXX steps"
                            # parts[1]: " Avg Reward: Y "
                            # parts[2]: " Episodes: Z"
                            
                            step_part = parts[0].split()[-2].replace(",", "")
                            reward_part = parts[1].split(':')[-1].strip()
                            episodes_part = parts[2].split(':')[-1].strip()
                            
                            step = int(step_part)
                            reward = float(reward_part)
                            episodes = int(episodes_part)
                            
                            milestones.append({
                                'step': step,
                                'avg_reward': reward,
                                'episodes': episodes
                            })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse line: {line[:80]}")
                        continue
    except FileNotFoundError:
        logger.warning(f"Log file not found: {log_file}")
        return []
    
    logger.info(f"Extracted {len(milestones)} milestone entries")
    return milestones


def analyze_reward_trajectory(milestones: List[Dict]) -> Dict:
    """報酬曲線の分析"""
    logger.info("Analyzing reward trajectory...")
    
    if not milestones:
        logger.warning("No milestones to analyze")
        return {}
    
    df = pd.DataFrame(milestones)
    
    analysis = {
        'total_milestones': len(df),
        'total_steps': df['step'].max() if len(df) > 0 else 0,
        'initial_reward': df['avg_reward'].iloc[0] if len(df) > 0 else 0,
        'final_reward': df['avg_reward'].iloc[-1] if len(df) > 0 else 0,
        'max_reward': df['avg_reward'].max(),
        'min_reward': df['avg_reward'].min(),
        'mean_reward': df['avg_reward'].mean(),
        'std_reward': df['avg_reward'].std(),
        'total_episodes': df['episodes'].max() if len(df) > 0 else 0,
    }
    
    # 改善度を計算
    if analysis['initial_reward'] != 0:
        improvement = (
            (analysis['final_reward'] - analysis['initial_reward']) / 
            abs(analysis['initial_reward']) * 100
        )
        analysis['reward_improvement_pct'] = improvement
    else:
        analysis['reward_improvement_pct'] = 0
    
    # 報酬の増減傾向
    if len(df) > 10:
        early_rewards = df['avg_reward'].iloc[:len(df)//3].mean()
        late_rewards = df['avg_reward'].iloc[-len(df)//3:].mean()
        analysis['early_avg_reward'] = early_rewards
        analysis['late_avg_reward'] = late_rewards
        analysis['convergence_trend'] = 'improving' if late_rewards > early_rewards else 'degrading'
    
    return analysis, df


def create_visualization(df: pd.DataFrame, analysis: Dict, output_dir: str = "analysis_results"):
    """訓練結果の可視化"""
    logger.info("Creating visualizations...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('v456 Training Analysis (50,000 steps)', fontsize=16, fontweight='bold')
    
    # 1. 報酬曲線
    ax = axes[0, 0]
    ax.plot(df['step'], df['avg_reward'], marker='o', markersize=3, linewidth=1.5, label='Avg Reward')
    ax.fill_between(df['step'], df['avg_reward'].min(), df['avg_reward'].max(), alpha=0.1)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('Reward Trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. 報酬分布
    ax = axes[0, 1]
    ax.hist(df['avg_reward'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(analysis['mean_reward'], color='red', linestyle='--', label='Mean')
    ax.set_xlabel('Average Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    
    # 3. エピソード数
    ax = axes[1, 0]
    ax.plot(df['step'], df['episodes'], marker='s', markersize=3, linewidth=1.5, color='green')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Total Episodes')
    ax.set_title('Episode Progress')
    ax.grid(True, alpha=0.3)
    
    # 4. 統計情報（テキスト）
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
Training Statistics
─────────────────────
Total Steps: {analysis['total_steps']:,}
Total Episodes: {analysis['total_episodes']}
Milestones: {analysis['total_milestones']}

Reward Metrics
─────────────────────
Initial: {analysis['initial_reward']:.4f}
Final: {analysis['final_reward']:.4f}
Improvement: {analysis['reward_improvement_pct']:.2f}%
Mean: {analysis['mean_reward']:.4f}
Std Dev: {analysis['std_reward']:.4f}

Performance
─────────────────────
Max Reward: {analysis['max_reward']:.4f}
Min Reward: {analysis['min_reward']:.4f}
Trend: {analysis.get('convergence_trend', 'N/A')}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
    
    # 保存
    plot_path = Path(output_dir) / "v456_training_analysis.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved: {plot_path}")
    
    plt.close()


def generate_report(analysis: Dict, df: pd.DataFrame, output_dir: str = "analysis_results") -> str:
    """詳細レポートを生成"""
    logger.info("Generating analysis report...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report = f"""
# v456 訓練結果解析レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 訓練概要

- **総ステップ数**: {analysis['total_steps']:,}
- **総エピソード数**: {analysis['total_episodes']}
- **マイルストーン数**: {analysis['total_milestones']}

## 報酬分析

### 主要指標
- **初期報酬**: {analysis['initial_reward']:.4f}
- **最終報酬**: {analysis['final_reward']:.4f}
- **改善度**: {analysis['reward_improvement_pct']:.2f}%
- **最大報酬**: {analysis['max_reward']:.4f}
- **最小報酬**: {analysis['min_reward']:.4f}
- **平均報酬**: {analysis['mean_reward']:.4f}
- **標準偏差**: {analysis['std_reward']:.4f}

### 学習進捗

"""
    
    if 'convergence_trend' in analysis:
        report += f"- **収束傾向**: {analysis['convergence_trend'].upper()}\n"
        if 'early_avg_reward' in analysis:
            report += f"  - 初期平均: {analysis['early_avg_reward']:.4f}\n"
            report += f"  - 最終平均: {analysis['late_avg_reward']:.4f}\n"
    
    report += f"""

## 詳細統計

### 報酬の変動
- **範囲**: [{analysis['min_reward']:.4f}, {analysis['max_reward']:.4f}]
- **中央値**: {df['avg_reward'].median():.4f}
- **四分位範囲（IQR）**: {df['avg_reward'].quantile(0.75) - df['avg_reward'].quantile(0.25):.4f}

### エピソード統計
- **最小エピソード数**: {df['episodes'].min()}
- **最大エピソード数**: {df['episodes'].max()}
- **平均エピソード数**: {df['episodes'].mean():.1f}

## マイルストーン詳細

| Step | Avg Reward | Episodes |
|------|-----------|----------|
"""
    
    # 均等間隔でサンプリング（全部は多すぎるため）
    sample_indices = np.linspace(0, len(df) - 1, min(20, len(df)), dtype=int)
    for idx in sample_indices:
        row = df.iloc[idx]
        report += f"| {int(row['step']):,} | {row['avg_reward']:.4f} | {int(row['episodes'])} |\n"
    
    report += f"""

## 学習品質評価

### 分析結果
"""
    
    # 品質評価
    if analysis['reward_improvement_pct'] > 0:
        report += f"✅ **正の改善傾向**: 報酬が {analysis['reward_improvement_pct']:.2f}% 改善\n"
    else:
        report += f"⚠️  **負の傾向**: 報酬が {abs(analysis['reward_improvement_pct']):.2f}% 悪化\n"
    
    if analysis['std_reward'] < abs(analysis['mean_reward']) * 0.5:
        report += "✅ **安定性**: 報酬の変動が低く、モデルが安定\n"
    else:
        report += "⚠️  **不安定性**: 報酬の変動が大きい\n"
    
    convergence = analysis.get('convergence_trend', 'N/A')
    if convergence == 'improving':
        report += "✅ **収束傾向**: モデルが継続的に改善\n"
    elif convergence == 'degrading':
        report += "⚠️  **退化傾向**: モデルの性能が低下\n"
    
    report += """

## 推奨事項

### 次のステップ
1. さらなる訓練の継続（100,000+ ステップ）
2. ハイパーパラメータのチューニング
3. 報酬関数の調整検討

## 技術情報

- **フレームワーク**: Stable-Baselines3 (SAC)
- **最適化**: Phase 1-3 統合済み
- **環境**: FastIntradayEnvV456 (特徴量: 88次元)
"""
    
    # ファイルに保存
    report_path = Path(output_dir) / "v456_training_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved: {report_path}")
    return report


def save_analysis_json(analysis: Dict, df: pd.DataFrame, output_dir: str = "analysis_results"):
    """分析結果を JSON で保存"""
    logger.info("Saving analysis data as JSON...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    analysis_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'version': 'v456',
            'model': 'SAC'
        },
        'summary': analysis,
        'milestones': df.to_dict('records')
    }
    
    json_path = Path(output_dir) / "v456_training_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON saved: {json_path}")


def main():
    """メイン処理"""
    logger.info("=" * 70)
    logger.info("v456 Training Analysis")
    logger.info("=" * 70)
    
    # ログ読み込み
    milestones = load_training_log("training_50k_log.txt")
    
    if not milestones:
        logger.error("No training data found. Please ensure training has completed.")
        return
    
    # 分析実行
    analysis, df = analyze_reward_trajectory(milestones)
    
    # 出力フォルダ作成
    output_dir = "analysis_results/v456"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 可視化
    create_visualization(df, analysis, output_dir)
    
    # レポート生成
    report = generate_report(analysis, df, output_dir)
    
    # JSON 保存
    save_analysis_json(analysis, df, output_dir)
    
    # 結果表示
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    print(report)
    
    logger.info("=" * 70)
    logger.info("Analysis completed successfully!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
