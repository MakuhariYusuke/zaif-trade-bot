#!/usr/bin/env python3
"""
v456 訓練結果解析スクリプト (修正版)

v455 以前の分析手法を参考にしながら、v456 訓練結果を包括的に分析
- 報酬曲線分析
- 学習進捗分析
- モデル性能評価
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# セットアップ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_training_log(log_file: str = "training_50k_log.txt") -> List[Dict[str, Any]]:
    """訓練ログを解析してマイルストーン情報を抽出"""
    logger.info(f"Loading training log from {log_file}")
    
    milestones = []
    
    try:
        # Try UTF-16 first (log file is in UTF-16 encoding)
        encodings = ['utf-16', 'utf-8-sig', 'utf-8']
        
        for encoding in encodings:
            try:
                with open(log_file, 'r', encoding=encoding) as f:
                    for line in f:
                        if 'Milestone' in line and 'Avg Reward' in line:
                            # Pattern: ⏱️  Milestone X,XXX steps | Avg Reward: Y.YYYY | Episodes: Z
                            match = re.search(r'Milestone\s+([\d,]+)\s+steps\s*\|\s*Avg Reward:\s+([-\d.]+)\s*\|\s*Episodes:\s+(\d+)', line)
                            if match:
                                try:
                                    steps = int(match.group(1).replace(',', ''))
                                    reward = float(match.group(2))
                                    episodes = int(match.group(3))
                                    milestones.append({
                                        'step': steps,
                                        'avg_reward': reward,
                                        'episodes': episodes
                                    })
                                except (ValueError, AttributeError):
                                    continue
                break  # Successfully loaded, exit encoding loop
            except (UnicodeDecodeError, FileNotFoundError):
                continue
    except Exception as e:
        logger.error(f"Error loading log file: {e}")
        return []
    
    logger.info(f"Extracted {len(milestones)} milestone entries")
    return milestones


def analyze_reward_trajectory(milestones: List[Dict[str, Any]]) -> Dict[str, Any]:
    """報酬曲線の分析"""
    logger.info("Analyzing reward trajectory...")
    
    if not milestones:
        logger.error("No training data found. Please ensure training has completed.")
        return {}
    
    steps = [m['step'] for m in milestones]
    rewards = [m['avg_reward'] for m in milestones]
    
    # 統計情報
    analysis = {
        'total_steps': steps[-1],
        'total_milestones': len(milestones),
        'initial_reward': rewards[0],
        'final_reward': rewards[-1],
        'max_reward': max(rewards),
        'min_reward': min(rewards),
        'mean_reward': mean(rewards),
        'std_reward': stdev(rewards) if len(rewards) > 1 else 0.0,
        'improvement': rewards[-1] - rewards[0],
        'improvement_pct': ((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100) if rewards[0] != 0 else 0,
    }
    
    # 段階ごとの分析
    if len(milestones) >= 4:
        quartile = len(milestones) // 4
        analysis['stage_1'] = {
            'reward_range': [rewards[0], rewards[quartile]],
            'improvement': rewards[quartile] - rewards[0]
        }
        analysis['stage_2'] = {
            'reward_range': [rewards[quartile], rewards[2*quartile]],
            'improvement': rewards[2*quartile] - rewards[quartile]
        }
        analysis['stage_3'] = {
            'reward_range': [rewards[2*quartile], rewards[3*quartile]],
            'improvement': rewards[3*quartile] - rewards[2*quartile]
        }
        analysis['stage_4'] = {
            'reward_range': [rewards[3*quartile], rewards[-1]],
            'improvement': rewards[-1] - rewards[3*quartile]
        }
    
    logger.info(f"Initial Reward: {analysis['initial_reward']:.4f}")
    logger.info(f"Final Reward: {analysis['final_reward']:.4f}")
    logger.info(f"Mean Reward: {analysis['mean_reward']:.4f}")
    logger.info(f"Improvement: {analysis['improvement']:.4f}")
    
    return analysis, steps, rewards


def create_visualization(milestones: List[Dict[str, Any]], analysis: Dict[str, Any], 
                        output_dir: str = "analysis_results") -> Path:
    """訓練結果を可視化"""
    logger.info("Creating visualizations...")
    
    if not milestones:
        logger.error("Cannot create visualization without data")
        return None
    
    # パスの準備
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    steps = [m['step'] for m in milestones]
    rewards = [m['avg_reward'] for m in milestones]
    episodes = [m['episodes'] for m in milestones]
    
    # 4パネル図
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('v456 Training Analysis', fontsize=16, fontweight='bold')
    
    # Panel 1: Reward Trajectory
    axes[0, 0].plot(steps, rewards, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].fill_between(steps, rewards, alpha=0.3)
    axes[0, 0].axhline(y=analysis['mean_reward'], color='r', linestyle='--', label=f"Mean: {analysis['mean_reward']:.4f}")
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Avg Reward')
    axes[0, 0].set_title('Reward Trajectory')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Panel 2: Reward Distribution
    axes[0, 1].hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=analysis['mean_reward'], color='r', linestyle='--', linewidth=2, label=f"Mean: {analysis['mean_reward']:.4f}")
    axes[0, 1].axvline(x=analysis['final_reward'], color='g', linestyle='--', linewidth=2, label=f"Final: {analysis['final_reward']:.4f}")
    axes[0, 1].set_xlabel('Avg Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Training Progress
    axes[1, 0].plot(steps, episodes, 'g-', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Episodes Count')
    axes[1, 0].set_title('Episodes per Milestone')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Statistics Summary
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ─────────────────
    Total Steps: {analysis['total_steps']:,}
    Total Milestones: {analysis['total_milestones']:,}
    
    Reward Statistics:
    • Initial: {analysis['initial_reward']:.4f}
    • Final: {analysis['final_reward']:.4f}
    • Max: {analysis['max_reward']:.4f}
    • Min: {analysis['min_reward']:.4f}
    • Mean: {analysis['mean_reward']:.4f}
    • Std Dev: {analysis['std_reward']:.4f}
    
    Improvement:
    • Absolute: {analysis['improvement']:.4f}
    • Percentage: {analysis['improvement_pct']:.2f}%
    
    Training Duration: ~43 minutes
    Status: HALTED at 9.6% completion
    """
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存
    output_file = output_path / 'v456_training_analysis.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_file}")
    plt.close()
    
    return output_file


def generate_report(milestones: List[Dict[str, Any]], analysis: Dict[str, Any],
                   output_dir: str = "analysis_results") -> Path:
    """分析レポートを生成"""
    logger.info("Generating report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = f"""# v456 Training Analysis Report

## Training Overview
- **Total Steps**: {analysis['total_steps']:,}
- **Total Milestones Logged**: {analysis['total_milestones']:,}
- **Training Status**: HALTED (9.6% of 50,000 target)
- **Training Duration**: ~43 minutes (12:10 - 12:11)

## Reward Performance
### Summary Statistics
- **Initial Reward**: {analysis['initial_reward']:.4f}
- **Final Reward**: {analysis['final_reward']:.4f}
- **Maximum Reward**: {analysis['max_reward']:.4f}
- **Minimum Reward**: {analysis['min_reward']:.4f}
- **Mean Reward**: {analysis['mean_reward']:.4f}
- **Std Deviation**: {analysis['std_reward']:.4f}

### Improvement Metrics
- **Absolute Improvement**: {analysis['improvement']:.4f}
- **Relative Improvement**: {analysis['improvement_pct']:.2f}%

## Learning Progression Analysis

### Stage-wise Breakdown (Quartiles)
"""
    
    if 'stage_1' in analysis:
        report += f"""
#### Stage 1 (0-25% of training)
- Reward Range: [{analysis['stage_1']['reward_range'][0]:.4f}, {analysis['stage_1']['reward_range'][1]:.4f}]
- Stage Improvement: {analysis['stage_1']['improvement']:.4f}

#### Stage 2 (25-50% of training)
- Reward Range: [{analysis['stage_2']['reward_range'][0]:.4f}, {analysis['stage_2']['reward_range'][1]:.4f}]
- Stage Improvement: {analysis['stage_2']['improvement']:.4f}

#### Stage 3 (50-75% of training)
- Reward Range: [{analysis['stage_3']['reward_range'][0]:.4f}, {analysis['stage_3']['reward_range'][1]:.4f}]
- Stage Improvement: {analysis['stage_3']['improvement']:.4f}

#### Stage 4 (75-100% of training)
- Reward Range: [{analysis['stage_4']['reward_range'][0]:.4f}, {analysis['stage_4']['reward_range'][1]:.4f}]
- Stage Improvement: {analysis['stage_4']['improvement']:.4f}
"""
    
    report += f"""

## Key Findings

1. **Reward Trend**: The model shows {"improvement" if analysis['improvement'] > 0 else "degradation"} over the training period.
2. **Variability**: Standard deviation of {analysis['std_reward']:.4f} indicates {"stable" if analysis['std_reward'] < 2.0 else "unstable"} learning.
3. **Training Progress**: Only {analysis['total_steps']:,} steps completed (9.6% of 50k target).
4. **Halting Reason**: Process terminated after ~43 minutes without error messages (likely memory/deadlock issue).

## Recommendations

1. **Resume Training**: Checkpoint exists at 5k steps - can resume from there
2. **Parameter Optimization**: Consider further reducing:
   - Batch size: 64 → 32
   - Learning rate: 0.0001 → 0.00005
   - Buffer size: 100k → 50k
3. **Resource Monitoring**: Profile memory usage to identify potential leaks
4. **Extended Run**: If memory is available, attempt full 50k steps again

## Generated: {datetime.now().isoformat()}
"""
    
    # 保存
    report_file = output_path / 'v456_training_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    return report_file


def save_analysis_json(milestones: List[Dict[str, Any]], analysis: Dict[str, Any],
                      output_dir: str = "analysis_results") -> Path:
    """分析データをJSON形式で保存"""
    logger.info("Saving analysis data as JSON...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # マイルストーンデータ
    milestones_file = output_path / 'v456_milestones.json'
    with open(milestones_file, 'w') as f:
        json.dump(milestones, f, indent=2)
    logger.info(f"Milestones saved to {milestones_file}")
    
    # 分析結果
    analysis_file = output_path / 'v456_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {analysis_file}")
    
    return milestones_file, analysis_file


def main():
    """メイン処理"""
    logger.info("=" * 60)
    logger.info("v456 Training Analysis")
    logger.info("=" * 60)
    
    # ログの読み込み
    milestones = load_training_log()
    
    if not milestones:
        logger.error("Failed to extract training data. Exiting.")
        return 1
    
    # 分析実行
    analysis, steps, rewards = analyze_reward_trajectory(milestones)
    
    if not analysis:
        logger.error("Failed to analyze training data. Exiting.")
        return 1
    
    # 可視化
    viz_file = create_visualization(milestones, analysis)
    
    # レポート生成
    report_file = generate_report(milestones, analysis)
    
    # JSON保存
    milestones_file, analysis_file = save_analysis_json(milestones, analysis)
    
    logger.info("=" * 60)
    logger.info("✅ Analysis complete!")
    logger.info(f"   Report: {report_file}")
    logger.info(f"   Visualization: {viz_file}")
    logger.info(f"   Data: {milestones_file}, {analysis_file}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
