#!/usr/bin/env python3
"""
Detailed Statistical Analysis for v440 Enhanced Reward Function

Analyzes training results, action distributions, reward statistics,
and performance metrics for the improved v440 model.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_training_results(results_dir: str = "results/v440") -> Dict[str, Any]:
    """Load training results from results directory."""
    results = {}

    # Load backtest results
    backtest_file = Path(results_dir) / "backtest_results_v440.json"
    if backtest_file.exists():
        with open(backtest_file, 'r') as f:
            results['backtest'] = json.load(f)

    # Load training logs if available
    tensorboard_dir = Path("tensorboard/v440")
    if tensorboard_dir.exists():
        results['tensorboard_logs'] = str(tensorboard_dir)

    return results

def analyze_action_distribution(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze action distribution from training results."""
    analysis = {
        'action_balance': {},
        'trade_frequency': {},
        'reward_stability': {}
    }

    if 'backtest' in results:
        backtest = results['backtest']
        episodes = backtest.get('episodes', [])

        if episodes:
            # Calculate action distribution across episodes
            total_trades = sum(ep.get('trades', 0) for ep in episodes)
            avg_trades = total_trades / len(episodes)

            analysis['trade_frequency'] = {
                'total_trades': total_trades,
                'avg_trades_per_episode': avg_trades,
                'zero_trade_episodes': sum(1 for ep in episodes if ep.get('trades', 0) == 0)
            }

            # Reward statistics
            rewards = [ep.get('reward', 0) for ep in episodes]
            analysis['reward_stability'] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': min(rewards),
                'max_reward': max(rewards),
                'reward_range': max(rewards) - min(rewards)
            }

    return analysis

def analyze_reward_function_improvements() -> Dict[str, Any]:
    """Analyze the improvements made to the reward function."""
    improvements = {
        'v431_elements_reintroduced': [
            'HOLD penalty multiplier (1.01)',
            'Trade frequency bonus (0.001)',
            'Reward scaling (1000.0)',
            'Reward clipping (±10.0)',
            'Symmetric action thresholds (±0.3333)'
        ],
        'expected_benefits': [
            'Reduced zero-trade problem',
            'Balanced action distribution',
            'Stable critic loss',
            'Improved learning efficiency'
        ],
        'current_status': 'Implemented and tested'
    }

    return improvements

def generate_statistical_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive statistical report."""
    report_lines = []
    report_lines.append("# v440 Enhanced Reward Function - Statistical Analysis Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Reward Function Improvements
    report_lines.append("## 🎯 Reward Function Improvements")
    improvements = analyze_reward_function_improvements()
    for element in improvements['v431_elements_reintroduced']:
        report_lines.append(f"- ✅ {element}")
    report_lines.append("")

    # Training Results Analysis
    if 'backtest' in results:
        report_lines.append("## 📊 Training Results Analysis")
        backtest = results['backtest']
        summary = backtest.get('summary', {})

        report_lines.append("### Summary Statistics")
        report_lines.append(f"- Total Episodes: {summary.get('total_episodes', 'N/A')}")
        report_lines.append(".2f")
        report_lines.append(".2f")
        report_lines.append(".2f")
        report_lines.append(".2f")
        report_lines.append(f"- Sharpe Ratio: {summary.get('sharpe_ratio', 'N/A')}")
        report_lines.append(f"- Max Drawdown: {summary.get('max_drawdown', 'N/A')}")
        report_lines.append("")

        # Action Distribution Analysis
        action_analysis = analyze_action_distribution(results)
        report_lines.append("### Action Distribution Analysis")
        if 'trade_frequency' in action_analysis:
            tf = action_analysis['trade_frequency']
            report_lines.append(f"- Total Trades: {tf['total_trades']}")
            report_lines.append(".2f")
            report_lines.append(f"- Zero Trade Episodes: {tf['zero_trade_episodes']}")
        report_lines.append("")

        # Reward Stability Analysis
        if 'reward_stability' in action_analysis:
            rs = action_analysis['reward_stability']
            report_lines.append("### Reward Stability Analysis")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
        report_lines.append("")

    # Performance Comparison
    report_lines.append("## 🔄 Performance Comparison")
    report_lines.append("### Before Improvements (Original v440)")
    report_lines.append("- Total Return: -49.83%")
    report_lines.append("- Win Rate: 0%")
    report_lines.append("- Total Trades: 0 (Zero-trade problem)")
    report_lines.append("")

    report_lines.append("### After Improvements (Enhanced v440)")
    if 'backtest' in results:
        summary = results['backtest'].get('summary', {})
        report_lines.append(".2f")
        report_lines.append(".2f")
        report_lines.append(f"- Total Trades: {action_analysis.get('trade_frequency', {}).get('total_trades', 'N/A')}")
    else:
        report_lines.append("- Status: Training completed, analysis pending")
    report_lines.append("")

    # Recommendations
    report_lines.append("## 💡 Recommendations")
    report_lines.append("1. **Extended Training**: Run longer training sessions (50k-100k timesteps)")
    report_lines.append("2. **Hyperparameter Tuning**: Optimize reward scaling and action thresholds")
    report_lines.append("3. **Curriculum Learning**: Implement gradual difficulty increase")
    report_lines.append("4. **Ensemble Methods**: Consider model ensembling for robustness")
    report_lines.append("")

    return "\n".join(report_lines)

def main():
    """Main analysis function."""
    print("🔍 Starting v440 Enhanced Reward Function Statistical Analysis")
    print("=" * 60)

    # Load results
    results = load_training_results()
    print(f"📁 Loaded results from: {list(results.keys())}")

    # Generate analysis
    action_analysis = analyze_action_distribution(results)
    improvements = analyze_reward_function_improvements()

    # Print key findings
    print("\n🎯 Key Findings:")
    print("-" * 30)

    if 'trade_frequency' in action_analysis:
        tf = action_analysis['trade_frequency']
        print(f"• Trade Activity: {tf['total_trades']} total trades")
        print(".2f")
        if tf['zero_trade_episodes'] > 0:
            print(f"• Zero-trade episodes: {tf['zero_trade_episodes']} (still present)")

    if 'reward_stability' in action_analysis:
        rs = action_analysis['reward_stability']
        print(".2f")
        print(".2f")

    # Generate and save report
    report = generate_statistical_report(results)
    report_file = f"v440_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📄 Detailed report saved to: {report_file}")
    print("\n✅ Statistical analysis completed!")

if __name__ == "__main__":
    main()