#!/usr/bin/env python3
"""
SAC v444 Parameter Tuning Analysis
異なるBalance Penalty ScaleとAction Bonusesの効果を分析
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class ParameterAnalyzer:
    """パラメータ効果の分析"""
    
    def __init__(self):
        self.results = {}
        self.config_sets = {
            "original": {
                "balance_penalty": 1000.0,
                "buy_bonus": 5.0,
                "sell_bonus": 0.0,
                "hold_bonus": 0.0,
                "expected": {
                    "mean_reward": -9845,
                    "buy_ratio": 0.18,
                    "sell_ratio": 0.6685,
                    "hold_ratio": 0.1515
                }
            },
            "scale_200": {
                "balance_penalty": 200.0,
                "buy_bonus": 10.0,
                "sell_bonus": 5.0,
                "hold_bonus": 2.0,
                "target": {
                    "mean_reward": (-5000, -2000),
                    "buy_ratio": (0.25, 0.40),
                    "sell_ratio": (0.25, 0.40),
                    "hold_ratio": (0.20, 0.30)
                }
            },
            "scale_300": {
                "balance_penalty": 300.0,
                "buy_bonus": 15.0,
                "sell_bonus": 10.0,
                "hold_bonus": 3.0,
                "target": {
                    "mean_reward": (-4000, -1500),
                    "buy_ratio": (0.30, 0.45),
                    "sell_ratio": (0.30, 0.45),
                    "hold_ratio": (0.20, 0.35)
                }
            },
            "scale_500": {
                "balance_penalty": 500.0,
                "buy_bonus": 20.0,
                "sell_bonus": 15.0,
                "hold_bonus": 5.0,
                "target": {
                    "mean_reward": (-3000, -500),
                    "buy_ratio": (0.35, 0.50),
                    "sell_ratio": (0.35, 0.50),
                    "hold_ratio": (0.20, 0.40)
                }
            }
        }
    
    def calculate_penalty_impact(self) -> Dict:
        """
        balance_penalty の影響を計算
        Reward = PnL - balance_penalty * abs(buy_ratio - sell_ratio)
        """
        impact_analysis = {}
        
        # 仮定: PnL = -100 (保守的な推定)
        pnl = -100.0
        
        for config_name, config in self.config_sets.items():
            if "balance_penalty" not in config:
                continue
            
            penalty_scale = config["balance_penalty"]
            
            # Different BUY/SELL ratios
            ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            
            impact_analysis[config_name] = {
                "penalty_scale": penalty_scale,
                "impact_by_ratio": {}
            }
            
            for buy_ratio in ratios:
                sell_ratio = 0.5  # Fixed for analysis
                diff = abs(buy_ratio - sell_ratio)
                penalty = penalty_scale * diff
                total_reward = pnl - penalty
                
                impact_analysis[config_name]["impact_by_ratio"][f"{buy_ratio:.1f}"] = {
                    "buy_ratio": buy_ratio,
                    "sell_ratio": sell_ratio,
                    "difference": diff,
                    "penalty": penalty,
                    "total_reward": total_reward
                }
        
        return impact_analysis
    
    def analyze_action_bonuses(self) -> Dict:
        """
        Action Bonuses の効果分析
        """
        bonus_analysis = {}
        
        for config_name, config in self.config_sets.items():
            bonus_analysis[config_name] = {
                "buy_bonus": config.get("buy_bonus", 0),
                "sell_bonus": config.get("sell_bonus", 0),
                "hold_bonus": config.get("hold_bonus", 0),
                "total_bonus": config.get("buy_bonus", 0) + config.get("sell_bonus", 0) + config.get("hold_bonus", 0)
            }
        
        return bonus_analysis
    
    def generate_recommendations(self) -> str:
        """
        改善の推奨を生成
        """
        recommendations = []
        recommendations.append("="*80)
        recommendations.append("🎯 SAC v444 PARAMETER TUNING RECOMMENDATIONS")
        recommendations.append("="*80)
        
        # Penalty Impact Analysis
        penalty_impact = self.calculate_penalty_impact()
        recommendations.append("\n1️⃣ BALANCE PENALTY IMPACT ANALYSIS:")
        recommendations.append("-" * 80)
        
        for config_name, impact in penalty_impact.items():
            recommendations.append(f"\n  {config_name.upper()}:")
            recommendations.append(f"    • Penalty Scale: {impact['penalty_scale']}")
            
            # Show impact at key ratio differences
            for ratio_str, impact_data in list(impact['impact_by_ratio'].items())[:3]:
                recommendations.append(
                    f"    • Buy Ratio {ratio_str}: Penalty = {impact_data['penalty']:.1f}, "
                    f"Total Reward = {impact_data['total_reward']:.1f}"
                )
        
        # Action Bonuses Analysis
        bonus_analysis = self.analyze_action_bonuses()
        recommendations.append("\n\n2️⃣ ACTION BONUSES COMPARISON:")
        recommendations.append("-" * 80)
        
        for config_name, bonuses in bonus_analysis.items():
            recommendations.append(f"\n  {config_name.upper()}:")
            recommendations.append(f"    • BUY:  {bonuses['buy_bonus']:>6.1f}")
            recommendations.append(f"    • SELL: {bonuses['sell_bonus']:>6.1f}")
            recommendations.append(f"    • HOLD: {bonuses['hold_bonus']:>6.1f}")
            recommendations.append(f"    • TOTAL: {bonuses['total_bonus']:>6.1f}")
        
        # Testing Strategy
        recommendations.append("\n\n3️⃣ RECOMMENDED TESTING STRATEGY:")
        recommendations.append("-" * 80)
        recommendations.append("""
  Phase 1: Test scale_200 (Minimal Penalty)
    → Objective: Check if reward becomes positive/less negative
    → Expected: Reward -5000 to -2000, More balanced actions
    → Time: 3000 steps
    
  Phase 2: Test scale_300 (Moderate Penalty)
    → Objective: Find optimal balance between penalty and bonuses
    → Expected: Reward -4000 to -1500, BUY 30-45%, SELL 30-45%
    → Time: 3000 steps
    
  Phase 3: Test scale_500 (Higher Penalty)
    → Objective: Maximize action balance enforcement
    → Expected: Reward -3000 to -500, BUY 35-50%, SELL 35-50%
    → Time: 3000 steps
        """)
        
        # Success Criteria
        recommendations.append("\n4️⃣ SUCCESS CRITERIA (優先度順):")
        recommendations.append("-" * 80)
        recommendations.append("""
  ✅ Priority 1: Mean Reward Improvement
     • Current: -9845
     • Target: > -5000 (50% improvement)
     • Metric: Reward not collapsing due to balance penalty
  
  ✅ Priority 2: BUY/SELL Balance
     • Current: BUY 18%, SELL 66.85% (Diff: 48.85%)
     • Target: BUY 30-40%, SELL 30-40% (Diff: < 10%)
     • Metric: |BUY_ratio - SELL_ratio| < 0.1
  
  ✅ Priority 3: HOLD Action Increase
     • Current: 15.15%
     • Target: 20-30%
     • Metric: More stable trading behavior
  
  ✅ Priority 4: Continuous Action Distribution
     • Current: Mean -0.4968 (SELL biased)
     • Target: Mean closer to 0 (balanced)
     • Metric: Reduced negative skew
        """)
        
        # Implementation Notes
        recommendations.append("\n5️⃣ IMPLEMENTATION NOTES:")
        recommendations.append("-" * 80)
        recommendations.append("""
  • Penalties are cumulative: Each step, the balance penalty is applied
  • Current issue: -1000 * 0.4885 = -488.5 penalty EVERY STEP
  • With scale_200: -200 * similar_imbalance ≈ -100 penalty per step
  • This should allow positive PnL to emerge
  
  • Action bonuses provide positive incentive to diverse actions
  • SELL bias comes from strong negative continuous action distribution
  • Need to monitor if regime-specific targets help
        """)
        
        recommendations.append("\n" + "="*80)
        
        return "\n".join(recommendations)
    
    def create_visual_comparison(self):
        """
        パラメータ比較の可視化を作成
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SAC v444 Parameter Tuning Analysis', fontsize=16, fontweight='bold')
        
        # 1. Penalty Scale Comparison
        ax = axes[0, 0]
        configs = ['scale_200', 'scale_300', 'scale_500']
        penalty_scales = [200, 300, 500]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        ax.bar(configs, penalty_scales, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Balance Penalty Scale', fontsize=11, fontweight='bold')
        ax.set_title('1. Balance Penalty Scale Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(penalty_scales):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # 2. Action Bonuses Comparison
        ax = axes[0, 1]
        bonus_data = {
            'scale_200': [10.0, 5.0, 2.0],
            'scale_300': [15.0, 10.0, 3.0],
            'scale_500': [20.0, 15.0, 5.0],
        }
        
        x = np.arange(len(configs))
        width = 0.25
        
        buy_bonuses = [bonus_data[c][0] for c in configs]
        sell_bonuses = [bonus_data[c][1] for c in configs]
        hold_bonuses = [bonus_data[c][2] for c in configs]
        
        ax.bar(x - width, buy_bonuses, width, label='BUY', color='#3498db', edgecolor='black')
        ax.bar(x, sell_bonuses, width, label='SELL', color='#e74c3c', edgecolor='black')
        ax.bar(x + width, hold_bonuses, width, label='HOLD', color='#95a5a6', edgecolor='black')
        
        ax.set_ylabel('Bonus Amount', fontsize=11, fontweight='bold')
        ax.set_title('2. Action Bonuses Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Expected Reward Ranges
        ax = axes[1, 0]
        targets = {
            'scale_200': (-5000, -2000),
            'scale_300': (-4000, -1500),
            'scale_500': (-3000, -500),
        }
        
        y_pos = np.arange(len(targets))
        for i, (config, (low, high)) in enumerate(targets.items()):
            ax.barh(i, high - low, left=low, height=0.6, 
                   color=colors[i], alpha=0.7, edgecolor='black', linewidth=2)
            ax.text((low + high) / 2, i, f'{low} to {high}', 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(targets.keys()))
        ax.set_xlabel('Expected Mean Reward Range', fontsize=11, fontweight='bold')
        ax.set_title('3. Expected Reward Ranges', fontsize=12, fontweight='bold')
        ax.axvline(x=-9845, color='red', linestyle='--', linewidth=2, label='Current: -9845')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Target Action Ratios
        ax = axes[1, 1]
        action_targets = {
            'scale_200': {'buy': 0.325, 'sell': 0.325, 'hold': 0.25},
            'scale_300': {'buy': 0.375, 'sell': 0.375, 'hold': 0.275},
            'scale_500': {'buy': 0.425, 'sell': 0.425, 'hold': 0.30},
        }
        
        x_pos = np.arange(len(configs))
        buy_targets = [action_targets[c]['buy'] for c in configs]
        sell_targets = [action_targets[c]['sell'] for c in configs]
        hold_targets = [action_targets[c]['hold'] for c in configs]
        
        ax.bar(x_pos - width, buy_targets, width, label='BUY Target', color='#3498db', edgecolor='black')
        ax.bar(x_pos, sell_targets, width, label='SELL Target', color='#e74c3c', edgecolor='black')
        ax.bar(x_pos + width, hold_targets, width, label='HOLD Target', color='#95a5a6', edgecolor='black')
        
        # Current status
        ax.axhline(y=0.18, color='#3498db', linestyle=':', linewidth=2, alpha=0.7, label='Current BUY: 18%')
        ax.axhline(y=0.6685, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.7, label='Current SELL: 66.85%')
        
        ax.set_ylabel('Action Ratio', fontsize=11, fontweight='bold')
        ax.set_title('4. Target Action Ratios', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs)
        ax.set_ylim([0, 0.8])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = f"analysis/parameter_tuning_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {output_path}")
        
        plt.show()


def main():
    analyzer = ParameterAnalyzer()
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations()
    print(recommendations)
    
    # Save recommendations
    output_path = f"analysis/parameter_tuning_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(recommendations)
    print(f"\n💾 Recommendations saved to {output_path}")
    
    # Create visualization
    analyzer.create_visual_comparison()


if __name__ == "__main__":
    main()
